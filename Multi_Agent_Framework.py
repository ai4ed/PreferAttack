import json, time, argparse
from pathlib import Path
import math

from utils.pairwise_loader import load_pairwise_json, load_from_split_info

from utils.opt_utils import (
    adaptive_word_level_recombination,
    multi_granularity_hybrid,       # 如需word_level，切换这里
)
from utils.string_utils import load_conversation_template
from utils.vllm_judge import create_vllm_judge
from utils.data_types import PairwiseExample
from utils.rl_controller import RLController


def build_attacked_instruction(instruction: str, suffix: str) -> str:
    # 这里定义把后缀拼到指令的方式（与你现有模板保持一致），中间空格是为了分隔指令与后缀
    # 为什么要用空格将后缀和指令分开？因为有些指令可能没有以空格结尾，直接拼接会导致词语连在一起，影响模型理解。
    return instruction.rstrip() + " " + suffix


def extract_usage(resp) -> dict:
    usage = getattr(resp, "usage", None) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def add_usage(accumulator: dict, usage: dict) -> None:
    accumulator["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    accumulator["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    accumulator["total_tokens"] += int(usage.get("total_tokens", 0) or 0)


def merged_usage(base_usage: dict, extra_usage: dict) -> dict:
    return {
        "prompt_tokens": int(base_usage.get("prompt_tokens", 0) or 0) + int(extra_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(base_usage.get("completion_tokens", 0) or 0) + int(extra_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(base_usage.get("total_tokens", 0) or 0) + int(extra_usage.get("total_tokens", 0) or 0),
    }


def _example_field(ex, field_name: str, default=""):
    if isinstance(ex, dict):
        return ex.get(field_name, default)
    return getattr(ex, field_name, default)


def _example_id(ex, fallback):
    if isinstance(ex, dict):
        return ex.get("meta", {}).get("id", fallback)
    return getattr(ex, "question_id", fallback)


class ScoringAgent:
    def __init__(self, judge, args):
        self.judge = judge
        self.args = args

    def _build_candidate(self, ex, instruction, suffix, append_mode):
        if append_mode == 'instruction':
            attacked_instruction = build_attacked_instruction(instruction, suffix)
            resp_a = _example_field(ex, "response_a")
            resp_b = _example_field(ex, "response_b")
            attacked_repr = attacked_instruction
        elif append_mode == 'a':
            attacked_instruction = instruction
            resp_a = _example_field(ex, "response_a") + suffix
            resp_b = _example_field(ex, "response_b")
            attacked_repr = resp_a
        else:
            attacked_instruction = instruction
            resp_a = _example_field(ex, "response_a")
            resp_b = _example_field(ex, "response_b") + suffix
            attacked_repr = resp_b

        return PairwiseExample(
            question_id=str(_example_id(ex, "")),
            instruction=attacked_instruction,
            response_a=resp_a,
            response_b=resp_b,
            model_a=_example_field(ex, "model_a", ""),
            model_b=_example_field(ex, "model_b", ""),
        ), attacked_repr

    def score_candidates(self, candidates, *, ex, instruction, append_mode, base_choice, cache, stats):
        if not candidates:
            return []

        results_map = {}
        examples = []
        idxs_to_request = []
        attacked_list_to_request = []

        for i, suffix in enumerate(candidates):
            candidate_example, attacked_repr = self._build_candidate(ex, instruction, suffix, append_mode)
            if cache is not None and attacked_repr in cache:
                results_map[i] = cache[attacked_repr]
            else:
                idxs_to_request.append(i)
                attacked_list_to_request.append(attacked_repr)
                examples.append(candidate_example)

        if examples:
            try:
                batch_sz = min(self.args.batch_max_size, max(1, len(examples)))
                res = self.judge.judge_examples(examples, batch_size=batch_sz,
                                                original_preferences=[base_choice] * len(examples))
                if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], int):
                    judge_results, real_batches = res
                else:
                    judge_results = res
                    real_batches = math.ceil(len(examples) / float(batch_sz))
                for idx_pos, resp, attacked in zip(idxs_to_request, judge_results, attacked_list_to_request):
                    results_map[idx_pos] = resp
                    add_usage(stats["query_token_usage"], extract_usage(resp))
                    if cache is not None:
                        cache[attacked] = resp
                stats["api_calls"] += real_batches
                stats["candidates_evaluated"] += len(examples)
            except Exception:
                for idx_pos, example, attacked in zip(idxs_to_request, examples, attacked_list_to_request):
                    try:
                        single_resp = self.judge.judge_pairwise(example, original_preference=base_choice)
                        results_map[idx_pos] = single_resp
                        add_usage(stats["query_token_usage"], extract_usage(single_resp))
                        if cache is not None:
                            cache[attacked] = single_resp
                        stats["api_calls"] += 1
                        stats["candidates_evaluated"] += 1
                    except Exception:
                        from utils.data_types import JudgeResponse as _JR
                        results_map[idx_pos] = _JR(preference=None, confidence=0.5, raw_response=None)

        scores = []
        for i in range(len(candidates)):
            resp = results_map.get(i)
            if base_choice is None or resp is None or getattr(resp, 'preference', None) is None:
                scores.append(0.5)
                continue
            if resp.preference == base_choice:
                scores.append(float(resp.confidence))
            else:
                scores.append(float(1.0 - resp.confidence))
        return scores


class DecisionAgent:
    def __init__(self, args):
        self.args = args

    def choose_mode(self, params, search_mode):
        """选择攻击策略模式"""
        strategy_mode = None
        try:
            strategy_mode = params.get('strategy_mode') if isinstance(params, dict) else None
        except Exception:
            strategy_mode = None

        if strategy_mode and strategy_mode != 'auto':
            return strategy_mode
        return 'hybrid' if search_mode == 'hybrid' else 'word_level_only'

    def choose_params(self, *, diversity, patience_counter, avg_score, last_improvement, controller=None):
        """选择攻击过程中的参数：精英数、交叉率、变异率、topk等"""
        params = {
            'mutation': self.args.mutation,
            'crossover': self.args.crossover,
            'num_elites': max(1, int(self.args.batch_size * self.args.num_elites)),
            'word_dict_topk': int(self.args.word_dict_topk) if self.args.word_dict_topk is not None else 2000,
            'gpt_mutation_prob': self.args.gpt_mutation_prob if self.args.gpt_mutation_prob is not None else self.args.mutation,
        }

        action = None
        if controller is not None:
            state = {
                'improvement': last_improvement,
                'diversity': diversity,
                'patience': patience_counter,
                'avg_score': avg_score
            }
            action = controller.select_action(state)
            params = controller.apply_action(params, action)

        return params, action

    def decide_search_mode(self, *, first_success_recorded, patience_counter, search_mode, word_level_fallback_remaining):
        """决定搜索模式切换"""
        if first_success_recorded:
            return search_mode, word_level_fallback_remaining

        if patience_counter >= self.args.patience:
            if search_mode == 'hybrid' and word_level_fallback_remaining == 0:
                return 'word_level_fallback', 10
            elif search_mode == 'word_level_fallback' and word_level_fallback_remaining <= 0:
                return 'stop', 0

        return search_mode, word_level_fallback_remaining


class AttackAgent:
    def __init__(self, judge, args):
        self.judge = judge
        self.args = args

    def build_pairwise_example(self, ex, instruction, response_a=None, response_b=None):
        return PairwiseExample(
            question_id=str(_example_id(ex, "")),
            instruction=instruction,
            response_a=_example_field(ex, "response_a") if response_a is None else response_a,
            response_b=_example_field(ex, "response_b") if response_b is None else response_b,
            model_a=_example_field(ex, "model_a", ""),
            model_b=_example_field(ex, "model_b", ""),
        )

    def build_attacked_pairwise(self, ex, instruction, best_suffix, append_mode):
        if append_mode == 'instruction':
            attacked_instruction = build_attacked_instruction(instruction, best_suffix)
            final_resp_a = _example_field(ex, "response_a")
            final_resp_b = _example_field(ex, "response_b")
        elif append_mode == 'a':
            attacked_instruction = instruction
            final_resp_a = _example_field(ex, "response_a") + best_suffix
            final_resp_b = _example_field(ex, "response_b")
        else:
            attacked_instruction = instruction
            final_resp_a = _example_field(ex, "response_a")
            final_resp_b = _example_field(ex, "response_b") + best_suffix
        return self.build_pairwise_example(ex, attacked_instruction, final_resp_a, final_resp_b)

    def judge_pairwise(self, example, original_preference=None):
        return self.judge.judge_pairwise(example, original_preference=original_preference)

    def run_update(self, *, params, search_mode, population, score_list, word_dict, num_elites_used,
                   batch_size, crossover_used, mutation_used, topk_used, reference):
        """执行种群更新：选择、交叉、变异"""
        def do_hybrid():
            return multi_granularity_hybrid(
                word_dict=word_dict,
                control_suffixs=population,
                score_list=score_list,
                num_elites=num_elites_used,
                batch_size=batch_size,
                crossover=crossover_used,
                mutation=mutation_used,
                API_key=None,
                reference=reference,
                topk=topk_used,
            )

        def do_word_level():
            return adaptive_word_level_recombination(
                word_dict=word_dict,
                control_suffixs=population,
                score_list=score_list,
                num_elites=num_elites_used,
                batch_size=batch_size,
                crossover=crossover_used,
                mutation=mutation_used,
                API_key=None,
                reference=reference,
                topk=topk_used,
            )

        # 优先使用 params 中 RL 控制器指定的 strategy_mode
        strategy_mode = params.get('strategy_mode') if isinstance(params, dict) else None
        if strategy_mode and strategy_mode != 'auto':
            chosen_mode = strategy_mode
        else:
            chosen_mode = 'hybrid' if search_mode == 'hybrid' else 'word_level_only'

        if chosen_mode == 'hybrid_only' or chosen_mode == 'hybrid':
            return do_hybrid()
        if chosen_mode == 'word_level_only' or chosen_mode == 'word_level_fallback':
            return do_word_level()
        if chosen_mode == 'both_hybrid_then_word_level':
            pop_h, wd_h = do_hybrid()
            pop_g, wd_g = do_word_level()
            new_pop = []
            for i in range(max(len(pop_h), len(pop_g))):
                if i < len(pop_h):
                    new_pop.append(pop_h[i])
                if i < len(pop_g):
                    new_pop.append(pop_g[i])
            return new_pop[:batch_size], wd_g if wd_g else wd_h
        if chosen_mode == 'both_word_level_then_hybrid':
            pop_g, wd_g = do_word_level()
            pop_h, wd_h = do_hybrid()
            new_pop = []
            for i in range(max(len(pop_g), len(pop_h))):
                if i < len(pop_g):
                    new_pop.append(pop_g[i])
                if i < len(pop_h):
                    new_pop.append(pop_h[i])
            return new_pop[:batch_size], wd_h if wd_h else wd_g
        return do_hybrid()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model", type=str, default="llama2")
    ap.add_argument("--split_info", type=str, default=None,
                    help="可选：使用 split 描述文件（例如 data/split/alpaca_eval_split_info.json）")
    ap.add_argument("--data_json", type=str, default=None,
                    help="或者直接给 JSON 文件路径（与 --split_info 二选一）")
    ap.add_argument("--split", type=str, default="test", choices=["train","test"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=50)
            # build verification PairwiseExample according to append_mode or apply genome 
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_steps", type=int, default=100)
    ap.add_argument("--iter", type=int, default=5)
    ap.add_argument("--num_elites", type=float, default=0.05)
    ap.add_argument("--crossover", type=float, default=0.5)
    ap.add_argument("--mutation", type=float, default=0.01)
    ap.add_argument("--save_path", type=str, default="results/pairwise_ga.json")
    # controls for verification, batching and caching
    ap.add_argument("--verify_best_every", type=int, default=0,
                    help="If >0, verify the best suffix every N generations (0=never).\n"
                         "If --stop_on_first_success is set, this will be treated as 1.\n"
                         "If =0 表示禁用.")
    ap.add_argument("--stop_on_first_success", action='store_true',
                    help="Stop the inner search loop as soon as a flip is detected for this sample.")
    ap.add_argument("--batch_max_size", type=int, default=32,
                    help="Maximum batch size to pass to judge.judge_examples. Increase to reduce number of batches if judge supports it.")
    ap.add_argument("--use_cache", action='store_true', help="Enable simple per-sample cache of judge responses to avoid duplicate requests.")
    ap.add_argument("--append_to", type=str, default="instruction", choices=["instruction","a","b","target"],
                    help="Where to append the suffix: 'instruction' (default), 'a' (append to response_a), 'b' (append to response_b), or 'target' (append to the target side obtained by flipping baseline choice).")
    # early stopping and word_dict pruning
    ap.add_argument("--patience", type=int, default=10, help="early stopping patience (generations without improvement)")
    ap.add_argument("--word_dict_topk", type=int, default=2000, help="prune momentum word_dict to top-K entries (use -1 to disable)")
    # RL controller options
    ap.add_argument("--use_rl_controller", action='store_true', help="Enable the RL controller to steer GA hyperparameters")
    ap.add_argument("--rl_lr", type=float, default=0.1, help="RL learning rate (Q-learning)")
    ap.add_argument("--rl_gamma", type=float, default=0.9, help="RL discount factor")
    ap.add_argument("--rl_epsilon", type=float, default=0.2, help="RL epsilon for exploration")
    ap.add_argument("--rl_save_path", type=str, default=None, help="Optional path to save RL controller Q-table (JSON)")
    ap.add_argument("--gpt_mutation_prob", type=float, default=None, help="Explicit GPT mutation probability (if set overrides --mutation for GPT calls)")
    # argparse 增加参数：
    ap.add_argument("--qwen3_path", type=str, default=None, help="本地 Qwen3 模型路径，例如 /data/Qwen3-8B-Instruct")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto","bf16","fp16"])
    ap.add_argument("--use_attack_combiner", action='store_true', help="Enable GA over attack-method combinations")
    ap.add_argument("--genome_len", type=int, default=3, help="Number of attack steps per genome")
    ap.add_argument("--start_with_combiner", action='store_true', help="Start search with attack-combiner genomes instead of suffix word_level")
    ap.add_argument("--fallback_to_combiner_after", type=int, default=None, help="If >0 and no improvement for this many generations, switch from suffix-word_level to attack-combiner mode. Default=patience")
    ap.add_argument("--attack_evolve_steps", type=int, default=5, help="Number of GA generations to evolve attack-method genomes when falling back to combiner")
    ap.add_argument("--attack_eval_samples", type=int, default=8, help="Number of suffixes sampled to evaluate each attack genome (reduces cost)")
    # Token credit guidance for GA/word_level mutation
    ap.add_argument("--use_token_credit_guidance", action='store_true', help="Use causal token-credit to freeze key tokens and focus mutations on low-credit tokens")
    ap.add_argument("--credit_high_thresh", type=float, default=0.05, help="Threshold above which a token is treated as high-contribution and frozen")
    ap.add_argument("--credit_low_thresh", type=float, default=-0.01, help="Threshold below which a token is treated as low/negative contribution and prioritized for replacement")
    ap.add_argument("--freeze_high_prob", type=float, default=0.0, help="Replacement probability for high-credit tokens (0 disables replacements)")
    ap.add_argument("--boost_low_factor", type=float, default=2.0, help="Factor to boost replacement probability for low-credit tokens")
    # resume / append options
    ap.add_argument("--previous_results", type=str, default=None, help="Optional path to previous results JSON to resume from")
    ap.add_argument("--resume", action='store_true', help="If set along with --previous_results, only run remaining samples not present in previous results and append new records to it")
    ap.add_argument("--seed", type=int, default=None,
                    help="Global RNG seed (random / numpy / torch / vLLM). Default None preserves the original non-deterministic behavior; set an int to make runs reproducible.")
    args = ap.parse_args()

    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        try:
            import torch as _torch
            _torch.manual_seed(args.seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(args.seed)
        except ImportError:
            pass
        print(f"[seed] Global RNGs seeded with seed={args.seed} (random / numpy / torch)")

    template_name = "qwen3" if args.qwen3_path else args.model

    # Pass dtype through to vLLM judge (helps reduce GPU memory usage: bf16/fp16)
    judge = create_vllm_judge(model_path=args.qwen3_path or "Qwen/Qwen3-8B", dtype=args.dtype, seed=args.seed)
    # model, tokenizer = load_qwen3_local(args.qwen3_path, device=args.device, dtype=args.dtype)
    scoring_agent = ScoringAgent(judge, args)
    decision_agent = DecisionAgent(args)
    attack_agent = AttackAgent(judge, args)

    # 2) 读取数据
    if args.split_info:
        data, used_path = load_from_split_info(args.split_info, split=args.split)
    elif args.data_json:
        data = load_pairwise_json(args.data_json)
        used_path = args.data_json
    else:
        raise ValueError("Provide --split_info or --data_json")

    end = min(len(data), args.start + args.num_samples)

    # Determine indices to process. If resuming from previous results, skip already-processed ids.
    prev_ids = set()
    results = None
    if args.previous_results and args.resume:
        try:
            with open(args.previous_results, "r", encoding="utf-8") as pf:
                prev = json.load(pf)
            prev_records = prev.get("records", []) if isinstance(prev, dict) else []
            # Normalize ids to strings for robust comparison
            prev_ids = set(str(r.get("id")) for r in prev_records if r.get("id") is not None)
            results = prev if isinstance(prev, dict) else {"meta": {}, "records": []}
            print(f"Loaded {len(prev_records)} previous records from {args.previous_results}; will skip these when resuming.")
        except Exception as e:
            print(f"Warning: failed to load previous results {args.previous_results}: {e}")
            prev_ids = set()
            results = None

    # Build list of (original_index, example) to process
    indices = list(range(args.start, end))
    todo_items = []
    for i in indices:
        ex = data[i]
        ex_id = ex.get("meta", {}).get("id", i) if isinstance(ex, dict) else getattr(ex, "question_id", i)
        if prev_ids and str(ex_id) in prev_ids:
            continue
        todo_items.append((i, ex))

    if not todo_items and results is not None:
        # Nothing to do; write out (or copy) existing results to save_path and exit gracefully
        print("No remaining samples to process; copying previous results to save_path and exiting.")
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved copied results to: {args.save_path}")
        except Exception as e:
            print(f"Warning: failed to write merged results to {args.save_path}: {e}")
        return

    # If not resuming, or failed to load previous, initialize empty results
    if results is None:
        subset_len = len(todo_items)
        results = {
            "meta": {
                "source": used_path,
                "model": args.qwen3_path,
                "search": "GA",
                "start": args.start,
                "num_samples": subset_len,
                "time": time.asctime()
            },
            "records": []
        }
    else:
        # when resuming, update meta num_samples/time later after appending
        if "meta" not in results:
            results["meta"] = {"source": used_path, "model": args.model, "search": "GA", "time": time.asctime()}

    # 全局 wall-clock 起点，用于吞吐率 / 成本统计
    overall_t0 = time.time()
    # 逐样本
    for idx, ex in todo_items:
        instruction, a, b = ex["instruction"], ex["response_a"], ex["response_b"]

        query_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 2.1 baseline 评判（未攻击）
        # `ex` 来自 JSON/loader，类型是 dict；vLLMJudge.judge_pairwise 期望一个 PairwiseExample
        pa = attack_agent.build_pairwise_example(ex, instruction, a, b)
        base_resp = attack_agent.judge_pairwise(pa)
        base_choice = base_resp.preference
        base_conf = base_resp.confidence
        base_text = base_resp.raw_response
        baseline_token_usage = extract_usage(base_resp)

        # 依据用户选择，如果选择 'target' 则对每个样本把 base_choice 翻转得到目标侧（0->A,1->B），
        # 否则直接使用 args.append_to 作为本样本的 append 模式。
        target_pref = 1 - int(base_choice)
        target_pref = 'a' if target_pref == 0 else 'b'
        if args.append_to == 'target':
            if base_choice is None:
                append_mode = 'instruction'  # 无 baseline 时退回到追加到 instruction
            else:
                target_pref = 1 - int(base_choice)
                append_mode = 'a' if target_pref == 0 else 'b'
        else:
            append_mode = args.append_to

        api_calls = 0
        # candidates_evaluated是用来统计该样本评估过的候选后缀数量
        candidates_evaluated = 0
        # first_success_recorded 标记是否已经记录到首次成功翻转
        first_success_recorded = False
        # queries_until_success 记录首次成功翻转时的信息
        queries_until_success = None

        # per-sample observed strategies that succeeded (for recording)
        succeeded_strategy_types = []

        # controls for verification frequency and caching per sample
        verify_every = args.verify_best_every
        if args.stop_on_first_success:
            verify_every = 1
            print("Note: --stop_on_first_success is set; will verify every generation and stop immediately upon detecting a flip.")
        cache = {} if args.use_cache else None
        # last-best tracking to avoid final re-eval
        last_best_suffix = None
        last_best_score = None

        sample_stats = {
            "api_calls": 0,
            "candidates_evaluated": 0,
            "query_token_usage": query_token_usage,
        }

        def score_function(candidates):
            nonlocal sample_stats, cache
            return scoring_agent.score_candidates(
                candidates,
                ex=ex,
                instruction=instruction,
                append_mode=append_mode,
                base_choice=base_choice,
                cache=cache,
                stats=sample_stats,
            )

        api_calls = sample_stats["api_calls"]
        candidates_evaluated = sample_stats["candidates_evaluated"]

        t0 = time.time()
        first_success_t = None  # 记录本样本首次翻转的时间戳
        # 初始化种群
        import torch as _torch
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            reference = _torch.load('assets/prompt_group.pth', map_location='cpu')
        # 默认先使用后缀 word_level；攻击方法（combiner）种群仅作为可选的组合生成器
        # Keep suffix population (from prompt_group.pth) as the primary population.
        population = reference[:args.batch_size]
        # determine fallback threshold (None -> fallback after patience)
        fallback_after = args.fallback_to_combiner_after if args.fallback_to_combiner_after is not None else args.patience
        num_elites_cnt = max(1, int(args.batch_size * args.num_elites))
        history = []
        # RL controller: optional module to steer GA hyperparameters
        controller = None
        if args.use_rl_controller:
            controller = RLController(state_dim=4, lr=args.rl_lr, gamma=args.rl_gamma, epsilon=args.rl_epsilon, device=args.device)
        # current_params stores the mutable hyperparameters used each generation
        current_params = {
            'mutation': args.mutation,
            'crossover': args.crossover,
            'num_elites': int(num_elites_cnt),
            'word_dict_topk': int(args.word_dict_topk) if args.word_dict_topk is not None else 2000,
            'gpt_mutation_prob': args.gpt_mutation_prob if args.gpt_mutation_prob is not None else args.mutation,
        }
        last_improvement = 0.0
        last_action = None
        last_state = None
        word_dict = {}
        prefix_string_init = None
        # 用于存放当前样本的中间快照（每 10 轮保存一次）
        partial_snapshots = []

        # early stopping state: lower score is better
        best_so_far = float('inf')
        patience_counter = 0
        # count consecutive word-level update failures or no-change events
        word_level_no_change_counter = 0
        # search mode: start with hybrid as before; possible values: 'hybrid', 'word_level_fallback'
        search_mode = 'hybrid'
        word_level_fallback_remaining = 0
        # track last synthesis info from judge.synthesize_from_minmax
        last_synth_info = None
        for _step in range(args.num_steps):
            # 循环内部的下方对population进行了更新（选择交叉变异），这里的population一直在变。计算适应度
            # 在这个函数中构建攻击串，并进行评分
            # capture previous best for reward computation
            sample_stats["api_calls"] = api_calls
            sample_stats["candidates_evaluated"] = candidates_evaluated
            prev_best = best_so_far
            score_list = score_function(population)
            api_calls = sample_stats["api_calls"]
            candidates_evaluated = sample_stats["candidates_evaluated"]
            try:
                _best_idx = min(range(len(score_list)), key=lambda k: score_list[k])
                # 这行代码是记录当前代的最优分数，分数越小越好
                current_best = score_list[_best_idx]
                history.append(current_best)
                # remember last best to avoid re-evaluation at the end
                last_best_suffix = population[_best_idx]
                last_best_score = current_best
            except Exception:
                current_best = None
                history.append(None)

            # compute avg_score for state (ignore None entries)
            try:
                valid_scores = [s for s in score_list if s is not None]
                avg_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
            except Exception:
                avg_score = 0.0

            # If RL controller is enabled, compute reward (flip-primary) and update agent
            try:
                if controller is not None and last_action is not None and last_state is not None:
                    # default improvement term
                    if current_best is not None and prev_best is not None and math.isfinite(prev_best):
                        improv = float(prev_best - current_best)
                         # 简单做个截断 + 归一化：[-1, 1]
                        improv = max(-1.0, min(1.0, improv))
                    else:
                        improv = 0.0

                    # optionally verify best candidate to detect flip (this costs 1 judge call)
                    flipped = False
                    try:
                        candidate_suffix = last_best_suffix
                        if candidate_suffix:
                            # suffix mode
                            pa_check = attack_agent.build_attacked_pairwise(ex, instruction, candidate_suffix, append_mode)
                            try:
                                check_resp = attack_agent.judge_pairwise(pa_check, original_preference=base_choice)
                                api_calls += 1
                                candidates_evaluated += 1
                                add_usage(query_token_usage, extract_usage(check_resp))
                                if (base_choice is not None and check_resp is not None and getattr(check_resp, 'preference', None) is not None
                                        and check_resp.preference != base_choice):
                                    flipped = True
                                    first_success_recorded = True
                                    if first_success_t is None:
                                        first_success_t = time.time()
                                    # If the verified candidate was produced by the synthesizer, register its strategy
                                    # Also record the verifier's returned choice/raw response so we can avoid re-checking later
                                    queries_until_success = {
                                        "generation": _step + 1,
                                        "api_calls": api_calls,
                                        "candidates_evaluated": candidates_evaluated,
                                        "prompt_tokens": query_token_usage["prompt_tokens"],
                                        "completion_tokens": query_token_usage["completion_tokens"],
                                        "total_tokens": query_token_usage["total_tokens"],
                                        "prompt_tokens_including_baseline": baseline_token_usage["prompt_tokens"] + query_token_usage["prompt_tokens"],
                                        "completion_tokens_including_baseline": baseline_token_usage["completion_tokens"] + query_token_usage["completion_tokens"],
                                        "total_tokens_including_baseline": baseline_token_usage["total_tokens"] + query_token_usage["total_tokens"],
                                        "best_suffix": candidate_suffix,
                                        "confidence": getattr(check_resp, "confidence", None),
                                        "new_choice": getattr(check_resp, "preference", None),
                                        "raw": getattr(check_resp, "raw_response", None),
                                        "wall_clock_sec": round(first_success_t - t0, 3) if first_success_t else None,
                                    }
                                # 一旦找到第一个攻击成功的后缀就立即进行早停
                                if args.stop_on_first_success:
                                    # stop immediately
                                    break
                            except Exception:
                                flipped = False
                    except Exception:
                        flipped = False
                    # 高层：flip 是否成功
                    r_flip = 1.0 if flipped else 0.0

                    # 中层：逐步逼近 flip 的进展（改得更好就给正奖励）
                    # 可以直接用 improv，也可以用一个分段函数，比如：
                    if improv > 0:
                        r_progress = 0.5   # 有进步但没成功flip，也值得给固定奖励
                    elif improv < 0:
                        r_progress = -0.2  # 变差了，略微惩罚
                    else:
                        r_progress = 0.0

                    # 最终层级奖励：高层目标优先，其次进展信号
                    reward = r_flip * 1.0 + r_progress

                    # next state uses diversity after this generation and current patience
                    try:
                        uniq = {tuple(p) if isinstance(p, (list, tuple)) else p for p in population}
                        next_diversity = float(len(uniq)) / max(1, len(population))
                    except Exception:
                        next_diversity = 0.0
                    next_state = {'improvement': improv, 'diversity': next_diversity, 'patience': patience_counter, 'avg_score': avg_score}
                    controller.update(last_state, last_action, reward, next_state)
                    # commit params returned by last action as current_params for next selection
                    try:
                        current_params = params
                    except Exception:
                        pass
                    # record last improvement for next state's selection
                    last_improvement = improv
            except Exception:
                # never fail the GA loop because of controller issues
                pass
            # RL-steered parameter selection (optional)
            # 使用 DecisionAgent 选择参数
            try:
                uniq = {tuple(p) if isinstance(p, (list, tuple)) else p for p in population}
                diversity = float(len(uniq)) / max(1, len(population))
            except Exception:
                diversity = 0.0

            params, action = decision_agent.choose_params(
                diversity=diversity,
                patience_counter=patience_counter,
                avg_score=avg_score,
                last_improvement=last_improvement,
                controller=controller
            )

            # 解析参数
            mutation_used = params.get('gpt_mutation_prob', args.mutation)
            crossover_used = params.get('crossover', args.crossover)
            num_elites_used = int(params.get('num_elites', num_elites_cnt))
            topk_used = int(params.get('word_dict_topk', args.word_dict_topk))

            # 保存当前参数供 RL 控制器后续使用
            current_params = params
            if controller is not None:
                state = {'improvement': last_improvement, 'diversity': diversity, 'patience': patience_counter, 'avg_score': avg_score}
                # 保存选择的 action，供下一轮 RL 更新使用
                last_action = action if action is not None else None
                last_state = state

            try:
                new_pop, new_word_dict = attack_agent.run_update(
                    params=params,
                    search_mode=search_mode,
                    population=population,
                    score_list=score_list,
                    word_dict=word_dict,
                    num_elites_used=num_elites_used,
                    batch_size=args.batch_size,
                    crossover_used=crossover_used,
                    mutation_used=mutation_used,
                    topk_used=topk_used,
                    reference=reference,
                )

                # detect no-change (conservative equality)
                try:
                    no_change = new_pop == population
                except Exception:
                    no_change = False
                population = new_pop
                word_dict = new_word_dict
                if no_change:
                    word_level_no_change_counter += 1
                else:
                    word_level_no_change_counter = 0

                # If we are running word-level fallback, decrement remaining budget
                if search_mode == 'word_level_fallback' and word_level_fallback_remaining > 0:
                    word_level_fallback_remaining -= 1
                    # If we obtained success during fallback, revert to hybrid search
                    if first_success_recorded:
                        search_mode = 'hybrid'
                        word_level_fallback_remaining = 0
                        patience_counter = 0
                    # if fallback budget exhausted and still no success, stop outer loop later
            except Exception as e:
                print(f"Warning: update failed: {e}")
                # treat as a failed update
                word_level_no_change_counter += 1
            # ------------------ early stopping ------------------
            # 如果本代有可比较的最优分数，则判断是否有改进
            if current_best is not None:
                # 小幅度容差以避免浮点震荡被误判为改进
                if current_best + 1e-12 < best_so_far:
                    best_so_far = current_best
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                # 若当前代无法计算分数，则视为一次无改进
                patience_counter += 1

            if patience_counter >= args.patience:
                # Only early-stop when we have observed a successful attack AND
                # we've seen `patience` generations without improvement.
                if first_success_recorded:
                    # We previously verified a successful suffix; accept it as final
                    # and move on to the next sample. Ensure the recorded suffix is
                    # preserved as the best-known suffix for this sample.
                    try:
                        if queries_until_success and queries_until_success.get("best_suffix"):
                            last_best_suffix = queries_until_success.get("best_suffix")
                            # Prefer stored confidence if available
                            last_best_score = queries_until_success.get("confidence", last_best_score)
                    except Exception:
                        pass
                    print(f"Early stopping: recorded successful suffix accepted (step={_step}). Moving to next sample.")
                    break
                else:
                    # If we're running hybrid search, switch to word-level for a short fallback budget
                    if search_mode == 'hybrid' and word_level_fallback_remaining == 0:
                        word_level_fallback_remaining = 10
                        search_mode = 'word_level_fallback'
                        patience_counter = 0
                        print(f"No success after {args.patience} generations in hybrid; switching to word-level fallback for {word_level_fallback_remaining} generations.")
                    elif search_mode == 'word_level_fallback':
                        # fallback already used and budget exhausted -> stop
                        if word_level_fallback_remaining <= 0:
                            print(f"Early stopping triggered: word-level fallback exhausted without success (step={_step}).")
                            break
                        
                        # otherwise continue until fallback budget exhausted
                    else:
                        # other modes: be conservative and stop
                        print(f"Early stopping triggered: no improvement for {args.patience} generations (step={_step}).")
                        break

            # 每 10 轮保存一次中间结果并打印 checkpoint 信息
            if (_step + 1) % 10 == 0:
                try:
                    _best_idx_loc = min(range(len(score_list)), key=lambda k: score_list[k])
                    cur_best_suffix = population[_best_idx_loc]
                    cur_best_score = score_list[_best_idx_loc]
                except Exception:
                    cur_best_suffix = ""
                    cur_best_score = None

                snapshot = {
                    "step": _step,
                    "best_suffix": cur_best_suffix,
                    "best_score": cur_best_score,
                    "elapsed_sec": round(time.time() - t0, 2),
                    "word_dict_len": len(word_dict),
                    "api_calls": api_calls,
                    "candidates_evaluated": candidates_evaluated,
                    "queries_until_success": queries_until_success
                }
                partial_snapshots.append(snapshot)
                # 打印简洁信息，便于观察进度
                try:
                    if isinstance(cur_best_suffix, (list, tuple)):
                        srepr = ",".join(map(str, cur_best_suffix))
                    else:
                        srepr = str(cur_best_suffix)
                    sshort = (srepr[:80] + '...') if len(srepr) > 80 else srepr
                except Exception:
                    sshort = ""
                print(f"[{idx}] step {_step} checkpoint: best_score={cur_best_score} word_dict_len={len(word_dict)} suffix='{sshort}'")

                # 写入 checkpoint 文件（与最终 save_path 同目录）
                checkpoint = {"meta": results["meta"], "records": results["records"],
                              "current_partial": {"id": ex.get("meta", {}).get("id", idx), "intermediate": partial_snapshots}}
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(args.save_path + ".checkpoint.json", "w", encoding="utf-8") as f:
                        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Warning: failed to write checkpoint: {e}")
            # 每一轮结束时打印轮次信息，便于实时监控
            print(f"[{idx}] generation {_step+1}/{args.num_steps} finished — best={current_best} patience={patience_counter} word_dict_len={len(word_dict)}")
            # 末轮从种群中选最优后缀（重用最后一代记录，避免再一次完整评估）
            # If we recorded a verified successful suffix earlier, prefer that for final evaluation
            if first_success_recorded and queries_until_success and queries_until_success.get("best_suffix"):
                best_suffix = queries_until_success.get("best_suffix")
            else:
                best_suffix = last_best_suffix if last_best_suffix is not None else (population[0] if population else "")
            best_score = last_best_score
            t1 = time.time()

        # If we previously verified a successful suffix during search, prefer the recorded
        # verification result and skip a second judge call (avoids re-check flipping back).
        if first_success_recorded and queries_until_success and queries_until_success.get("new_choice") is not None:
            # Use recorded verified values from queries_until_success
            best_suffix = queries_until_success.get("best_suffix", best_suffix)
            new_choice = queries_until_success.get("new_choice")
            new_conf = queries_until_success.get("confidence")
            new_text = queries_until_success.get("raw")
            print(f"[{idx}] Final evaluation (from first_success_recorded): base_choice={base_choice} new_choice={new_choice}")
            is_success = (base_choice is not None and new_choice is not None and new_choice != base_choice)
        else:
            # 构造最终用于判定的被攻击输入；根据 --append_to 决定把后缀追加到 instruction / response_a / response_b
            pa_attacked = attack_agent.build_attacked_pairwise(ex, instruction, best_suffix, append_mode)
            new_resp = attack_agent.judge_pairwise(pa_attacked, original_preference=base_choice)
            api_calls += 1
            candidates_evaluated += 1
            add_usage(query_token_usage, extract_usage(new_resp))
            new_choice = new_resp.preference
            new_conf = new_resp.confidence
            new_text = new_resp.raw_response
            # 检查是否攻击成功的判断,早停时，就会跳出循环执行此处代码
            print(f"[{idx}] Final evaluation: base_choice={base_choice} new_choice={new_choice}")
            is_success = (base_choice is not None and new_choice is not None and new_choice != base_choice)


        results["records"].append({
            "id": ex.get("meta", {}).get("id", idx),
            "instruction": instruction,
            "response_a": a,
            "response_b": b,
            "baseline": {"choice": base_choice, "confidence": base_conf, "raw": base_text, "token_usage": baseline_token_usage},
            "attack": {
                "best_suffix": best_suffix,
                "new_choice": new_choice,
                "new_confidence": new_conf,
                "raw": new_text,
                "best_score": best_score,
                "runtime_sec": round(t1 - t0, 3),
                "succeeded_strategy_types": succeeded_strategy_types,
                "tts_sec": round(first_success_t - t0, 3) if first_success_t else None,
                # "token_credit": token_credit
            },
            "success": bool(is_success),
            "queries": {
                "api_calls": api_calls,
                "candidates_evaluated": candidates_evaluated,
                "prompt_tokens": query_token_usage["prompt_tokens"],
                "completion_tokens": query_token_usage["completion_tokens"],
                "total_tokens": query_token_usage["total_tokens"],
                "prompt_tokens_including_baseline": baseline_token_usage["prompt_tokens"] + query_token_usage["prompt_tokens"],
                "completion_tokens_including_baseline": baseline_token_usage["completion_tokens"] + query_token_usage["completion_tokens"],
                "total_tokens_including_baseline": baseline_token_usage["total_tokens"] + query_token_usage["total_tokens"],
            },
            "queries_until_success": queries_until_success
        })

        print(f"[{idx}] base={base_choice} -> new={new_choice}  "
              f"success={is_success}  suffix='{best_suffix[:40]}...'")

        # Optionally persist RL controller Q-table after this sample
        if controller is not None and args.rl_save_path:
            try:
                controller.save(args.rl_save_path)
            except Exception as e:
                print(f"Warning: failed to save RL controller to {args.rl_save_path}: {e}")

    results["meta"]["total_runtime_sec"] = round(time.time() - overall_t0, 3)
    # Update meta to reflect actual number of records when resuming/merging
    try:
        results["meta"]["num_samples"] = len(results.get("records", []))
        results["meta"]["time"] = time.asctime()
    except Exception:
        pass
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {args.save_path}")
    # --- 统计汇总：平均 query 次数与成功样本的 query 次数 ---
    try:
        recs = results.get("records", [])
        total = len(recs)
        total_api_calls = sum(r.get("queries", {}).get("api_calls", 0) for r in recs)
        total_candidates = sum(r.get("queries", {}).get("candidates_evaluated", 0) for r in recs)
        total_prompt_tokens = sum(r.get("queries", {}).get("prompt_tokens", 0) for r in recs)
        total_completion_tokens = sum(r.get("queries", {}).get("completion_tokens", 0) for r in recs)
        total_tokens = sum(r.get("queries", {}).get("total_tokens", 0) for r in recs)
        avg_api_calls = float(total_api_calls) / total if total else 0.0
        avg_candidates = float(total_candidates) / total if total else 0.0
        avg_total_tokens = float(total_tokens) / total if total else 0.0

        succ_recs = [r for r in recs if r.get("success")]
        succ_cnt = len(succ_recs)
        avg_api_calls_success = (float(sum(r.get("queries", {}).get("api_calls", 0) for r in succ_recs)) / succ_cnt) if succ_cnt else 0.0
        avg_candidates_success = (float(sum(r.get("queries", {}).get("candidates_evaluated", 0) for r in succ_recs)) / succ_cnt) if succ_cnt else 0.0
        avg_total_tokens_success = (float(sum(r.get("queries", {}).get("total_tokens", 0) for r in succ_recs)) / succ_cnt) if succ_cnt else 0.0

        total_runtime = results.get("meta", {}).get("total_runtime_sec") or 0.0
        throughput = (float(total) / total_runtime) if total_runtime else 0.0
        cost_success = (float(total_runtime) / succ_cnt) if succ_cnt and total_runtime else None

        # 首次成功时的 query 统计（如果有记录）
        until_success_list = [r.get("queries_until_success") for r in recs if r.get("queries_until_success")]
        until_count = len(until_success_list)
        avg_api_calls_until_success = float(sum(x.get("api_calls", 0) for x in until_success_list)) / until_count if until_count else 0.0
        avg_candidates_until_success = float(sum(x.get("candidates_evaluated", 0) for x in until_success_list)) / until_count if until_count else 0.0
        total_tokens_until_success = sum(x.get("total_tokens", 0) for x in until_success_list)
        avg_total_tokens_until_success = float(total_tokens_until_success) / until_count if until_count else 0.0
        total_tokens_until_success_with_baseline = sum(x.get("total_tokens_including_baseline", 0) for x in until_success_list)
        avg_total_tokens_until_success_with_baseline = float(total_tokens_until_success_with_baseline) / until_count if until_count else 0.0

        tts_list = [r.get("attack", {}).get("tts_sec") for r in succ_recs if r.get("attack", {}).get("tts_sec") is not None]
        tts_count = len(tts_list)
        avg_tts_success = (float(sum(tts_list)) / tts_count) if tts_count else None

        results["meta"]["aggregate_stats"] = {
            "total_samples": total,
            "total_successes": succ_cnt,
            "success_rate": (float(succ_cnt) / total) if total else 0.0,
            "avg_api_calls_per_sample": avg_api_calls,
            "avg_candidates_evaluated_per_sample": avg_candidates,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_total_tokens_per_sample": avg_total_tokens,
            "avg_api_calls_per_success_sample": avg_api_calls_success,
            "avg_candidates_per_success_sample": avg_candidates_success,
            "avg_total_tokens_per_success_sample": avg_total_tokens_success,
            "avg_api_calls_until_first_success": avg_api_calls_until_success,
            "avg_candidates_until_first_success": avg_candidates_until_success,
            "total_tokens_until_first_success": total_tokens_until_success,
            "avg_total_tokens_until_first_success": avg_total_tokens_until_success,
            "total_tokens_until_first_success_including_baseline": total_tokens_until_success_with_baseline,
            "avg_total_tokens_until_first_success_including_baseline": avg_total_tokens_until_success_with_baseline,
            "count_with_until_success_info": until_count,
            "avg_tts_success": avg_tts_success,
            "count_with_tts_info": tts_count,
            "throughput_samples_per_sec": throughput,
            "cost_success_sec_per_success": cost_success
        }

        # 将带汇总信息的结果覆盖存盘
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 打印简洁摘要
        print("\n=== Aggregate Summary ===")
        print(f"Samples: {total}  Successes: {succ_cnt}  SuccessRate: {results['meta']['aggregate_stats']['success_rate']:.3f}")
        print(f"Avg API calls (all): {avg_api_calls:.2f}, (successes): {avg_api_calls_success:.2f}")
        print(f"Avg candidates eval (all): {avg_candidates:.2f}, (successes): {avg_candidates_success:.2f}")
        print(f"Total tokens (search/final only): {total_tokens}  Avg/sample: {avg_total_tokens:.2f}  Avg/success: {avg_total_tokens_success:.2f}")
        if until_count:
            print(f"Avg API calls until first success (count={until_count}): {avg_api_calls_until_success:.2f}")
            print(f"Total tokens until first success: {total_tokens_until_success}  Avg: {avg_total_tokens_until_success:.2f}")
            print(f"Total tokens until first success (incl. baseline): {total_tokens_until_success_with_baseline}  Avg: {avg_total_tokens_until_success_with_baseline:.2f}")
        if tts_count:
            print(f"Avg TTS for successful samples (count={tts_count}): {avg_tts_success:.2f}s")
        if total_runtime:
            print(f"Throughput: {throughput:.3f} samples/sec; Cost_success: {cost_success if cost_success is not None else 'N/A'} sec/success")
    except Exception as e:
        print(f"Warning: failed to compute aggregate stats: {e}")

if __name__ == "__main__":
    main()
