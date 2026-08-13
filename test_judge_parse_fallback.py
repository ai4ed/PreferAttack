"""
独立脚本: 测试 MiniCPM5-1B 作为 vllm judge 时, 输出无法被解析规则命中、
从而落入 c_min=0.3 兜底分支的比例。

参考: utils/vllm_judge.py::VLLMJudge._parse_response
  - 5 条优先级规则 (conf = 0.95 / 0.9 / 0.85 / 0.8 / 0.7)
  - 全不命中 → (random.randint(0,1), 0.3)  # 即 c_min

本脚本只做"判别器输出不规则率"统计, 不参与攻击/防御流程, 复用 VLLMJudge
本身的 _parse_response 以保证口径一致。
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import List, Tuple

# 复用项目内现成的解析器与 prompt 模板
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_types import PairwiseExample
from utils.vllm_judge import VLLMJudge, create_vllm_judge, get_judge_prompt  # noqa: E402


# 与 VLLMJudge._parse_response 内部规则完全一致, 这里再写一份只是为了
# 在不依赖随机数 (兜底分支返回 random.randint) 的情况下, 独立判断"是否命中"
# 某条规则, 并取出对应的置信度。
MATCH_RULES: List[Tuple[List[str], List[str], float]] = [
    (["^output\\s*\\(a\\)\\s*$", "^output\\s*\\(a\\)", "output\\s*\\(a\\)\\s*$"],
     ["^output\\s*\\(b\\)\\s*$", "^output\\s*\\(b\\)", "output\\s*\\(b\\)\\s*$"], 0.95),
    (["\"output\\s*\\(a\\)\"", "'output\\s*\\(a\\)'"],
     ["\"output\\s*\\(b\\)\"", "'output\\s*\\(b\\)'"], 0.9),
    (["^\\(a\\)\\s*$", "^\\(a\\)", "\\(a\\)\\s*$"],
     ["^\\(b\\)\\s*$", "^\\(b\\)", "\\(b\\)\\s*$"], 0.85),
    (["^a\\s*$", "^a\\s", "\\sa\\s*$"],
     ["^b\\s*$", "^b\\s", "\\sb\\s*$"], 0.8),
    (["response a", "a is better", "prefer a", "choose a"],
     ["response b", "b is better", "prefer b", "choose b"], 0.7),
]

C_MIN = 0.3


def classify_parse(text: str, prompt: str) -> Tuple[str, float]:
    """返回 (类别名, 命中置信度)。

    类别名:
      - "rule_0.95" / "rule_0.9" / "rule_0.85" / "rule_0.8" / "rule_0.7"
        表示命中对应优先级规则;
      - "c_min_fallback" 表示所有规则都不命中 → 兜底分支。
    """
    gen_text = text.replace(prompt, "").strip().lower()
    for a_pats, b_pats, conf in MATCH_RULES:
        a_match = any(re.search(p, gen_text) for p in a_pats)
        b_match = any(re.search(p, gen_text) for p in b_pats)
        if (a_match or b_match) and not (a_match and b_match):
            return f"rule_{conf}", conf
    return "c_min_fallback", C_MIN


def load_examples(json_path: str, limit: int) -> List[PairwiseExample]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples: List[PairwiseExample] = []
    for item in raw[:limit]:
        examples.append(
            PairwiseExample(
                question_id=item.get("question_id", ""),
                instruction=item.get("instruction", ""),
                response_a=item.get("response_a", ""),
                response_b=item.get("response_b", ""),
                model_a=item.get("model_a", ""),
                model_b=item.get("model_b", ""),
            )
        )
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/MiniCPM5-1B")
    ap.add_argument("--data_path", default="data/split/alpaca_eval_test.json")
    ap.add_argument("--num_samples", type=int, default=200,
                    help="测试样本数 (默认 200; 设 -1 表示用全部)")
    ap.add_argument("--max_new_tokens", type=int, default=16,
                    help="和 VLLMJudge 默认 max_tokens 保持一致")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=23872)
    ap.add_argument("--dump_path", default="results/judge_parse_fallback_Minicpm5_1B.json",
                    help="把每条原始生成 + 分类结果落盘到该 JSON, 便于事后查看不规则样本")
    args = ap.parse_args()

    # vllm 相关环境变量 (和 run_pairwise_qwen3_all.sh 对齐)
    os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", str(args.gpu_memory_utilization))
    os.environ.setdefault("VLLM_MAX_MODEL_LEN", str(args.max_model_len))
    os.environ.setdefault("VLLM_SWAP_SPACE_GB", "8")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    print(f"[info] model_path = {args.model_path}")
    print(f"[info] data_path  = {args.data_path}")
    print(f"[info] num_samples = {args.num_samples}")

    # ---- 1. 加载数据 ----
    examples = load_examples(args.data_path, args.num_samples if args.num_samples > 0 else 1 << 30)
    print(f"[info] loaded {len(examples)} examples")

    # ---- 2. 起 vllm judge ----
    judge: VLLMJudge = create_vllm_judge(
        model_path=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # ---- 3. 批量推理 (复用 judge_examples, 口径与正式攻击 pipeline 一致) ----
    print(f"[info] running judge on {len(examples)} examples ...")
    responses = judge.judge_examples(
        examples,
        batch_size=16,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=False,
    )

    # ---- 4. 对每条 raw_response 重新跑解析规则, 统计回退率 ----
    cat_counter = Counter()
    conf_counter = Counter()
    fallback_records = []
    all_records = []

    for ex, resp in zip(examples, responses):
        prompt = get_judge_prompt(ex)
        cat, conf = classify_parse(resp.raw_response or "", prompt)
        cat_counter[cat] += 1
        conf_counter[conf] += 1
        rec = {
            "question_id": ex.question_id,
            "category": cat,
            "confidence": conf,
            "preference": resp.preference,
            "raw_response": resp.raw_response,
            "raw_response_len": len(resp.raw_response or ""),
        }
        all_records.append(rec)
        if cat == "c_min_fallback":
            fallback_records.append(rec)

    total = len(responses)
    fallback_n = cat_counter["c_min_fallback"]
    fallback_rate = fallback_n / total if total else 0.0

    print("\n========== Summary ==========")
    print(f"Total samples        : {total}")
    print(f"c_min fallback count : {fallback_n}")
    print(f"c_min fallback rate  : {fallback_rate*100:.2f}%")
    print("\n[Category distribution]")
    for cat, n in sorted(cat_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<20s} : {n:>4d}  ({n/total*100:5.2f}%)")
    print("\n[Confidence distribution]")
    for conf, n in sorted(conf_counter.items(), reverse=True):
        print(f"  conf={conf:<5} : {n:>4d}  ({n/total*100:5.2f}%)")

    # 给一些不规则样本示例
    print("\n[Sample fallback raw_responses] (first 10)")
    for rec in fallback_records[:10]:
        print(f"  - qid={rec['question_id']} | len={rec['raw_response_len']} | "
              f"raw={rec['raw_response']!r}")

    # ---- 5. 落盘 ----
    os.makedirs(os.path.dirname(args.dump_path), exist_ok=True)
    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "num_samples": total,
        "c_min_fallback_count": fallback_n,
        "c_min_fallback_rate": fallback_rate,
        "category_distribution": dict(cat_counter),
        "confidence_distribution": {str(k): v for k, v in conf_counter.items()},
        "fallback_records": fallback_records,
        "all_records": all_records,
    }
    with open(args.dump_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[info] full dump → {args.dump_path}")


if __name__ == "__main__":
    main()
