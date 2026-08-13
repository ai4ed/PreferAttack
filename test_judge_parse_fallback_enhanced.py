"""
端到端测试 EnhancedVLLMJudge 在 MiniCPM5-1B 上的兜底率,
对照 baseline (原始 VLLMJudge) 和不同 max_tokens 配置。
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_types import PairwiseExample
from utils.vllm_judge_enhanced import _classify, create_enhanced_vllm_judge


def load_examples(json_path: str, limit: int) -> List[PairwiseExample]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = []
    for item in raw[:limit]:
        examples.append(PairwiseExample(
            question_id=item.get("question_id", ""),
            instruction=item.get("instruction", ""),
            response_a=item.get("response_a", ""),
            response_b=item.get("response_b", ""),
            model_a=item.get("model_a", ""),
            model_b=item.get("model_b", ""),
        ))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/autodl-tmp/MiniCPM5-1B")
    ap.add_argument("--data_path", default="data/split/alpaca_eval_test.json")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=16,
                    help="初始 max_tokens (retry 会用 4x)")
    ap.add_argument("--retry_max_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=23872)
    ap.add_argument("--dump_path", default="results/judge_parse_fallback_enhanced_Minicpm5_1B.json")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", str(args.gpu_memory_utilization))
    os.environ.setdefault("VLLM_MAX_MODEL_LEN", str(args.max_model_len))
    os.environ.setdefault("VLLM_SWAP_SPACE_GB", "8")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    print(f"[info] model_path = {args.model_path}")
    print(f"[info] data_path  = {args.data_path}")
    print(f"[info] max_new_tokens = {args.max_new_tokens}, retry_max_tokens = {args.retry_max_tokens}")

    examples = load_examples(args.data_path, args.num_samples if args.num_samples > 0 else 1 << 30)
    print(f"[info] loaded {len(examples)} examples")

    judge = create_enhanced_vllm_judge(
        model_path=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        seed=42,
        retry_max_tokens=args.retry_max_tokens,
    )
    print(f"[info] EnhancedVLLMJudge loaded; retry_max_tokens = {judge.retry_max_tokens}")

    print(f"[info] running judge on {len(examples)} examples (with retry) ...")
    responses = judge.judge_examples(
        examples,
        batch_size=16,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=False,
    )

    # 统计: 用增强解析器的 _classify 重新分类 (与 judge._parse_response 一致)
    cat_counter = Counter()
    conf_counter = Counter()
    none_records = []
    for ex, resp in zip(examples, responses):
        cat, conf = resp.preference, resp.confidence
        # 用 raw_response 重新分类拿到 stage 名
        from utils.vllm_judge import get_judge_prompt
        gen = (resp.raw_response or "").replace(get_judge_prompt(ex), "").strip().lower()
        # 但 judge 内部可能已经 retry 过, 我们这里只看最终结果
        if cat is None:
            stage = "invalid_none"
        else:
            stage = f"conf_{conf:.2f}"
        cat_counter[stage] += 1
        conf_counter[round(conf, 2)] += 1
        if cat is None:
            none_records.append({
                "question_id": ex.question_id,
                "raw_response": resp.raw_response,
                "raw_response_len": len(resp.raw_response or ""),
            })

    total = len(responses)
    none_n = sum(1 for r in responses if r.preference is None)
    none_rate = none_n / total if total else 0.0

    # 对照: 旧 random fallback 是把 None 当 (random, 0.3), 现在改成 None
    print("\n========== Enhanced Judge Summary ==========")
    print(f"Total samples           : {total}")
    print(f"Invalid (pref=None)     : {none_n}  ({none_rate*100:.2f}%)")
    print(f"  (旧版会把这些当 random+0.3 兜底, 现在标记为中性, 不污染优化)")
    print(f"\n[Stage / Confidence distribution]")
    for stage, n in sorted(cat_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:<22s} : {n:>4d}  ({n/total*100:5.2f}%)")

    print(f"\n[Sample invalid raw_responses] (first 10)")
    for r in none_records[:10]:
        raw = (r["raw_response"] or "").replace("\n", "\\n")
        print(f"  qid={r['question_id']:>15s} | len={r['raw_response_len']:>3d} | raw={raw[:90]!r}")

    os.makedirs(os.path.dirname(args.dump_path), exist_ok=True)
    summary = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_tokens": args.retry_max_tokens,
        "num_samples": total,
        "invalid_count": none_n,
        "invalid_rate": none_rate,
        "stage_distribution": dict(cat_counter),
        "confidence_distribution": {str(k): v for k, v in conf_counter.items()},
        "invalid_records": none_records,
        "all_records": [
            {"question_id": ex.question_id, "preference": r.preference,
             "confidence": r.confidence, "raw_response": r.raw_response}
            for ex, r in zip(examples, responses)
        ],
    }
    with open(args.dump_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[info] dump → {args.dump_path}")


if __name__ == "__main__":
    main()
