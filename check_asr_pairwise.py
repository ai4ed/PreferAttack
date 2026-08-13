# check_asr_pairwise.py
import json, argparse

def safe_get_queries(rec):
    q = rec.get("queries") or {}
    return {
        "api_calls": int(q.get("api_calls", 0) or 0),
        "candidates_evaluated": int(q.get("candidates_evaluated", 0) or 0),
        "prompt_tokens": int(q.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(q.get("completion_tokens", 0) or 0),
        "total_tokens": int(q.get("total_tokens", 0) or 0),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="./results/multi_agent_pairwise_eval_origin_opti_RL_GAMMA=0.95_alpaca_eval_20260309_084444.json")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    recs = data.get("records", [])

    n = len(recs)
    print(f"Total records loaded: {n}")
    succ_recs = [r for r in recs if r.get("success")]
    succ = len(succ_recs)
    asr = 0.0 if n == 0 else succ * 100.0 / n

    # 总体平均 query 次数
    total_api_calls = sum(safe_get_queries(r)["api_calls"] for r in recs)
    total_candidates = sum(safe_get_queries(r)["candidates_evaluated"] for r in recs)
    total_tokens = sum(safe_get_queries(r)["total_tokens"] for r in recs)
    print(f"Total API calls (all samples): {total_api_calls}")
    print(f"Total candidates evaluated (all samples): {total_candidates}")
    print(f"Total tokens (all samples, search/final only): {total_tokens}")
    avg_api_calls = float(total_api_calls) / n if n else 0.0
    avg_candidates = float(total_candidates) / n if n else 0.0
    avg_total_tokens = float(total_tokens) / n if n else 0.0

    # 成功样本的平均 query 次数
    succ_api_calls = sum(safe_get_queries(r)["api_calls"] for r in succ_recs)
    succ_candidates = sum(safe_get_queries(r)["candidates_evaluated"] for r in succ_recs)
    succ_total_tokens = sum(safe_get_queries(r)["total_tokens"] for r in succ_recs)
    avg_api_calls_succ = float(succ_api_calls) / succ if succ else 0.0
    avg_candidates_succ = float(succ_candidates) / succ if succ else 0.0
    avg_total_tokens_succ = float(succ_total_tokens) / succ if succ else 0.0

    # 如果存在首次成功时的记录，计算这些统计（可能数量少于 succ）
    until_list = [r.get("queries_until_success") for r in recs if r.get("queries_until_success")]
    until_count = len(until_list)
    avg_api_calls_until = float(sum(x.get("api_calls", 0) for x in until_list)) / until_count if until_count else 0.0
    avg_candidates_until = float(sum(x.get("candidates_evaluated", 0) for x in until_list)) / until_count if until_count else 0.0
    total_tokens_until = sum(x.get("total_tokens", 0) for x in until_list)
    avg_total_tokens_until = float(total_tokens_until) / until_count if until_count else 0.0
    total_tokens_until_with_baseline = sum(x.get("total_tokens_including_baseline", 0) for x in until_list)
    avg_total_tokens_until_with_baseline = float(total_tokens_until_with_baseline) / until_count if until_count else 0.0

    print(f"File: {args.path}")
    print(f"Samples: {n}  Success: {succ}  ASR: {asr:.2f}%")
    print("--- Queries (averages) ---")
    print(f"Avg API calls per sample (all): {avg_api_calls:.2f}")
    print(f"Avg candidates evaluated per sample (all): {avg_candidates:.2f}")
    print(f"Avg total tokens per sample (all): {avg_total_tokens:.2f}")
    print(f"Avg API calls per successful sample: {avg_api_calls_succ:.2f}")
    print(f"Avg candidates per successful sample: {avg_candidates_succ:.2f}")
    print(f"Avg total tokens per successful sample: {avg_total_tokens_succ:.2f}")
    if until_count:
        print(f"Avg API calls until first success (count={until_count}): {avg_api_calls_until:.2f}")
        print(f"Avg candidates until first success: {avg_candidates_until:.2f}")
        print(f"Total tokens until first success: {total_tokens_until}  Avg: {avg_total_tokens_until:.2f}")
        print(f"Total tokens until first success (incl. baseline): {total_tokens_until_with_baseline}  Avg: {avg_total_tokens_until_with_baseline:.2f}")

if __name__ == "__main__":
    main()
