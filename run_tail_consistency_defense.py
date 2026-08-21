#!/usr/bin/env python3
"""
Run Tail-Consistency Defense (TCD) against PreferAttack.

This is a STANDALONE script (per project convention: keep new methods in
independent files). It evaluates TCD on a PreferAttack attack-results JSON
using the SAME judge model that produced the attack, and reports metrics
aligned with the paper's §5.11 PPL-defense protocol:

    ASR / ASR-W / ASR-Reduction / FNR / FPR

Mechanism (see TCD_DEFENSE_DESIGN.md):
    For each attack sample:
      1. Reconstruct attacked_instruction = instruction + " " + suffix
      2. judge(attacked_instruction, A, B)  -> pref_full
      3. for K in K_set: judge(drop_last_K_tokens(attacked_instruction), A, B)
                                                      -> prefs_trunc
      4. agreement = mean(prefs_trunc == pref_full)
      5. flag if agreement < threshold (calibrated on clean data, FPR <= 1%)

    If flagged -> defense kicks in -> revert to truncated preference (which
    approximates baseline choice, since the suffix was the only perturbation).

Usage:
    # Smoke test on 10 samples
    python3 run_tail_consistency_defense.py \\
        --attack_json results/multi_agent_pairwise_eval_origin_opti_code_judge_bench_20260703_220310.json \\
        --judge_model /root/autodl-tmp/Qwen2.5-3B-Instruct/ \\
        --num_samples 10

    # Full run
    python3 run_tail_consistency_defense.py \\
        --attack_json results/multi_agent_pairwise_eval_origin_opti_code_judge_bench_20260703_220310.json \\
        --judge_model /root/autodl-tmp/Qwen2.5-3B-Instruct/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # src/defense -> repository root (so `src.*` / `utils.*` resolve)
sys.path.insert(0, str(REPO_ROOT))

from src.defense.tail_consistency_defense import (  # noqa: E402
    AttackRecord,
    TailConsistencyDefense,
    CleanPair,
    asr_under_defense,
    fnr_fpr,
    load_attack_pairs_from_json,
    load_clean_pairs_from_attack_json,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack_json", type=str, required=True,
                    help="PreferAttack attack results JSON (must match judge_model).")
    ap.add_argument("--judge_model", type=str, default="/root/autodl-tmp/Qwen2.5-3B-Instruct/",
                    help="Path to local HF model used to produce the attack (vLLM judge).")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16"])
    ap.add_argument("--K_set", type=str, default="30,60,120,200",
                    help="Comma-separated truncation lengths in tokens.")
    ap.add_argument("--target_fpr", type=float, default=0.01,
                    help="Target FPR on clean data; threshold calibrated to satisfy this.")
    ap.add_argument("--num_samples", type=int, default=0,
                    help="If >0, only evaluate first N attack samples (smoke test).")
    ap.add_argument("--num_clean", type=int, default=200,
                    help="Max clean pairwise queries used for FPR calibration.")
    ap.add_argument("--require_success", action="store_true",
                    help="Only evaluate attack samples that succeeded.")
    ap.add_argument("--use_v2", action="store_true",
                    help="Use v2 truncation-baseline-corrected TCD (position-bias corrected). "
                         "Costs 2x judge calls but eliminates Qwen2.5-3B position bias.")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Output summary JSON path. Default: results/tail_consistency_defense_summary_<basename>.json")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    K_set = tuple(int(x) for x in args.K_set.split(","))
    attack_path = args.attack_json
    if not os.path.exists(attack_path):
        print(f"[ERROR] attack JSON not found: {attack_path}")
        sys.exit(1)

    out_json = args.out_json or (
        HERE / "results" / f"tail_consistency_defense_summary_{Path(attack_path).stem}.json"
    )
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TCD DEFENSE — Tail-Consistency Defense")
    print("=" * 78)
    print(f"  Attack JSON      : {attack_path}")
    print(f"  Judge model      : {args.judge_model}")
    print(f"  K_set            : {K_set}")
    print(f"  target_fpr       : {args.target_fpr}")
    print(f"  use_v2           : {args.use_v2}")
    print(f"  num_samples      : {args.num_samples if args.num_samples > 0 else 'all'}")
    print(f"  num_clean        : {args.num_clean}")
    print(f"  out_json         : {out_json}")

    # ------------------------------------------------------------------ #
    # 1. Load judge (vLLM)                                               #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading vLLM judge...")
    t0 = time.time()
    from utils.vllm_judge import create_vllm_judge, SampleTooLongError
    judge = create_vllm_judge(model_path=args.judge_model, dtype=args.dtype)
    print(f"  Judge loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    # 2. Load data                                                       #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Loading data...")
    attack_records_raw = load_attack_pairs_from_json(
        attack_path, require_success=args.require_success,
    )
    if args.num_samples > 0:
        attack_records_raw = attack_records_raw[: args.num_samples]
    print(f"  Attack records: {len(attack_records_raw)} "
          f"(success={sum(1 for r in attack_records_raw if r.get('success'))})")

    clean_pairs_all = load_clean_pairs_from_attack_json(
        attack_path, cap=args.num_clean, seed=args.seed,
    )
    print(f"  Clean pairs  : {len(clean_pairs_all)} (for FPR calibration)")

    # ------------------------------------------------------------------ #
    # 3. Build TCD defense + calibrate threshold on clean                #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Calibrating TCD threshold on clean pairs...")
    tcd = TailConsistencyDefense(
        judge=judge,
        K_set=K_set,
        target_fpr=args.target_fpr,
    )
    t0 = time.time()
    clean_results = []
    for i, cp in enumerate(clean_pairs_all):
        try:
            if args.use_v2:
                r = tcd.evaluate_sample_v2(
                    cp.instruction, suffix="", response_a=cp.response_a,
                    response_b=cp.response_b, qid=cp.question_id,
                )
            else:
                r = tcd.evaluate_sample(
                    cp.instruction, suffix="", response_a=cp.response_a,
                    response_b=cp.response_b, qid=cp.question_id,
                )
            clean_results.append(r)
        except SampleTooLongError:
            continue
        except Exception as e:
            print(f"  [WARN] clean sample {i} failed: {e}")
            continue
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(clean_pairs_all)} clean evaluated")

    if args.use_v2:
        # v2: persistence_rate calibration
        clean_rates = [r["persistence_rate"] for r in clean_results if r["n_pairs_evaluated"] > 0]
        if clean_rates:
            tcd.threshold = float(np.percentile(clean_rates, 100.0 * (1.0 - args.target_fpr)))
        clean_fpr = float(np.mean([
            r["persistence_rate"] > tcd.threshold and r["persistence_rate"] > 0
            for r in clean_results
        ])) if clean_results else 0.0
        print(f"  Calibrated in {time.time() - t0:.1f}s")
        if clean_rates:
            n_zero = sum(1 for x in clean_rates if x == 0.0)
            print(f"  Clean persistence_rate: "
                  f"min={min(clean_rates):.3f}, "
                  f"median={float(np.median(clean_rates)):.3f}, "
                  f"max={max(clean_rates):.3f}  (n={len(clean_rates)}, "
                  f"{n_zero} exactly 0)")
        print(f"  Calibrated threshold = {tcd.threshold:.4f}")
        print(f"  Empirical FPR on clean = {clean_fpr:.4f} (target ≤ {args.target_fpr})")
    else:
        # v1: agreement calibration
        clean_agreements = [r["agreement"] for r in clean_results if r["n_trunc_evaluated"] > 0]
        if clean_agreements:
            tcd.threshold = float(np.percentile(clean_agreements, args.target_fpr * 100.0))
        clean_fpr = float(np.mean([r["agreement"] <= tcd.threshold and r["agreement"] < 1.0 for r in clean_results])) if clean_results else 0.0
        print(f"  Calibrated in {time.time() - t0:.1f}s")
        print(f"  Clean agreement distribution: "
              f"min={min(clean_agreements):.3f}, "
              f"median={float(np.median(clean_agreements)):.3f}, "
              f"max={max(clean_agreements):.3f}  (n={len(clean_agreements)})")
        print(f"  Calibrated threshold = {tcd.threshold:.4f}")
        print(f"  Empirical FPR on clean = {clean_fpr:.4f} (target ≤ {args.target_fpr})")

    # ------------------------------------------------------------------ #
    # 4. Evaluate TCD on attack samples                                  #
    # ------------------------------------------------------------------ #
    print(f"\n[4/4] Evaluating TCD on {len(attack_records_raw)} attack samples...")
    t0 = time.time()
    attack_results = []
    n_clean_too_long = 0
    for i, rec in enumerate(attack_records_raw):
        instr = rec.get("instruction", "")
        suffix = (rec.get("attack", {}) or {}).get("best_suffix", "")
        a = rec.get("response_a", "")
        b = rec.get("response_b", "")
        try:
            if args.use_v2:
                r = tcd.evaluate_sample_v2(instr, suffix=suffix, response_a=a, response_b=b,
                                           qid=str(rec.get("id", "")))
            else:
                r = tcd.evaluate_sample(instr, suffix=suffix, response_a=a, response_b=b,
                                        qid=str(rec.get("id", "")))
        except SampleTooLongError:
            n_clean_too_long += 1
            continue
        except Exception as e:
            print(f"  [WARN] attack sample {i} (id={rec.get('id')}) failed: {e}")
            continue
        # Augment with attack metadata for downstream analysis
        r["id"] = rec.get("id")
        r["baseline_choice"] = (rec.get("baseline", {}) or {}).get("choice")
        r["new_choice"] = (rec.get("attack", {}) or {}).get("new_choice")
        r["success"] = bool(rec.get("success"))
        attack_results.append(r)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    ... {i+1}/{len(attack_records_raw)} "
                  f"({rate:.2f} samples/sec, {elapsed:.0f}s elapsed)")
    eval_time = time.time() - t0
    print(f"  Evaluated {len(attack_results)} attack samples in {eval_time:.1f}s "
          f"({len(attack_results)/eval_time:.2f} samples/sec)")
    if n_clean_too_long > 0:
        print(f"  [INFO] {n_clean_too_long} samples skipped (prompt too long after truncation)")

    # ------------------------------------------------------------------ #
    # Compute metrics                                                    #
    # ------------------------------------------------------------------ #
    # Two notions of "attack success":
    #   (a) JSON-stale: success flag stored in the attack JSON (original run)
    #   (b) Reproduced: TCD's re-judged pref_full != baseline_choice (now)
    # (b) is stricter and reflects what TCD actually has to defend against.
    flag_arr = np.array([r["flag"] for r in attack_results], dtype=bool)
    succ_json = np.array([r["success"] for r in attack_results], dtype=bool)
    base_choices = np.array([r["baseline_choice"] for r in attack_results], dtype=object)
    if args.use_v2:
        pref_full_arr = np.array([r["pref_full_attacked"] for r in attack_results], dtype=object)
    else:
        pref_full_arr = np.array([r["pref_full"] for r in attack_results], dtype=object)

    # Reproduced-success mask: judge currently flips relative to baseline
    reproduced = np.array([
        (pf is not None and bc is not None and pf != bc)
        for pf, bc in zip(pref_full_arr, base_choices)
    ], dtype=bool)

    # (a) JSON-stale ASR / ASR-W
    records_json = [
        AttackRecord(instruction="", suffix="", success=bool(s), source="tcd_eval")
        for s in succ_json
    ]
    asr_json = asr_under_defense(records_json, flag_arr)

    # (b) Reproduced-attack ASR / ASR-W (stricter, what TCD actually sees)
    records_repr = [
        AttackRecord(instruction="", suffix="", success=bool(s), source="tcd_eval")
        for s in reproduced
    ]
    asr_repr = asr_under_defense(records_repr, flag_arr)

    # Detection quality on each "success" view
    det_json = fnr_fpr(
        np.array([1] * int(succ_json.sum()), dtype=int) if succ_json.sum() else np.array([], dtype=int),
        flag_arr[succ_json] if succ_json.sum() else np.array([], dtype=bool),
    ) if succ_json.sum() else {"fnr": float("nan"), "fpr": float("nan"), "n_adv": 0, "n_clean": 0, "n_flagged_adv": 0}

    det_repr = fnr_fpr(
        np.array([1] * int(reproduced.sum()), dtype=int) if reproduced.sum() else np.array([], dtype=int),
        flag_arr[reproduced] if reproduced.sum() else np.array([], dtype=bool),
    ) if reproduced.sum() else {"fnr": float("nan"), "fpr": float("nan"), "n_adv": 0, "n_clean": 0, "n_flagged_adv": 0}

    # "all attack samples" detector view
    y_true_all = np.array([1] * len(attack_results), dtype=int)
    det_all = fnr_fpr(y_true_all, flag_arr)

    # ------------------------------------------------------------------ #
    # Report                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  Total attack samples evaluated : {len(attack_results)}")
    print(f"  Successful attacks (JSON flag) : {int(succ_json.sum())}")
    print(f"  Successful attacks (reproduced): {int(reproduced.sum())}  "
          f"[judge currently flips pref_full ≠ baseline]")
    print(f"  Clean FPR (calibration)        : {clean_fpr:.4f} (target ≤ {args.target_fpr})")
    if args.use_v2:
        print(f"  Calibrated persistence threshold: {tcd.threshold:.4f}  "
              f"(flag if persistence_rate > threshold)")
    else:
        print(f"  Calibrated agreement threshold : {tcd.threshold:.4f}")
    print()
    print(f"  ---- Using JSON-stale success flag (original attack's record) ----")
    print(f"  ASR                            : {asr_json['asr']:.4f}")
    print(f"  ASR-W (under TCD defense)      : {asr_json['asr_w']:.4f}")
    print(f"  ASR-Reduction                  : {asr_json['asr_reduction']:.4f}  "
          f"({asr_json['asr_reduction']*100:.1f} pp)")
    print(f"  Caught successful (FNR@1%FPR)  : {det_json.get('n_flagged_adv',0)}/{det_json.get('n_adv',0)}  "
          f"(FNR={det_json.get('fnr', float('nan')):.4f})")
    print()
    print(f"  ---- Using reproduced-attack success (judge currently flips) ----")
    print(f"  ASR                            : {asr_repr['asr']:.4f}")
    print(f"  ASR-W (under TCD defense)      : {asr_repr['asr_w']:.4f}")
    print(f"  ASR-Reduction                  : {asr_repr['asr_reduction']:.4f}  "
          f"({asr_repr['asr_reduction']*100:.1f} pp)")
    print(f"  Caught successful (FNR@1%FPR)  : {det_repr.get('n_flagged_adv',0)}/{det_repr.get('n_adv',0)}  "
          f"(FNR={det_repr.get('fnr', float('nan')):.4f})")

    # Per-sample score distribution on attacks
    if args.use_v2:
        attack_scores = [r["persistence_rate"] for r in attack_results]
        # For v2, also track how often suffix_changed_full is True
        suffix_changed = [bool(r.get("suffix_changed_full")) for r in attack_results]
        print()
        print(f"  Attack persistence_rate distribution:")
        print(f"    min={min(attack_scores):.3f}, "
              f"median={float(np.median(attack_scores)):.3f}, "
              f"max={max(attack_scores):.3f}")
        print(f"    fraction with persistence_rate>0.0 (suffix has effect at some K): "
              f"{float(np.mean([a > 0.0 for a in attack_scores])):.3f}")
        print(f"    fraction with persistence_rate>0.5 : "
              f"{float(np.mean([a > 0.5 for a in attack_scores])):.3f}")
        print(f"    fraction where suffix_changed_full=True (attack flips full pref): "
              f"{float(np.mean(suffix_changed)):.3f}")
    else:
        attack_agreements = [r["agreement"] for r in attack_results]
        print()
        print(f"  Attack agreement distribution:")
        print(f"    min={min(attack_agreements):.3f}, "
              f"median={float(np.median(attack_agreements)):.3f}, "
              f"max={max(attack_agreements):.3f}")
        print(f"    fraction with agreement=0.0 (full reversal): "
              f"{float(np.mean([a == 0.0 for a in attack_agreements])):.3f}")
        print(f"    fraction with agreement<0.5 : "
              f"{float(np.mean([a < 0.5 for a in attack_agreements])):.3f}")

    # ------------------------------------------------------------------ #
    # Save summary                                                       #
    # ------------------------------------------------------------------ #
    summary = {
        "config": {
            "attack_json": attack_path,
            "judge_model": args.judge_model,
            "K_set": list(K_set),
            "target_fpr": args.target_fpr,
            "use_v2": args.use_v2,
            "threshold_calibrated": tcd.threshold,
            "num_samples_limit": args.num_samples,
            "require_success": args.require_success,
            "seed": args.seed,
        },
        "data": {
            "n_attack_total": len(attack_records_raw),
            "n_attack_evaluated": len(attack_results),
            "n_attack_success": int(succ_json.sum()),
            "n_attack_reproduced": int(reproduced.sum()),
            "n_clean_calibration": len(clean_results),
            "n_skipped_too_long": n_clean_too_long,
        },
        "calibration": {
            "clean_fpr_empirical": clean_fpr,
            **({
                "clean_persistence_min": float(min(clean_rates)) if clean_rates else None,
                "clean_persistence_median": float(np.median(clean_rates)) if clean_rates else None,
                "clean_persistence_max": float(max(clean_rates)) if clean_rates else None,
                "clean_persistence_n_zero": int(sum(1 for x in clean_rates if x == 0.0)) if clean_rates else 0,
            } if args.use_v2 else {
                "clean_agreement_min": float(min(clean_agreements)) if clean_agreements else None,
                "clean_agreement_median": float(np.median(clean_agreements)) if clean_agreements else None,
                "clean_agreement_max": float(max(clean_agreements)) if clean_agreements else None,
            }),
            "threshold": tcd.threshold,
        },
        "metrics": {
            "json_stale": {
                "asr": asr_json["asr"],
                "asr_w": asr_json["asr_w"],
                "asr_reduction": asr_json["asr_reduction"],
                "detector": det_json,
            },
            "reproduced": {
                "asr": asr_repr["asr"],
                "asr_w": asr_repr["asr_w"],
                "asr_reduction": asr_repr["asr_reduction"],
                "detector": det_repr,
            },
            "detector_all_attack_samples": det_all,
            **({
                "attack_persistence_min": float(min(attack_scores)) if attack_scores else None,
                "attack_persistence_median": float(np.median(attack_scores)) if attack_scores else None,
                "attack_persistence_max": float(max(attack_scores)) if attack_scores else None,
                "attack_persistence_frac_positive": float(np.mean([a > 0.0 for a in attack_scores])) if attack_scores else None,
                "attack_suffix_changed_full_rate": float(np.mean(suffix_changed)) if suffix_changed else None,
            } if args.use_v2 else {
                "attack_agreement_min": float(min(attack_agreements)) if attack_agreements else None,
                "attack_agreement_median": float(np.median(attack_agreements)) if attack_agreements else None,
                "attack_agreement_max": float(max(attack_agreements)) if attack_agreements else None,
                "attack_agreement_frac_full_reversal": float(np.mean([a == 0.0 for a in attack_agreements])) if attack_agreements else None,
            }),
            "frac_attack_reproduced": float(reproduced.mean()) if len(reproduced) else None,
        },
        "eval_runtime_sec": round(eval_time, 2),
        "per_sample_results": (
            [
                {
                    "id": r.get("id"),
                    "success": r["success"],
                    "reproduced": bool(r["pref_full_attacked"] is not None and r["baseline_choice"] is not None and r["pref_full_attacked"] != r["baseline_choice"]),
                    "baseline_choice": r["baseline_choice"],
                    "new_choice": r["new_choice"],
                    "pref_full_attacked": r["pref_full_attacked"],
                    "pref_full_clean": r["pref_full_clean"],
                    "prefs_trunc_attacked": r["prefs_trunc_attacked"],
                    "prefs_trunc_clean": r["prefs_trunc_clean"],
                    "K_used": r["K_used"],
                    "persistence_rate": r["persistence_rate"],
                    "suffix_changed_full": bool(r["suffix_changed_full"]),
                    "flag": bool(r["flag"]),
                }
                for r in attack_results
            ] if args.use_v2 else [
                {
                    "id": r.get("id"),
                    "success": r["success"],
                    "reproduced": bool(r["pref_full"] is not None and r["baseline_choice"] is not None and r["pref_full"] != r["baseline_choice"]),
                    "baseline_choice": r["baseline_choice"],
                    "new_choice": r["new_choice"],
                    "pref_full": r["pref_full"],
                    "prefs_trunc": r["prefs_trunc"],
                    "K_used": r["K_used"],
                    "agreement": r["agreement"],
                    "flag": bool(r["flag"]),
                }
                for r in attack_results
            ]
        ),
        "per_sample_clean_results": (
            [
                {
                    "id": r.get("id", ""),
                    "pref_full_attacked": r["pref_full_attacked"],
                    "pref_full_clean": r["pref_full_clean"],
                    "prefs_trunc_attacked": r["prefs_trunc_attacked"],
                    "prefs_trunc_clean": r["prefs_trunc_clean"],
                    "K_used": r["K_used"],
                    "persistence_rate": r["persistence_rate"],
                    "flag": bool(r["flag"]),
                }
                for r in clean_results
            ] if args.use_v2 else [
                {
                    "id": r.get("id", ""),
                    "pref_full": r["pref_full"],
                    "prefs_trunc": r["prefs_trunc"],
                    "K_used": r["K_used"],
                    "agreement": r["agreement"],
                    "flag": bool(r["flag"]),
                }
                for r in clean_results
            ]
        ),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print()
    print(f"Wrote summary to: {out_json}")


if __name__ == "__main__":
    main()
