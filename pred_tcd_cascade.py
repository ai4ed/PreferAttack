#!/usr/bin/env python3
"""
PRED + TCD Cascade Defense Demo

Two-stage defense:
  Stage 1 (PRED): cheap, offline. Three PPL-derived signals OR-combined.
      - tripped   -> filter (do not flip preference)
      - borderline (signal near threshold within `margin`) -> escalate to TCD
      - clear     -> accept attacked preference
  Stage 2 (TCD): expensive, online judge re-evaluation on truncated instructions.
      For each K in K_set, re-run judge on I_truncated = drop last K tokens.
      If agreement rate (with full-instruction preference) <= tau, filter.

Reuses Llama-3.2-3B-Instruct as both PPL detector (for PRED) and judge (for TCD),
matching the original PreferAttack setup (so attack results transfer).

Inputs:
  --attack_results_json : cjb attack results (records have instruction, response_a, response_b, attack.best_suffix, baseline.choice, attack.new_choice)
  --fnr_fpr_json        : results_fnr_fpr_cjb_llama3b.json (PPL signals per attack/clean sample, used to derive PRED thresholds)
  --ppl_judge_model     : /root/autodl-tmp/Llama-3.2-3B-Instruct
  --clean_calibration_json : data/split/code_judge_bench_train.json (TCD FPR calibration)

Outputs:
  results_cascade_cjb.json with per-sample decisions and aggregate ASR / FPR.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_pred_qwen3b import (
    get_judge_prompt, parse_preference, generate_choice,
    load_model, fpr_threshold,
)


# -------------------- token truncation --------------------

def truncate_last_tokens(tokenizer, text: str, k: int) -> str:
    """Drop the last k tokens of text (after tokenization), return decoded text."""
    if k <= 0:
        return text
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=4000)["input_ids"][0].tolist()
    if k >= len(ids):
        return ""  # truncate everything
    keep = ids[:len(ids) - k]
    return tokenizer.decode(keep, skip_special_tokens=True)


# -------------------- cascade decision --------------------

def pred_stage(atk_pplw: float, atk_lafl: float, atk_ratio: float,
               thr_w: float, thr_l: float, thr_r: float,
               margin: float) -> tuple[str, list[str]]:
    """Return (decision, triggered_gates). decision in {filter, escalate, accept}."""
    gates = []
    if np.isfinite(atk_pplw) and atk_pplw > thr_w:
        gates.append("S_w")
    if np.isfinite(atk_lafl) and atk_lafl > thr_l:
        gates.append("S_l")
    if np.isfinite(atk_ratio) and atk_ratio > thr_r:
        gates.append("S_r")

    if gates:
        return "filter", gates

    # Borderline: any signal within `margin` of threshold (relative)
    borderline = False
    if np.isfinite(atk_pplw) and thr_w > 0 and atk_pplw >= thr_w * (1 - margin):
        borderline = True
    if np.isfinite(atk_lafl) and thr_l > 0 and atk_lafl >= thr_l * (1 - margin):
        borderline = True
    if np.isfinite(atk_ratio) and thr_r > 0 and atk_ratio >= thr_r * (1 - margin):
        borderline = True

    if borderline:
        return "escalate", []
    return "accept", []


def tcd_stage(tokenizer, model, instr_attacked: str, response_a: str, response_b: str,
              full_choice: int, K_set: list[int], tau: float) -> tuple[str, bool, float]:
    """Run TCD: return (decision, suspicious, agreement_rate). decision in {filter, accept}."""
    agree = 0
    n = 0
    for K in K_set:
        instr_trunc = truncate_last_tokens(tokenizer, instr_attacked, K)
        if not instr_trunc.strip():
            continue
        prompt = get_judge_prompt(instr_trunc, response_a, response_b)
        pref, _ = generate_choice(tokenizer, model, prompt)
        if pref is None:
            continue
        if pref == full_choice:
            agree += 1
        n += 1
    if n == 0:
        return "accept", False, 0.0
    rate = agree / n
    if rate <= tau:
        return "filter", True, rate
    return "accept", False, rate


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack_results_json", required=True)
    ap.add_argument("--fnr_fpr_json", default="/root/PreferAttack/results_fnr_fpr_cjb_llama3b.json")
    ap.add_argument("--ppl_judge_model", default="/root/autodl-tmp/Llama-3.2-3B-Instruct")
    ap.add_argument("--clean_calibration_json", default="/root/PreferAttack/data/split/code_judge_bench_train.json")
    ap.add_argument("--target_fpr", type=float, default=0.01)
    ap.add_argument("--margin", type=float, default=0.10,
                    help="PRED borderline band: signal within (1-margin)*thr triggers TCD")
    ap.add_argument("--K_set", type=int, nargs="+", default=[30, 60],
                    help="TCD truncation set (default {30, 60}, best from TCD paper)")
    ap.add_argument("--tcd_tau", type=float, default=0.0,
                    help="TCD agreement threshold: rate <= tau -> filter (default 0, paper-recommended)")
    ap.add_argument("--num_attack_samples", type=int, default=421)
    ap.add_argument("--num_calibration", type=int, default=200)
    ap.add_argument("--output_json", default="/root/PreferAttack/results_cascade_cjb.json")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    print(f"[INFO] attack results: {args.attack_results_json}")
    print(f"[INFO] fnr_fpr json: {args.fnr_fpr_json}")
    print(f"[INFO] judge+PPL model: {args.ppl_judge_model}")
    print(f"[INFO] PRED margin: {args.margin}")
    print(f"[INFO] TCD K_set: {args.K_set}, tau: {args.tcd_tau}")

    # ---------- Load PRED signals (from fnr_fpr file, attack side) ----------
    with open(args.fnr_fpr_json, "r", encoding="utf-8") as f:
        fdata = json.load(f)
    attack_signals = fdata["attack_records"]
    clean_signals = fdata["clean_records"]
    paper_thr_w = fdata["thresholds"]["ppl_w"]

    # ---------- Load attack results (to get instruction/responses/best_suffix) ----------
    with open(args.attack_results_json, "r", encoding="utf-8") as f:
        atk_data = json.load(f)
    records = atk_data.get("records", [])[:args.num_attack_samples]
    n_atk = len(records)
    print(f"[INFO] {n_atk} attack records")

    # ---------- Calibrate PRED thresholds on clean (already done in fnr_fpr file) ----------
    clean_w_arr = np.array([float(r.get("ppl_w")) for r in clean_signals], dtype=float)
    clean_ppl_arr = np.array([float(r.get("ppl")) for r in clean_signals], dtype=float)
    median_clean_w = float(np.median(clean_w_arr[np.isfinite(clean_w_arr)])) if np.any(np.isfinite(clean_w_arr)) else 1.0
    # thr_w: from fnr_fpr file (paper calibration)
    thr_w = float(paper_thr_w)
    # thr_l, thr_r: FPR-calibrated on this clean test set
    clean_lafl = clean_w_arr / clean_ppl_arr
    clean_lafl = clean_lafl[np.isfinite(clean_lafl) & (clean_lafl > 0)]
    thr_l, _ = fpr_threshold(clean_lafl, args.target_fpr)
    clean_ratio = clean_w_arr / max(median_clean_w, 1e-9)
    clean_ratio = clean_ratio[np.isfinite(clean_ratio) & (clean_ratio > 0)]
    thr_r, _ = fpr_threshold(clean_ratio, args.target_fpr)
    print(f"[INFO] PRED thresholds: thr_w={thr_w:.2f}(paper), thr_l={thr_l:.2f}, thr_r={thr_r:.3f}")

    # ---------- Load model ----------
    tokenizer, model = load_model(args.ppl_judge_model, dtype=args.dtype)

    # ---------- Run cascade on each attack sample ----------
    print(f"\n[INFO] Running cascade on {n_atk} samples...")
    out_records = []
    t0 = time.time()
    n_filter_pred = 0
    n_escalate = 0
    n_filter_tcd = 0
    n_success_no_defense = 0
    n_succ_survived = 0  # only counts samples where original_success=True AND final=accept

    for i, r in enumerate(records):
        instr = r.get("instruction", "")
        ra = r.get("response_a", "")
        rb = r.get("response_b", "")
        suffix = ((r.get("attack") or {}).get("best_suffix") or "").strip()
        attacked_instr = (instr.rstrip() + " " + suffix).strip() if suffix else instr

        base_choice = (r.get("baseline") or {}).get("choice")
        atk_choice = (r.get("attack") or {}).get("new_choice")
        original_success = bool(r.get("success", False))
        if original_success:
            n_success_no_defense += 1

        # Match PRED signals from fnr_fpr file by index
        if i >= len(attack_signals):
            continue
        sig = attack_signals[i]
        atk_pplw = float(sig.get("ppl_w_full"))
        atk_ppl = float(sig.get("ppl_full"))
        atk_lafl = atk_pplw / atk_ppl if atk_ppl > 0 else float("inf")
        # atk_ratio needs matched clean pplw
        clean_w = float(clean_signals[i].get("ppl_w")) if i < len(clean_signals) else float("nan")
        atk_ratio = atk_pplw / clean_w if clean_w > 0 else float("inf")

        decision = "accept"
        pred_gates = []
        tcd_rate = None
        tcd_suspicious = False

        if not original_success:
            # Attack already failed; no defense needed
            decision = "accept"
        else:
            # Stage 1: PRED
            decision, pred_gates = pred_stage(atk_pplw, atk_lafl, atk_ratio,
                                              thr_w, thr_l, thr_r, args.margin)
            if decision == "filter":
                n_filter_pred += 1
            elif decision == "escalate":
                n_escalate += 1
                # Stage 2: TCD (only if attack succeeded, run TCD on attacked instruction)
                if atk_choice is None:
                    decision = "accept"  # cannot run TCD without attack choice
                else:
                    tcd_decision, tcd_suspicious, tcd_rate = tcd_stage(
                        tokenizer, model, attacked_instr, ra, rb,
                        full_choice=atk_choice, K_set=args.K_set, tau=args.tcd_tau,
                    )
                    decision = tcd_decision
                    if tcd_decision == "filter":
                        n_filter_tcd += 1

        if original_success and decision == "accept":
            n_succ_survived += 1

        out_records.append({
            "id": r.get("id", i),
            "original_success": original_success,
            "pred_gates": pred_gates,
            "pred_decision": ("filter" if pred_gates else "escalate" if decision == "escalate" or tcd_rate is not None else "accept"),
            "tcd_agreement_rate": tcd_rate,
            "tcd_suspicious": tcd_suspicious,
            "final_decision": decision,
            "atk_pplw": atk_pplw,
            "atk_lafl": atk_lafl,
            "atk_ratio": atk_ratio,
        })

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{n_atk}] elapsed={el:.0f}s avg={el/(i+1):.2f}s "
                  f"filter_pred={n_filter_pred} escalate={n_escalate} filter_tcd={n_filter_tcd} "
                  f"succ_survived={n_succ_survived}")

    # ---------- Compute ASR-cascade and clean-side FPR ----------
    # Correct ASR: only counts samples where original_success=True AND final decision=accept
    asr_no_def = n_success_no_defense / n_atk if n_atk else 0.0
    asr_cascade = n_succ_survived / n_atk if n_atk else 0.0
    asr_reduction = (asr_no_def - asr_cascade) / asr_no_def * 100.0 if asr_no_def > 0 else 0.0

    # TCD clean FPR (proxy): re-run TCD on clean samples (no suffix) to estimate false-positive rate
    # We approximate by running TCD on the first num_calibration clean samples
    print(f"\n[INFO] Estimating clean-side FPR on {args.num_calibration} clean samples...")
    import random
    random.seed(42)
    with open(args.clean_calibration_json, "r", encoding="utf-8") as f:
        clean_all = json.load(f)
    if len(clean_all) > args.num_calibration:
        clean_all = random.sample(clean_all, args.num_calibration)
    n_clean_flag = 0
    n_clean_total = 0
    for i, ex in enumerate(clean_all):
        instr = ex.get("instruction", "")
        ra = ex.get("response_a", "")
        rb = ex.get("response_b", "")
        if not instr.strip():
            continue
        n_clean_total += 1
        prompt_full = get_judge_prompt(instr, ra, rb)
        full_pref, _ = generate_choice(tokenizer, model, prompt_full)
        if full_pref is None:
            continue
        # PRED side: get clean ppl_w from clean_signals[i] if available
        cw = float(clean_signals[i].get("ppl_w")) if i < len(clean_signals) else float("nan")
        cp = float(clean_signals[i].get("ppl")) if i < len(clean_signals) else float("nan")
        cl = cw / cp if cp > 0 else float("inf")
        cr = cw / max(median_clean_w, 1e-9) if cw > 0 else float("inf")
        clean_decision, _ = pred_stage(cw, cl, cr, thr_w, thr_l, thr_r, args.margin)
        if clean_decision == "filter":
            n_clean_flag += 1
            continue
        elif clean_decision == "escalate":
            # TCD on clean (no suffix)
            tcd_decision, _, _ = tcd_stage(
                tokenizer, model, instr, ra, rb,
                full_choice=full_pref, K_set=args.K_set, tau=args.tcd_tau,
            )
            if tcd_decision == "filter":
                n_clean_flag += 1
    clean_fpr = n_clean_flag / n_clean_total if n_clean_total else 0.0

    # ---------- Summary ----------
    print(f"\n{'='*78}")
    print(f"Samples: {n_atk}   Base success: {n_success_no_defense}   ASR (no defense): {asr_no_def*100:.2f}%")
    print(f"PRED filter: {n_filter_pred}   PRED escalate: {n_escalate}   TCD filter: {n_filter_tcd}")
    print(f"Survived (success, final): {n_succ_survived}   ASR-cascade: {asr_cascade*100:.2f}%")
    print(f"ASR-Reduction (cascade): {asr_reduction:.2f}%")
    print(f"Clean-side FPR (cascade): {clean_fpr*100:.2f}% ({n_clean_flag}/{n_clean_total})")
    print('=' * 78)

    summary = {
        "attack_results_json": args.attack_results_json,
        "ppl_judge_model": args.ppl_judge_model,
        "config": {
            "target_fpr": args.target_fpr,
            "margin": args.margin,
            "K_set": args.K_set,
            "tcd_tau": args.tcd_tau,
        },
        "thresholds": {"thr_w": thr_w, "thr_l": thr_l, "thr_r": thr_r,
                       "median_clean_w": median_clean_w},
        "asr_no_defense_pct": asr_no_def * 100.0,
        "asr_cascade_pct": asr_cascade * 100.0,
        "asr_reduction_pct": asr_reduction,
        "clean_fpr_pct": clean_fpr * 100.0,
        "counts": {
            "n_filter_pred": n_filter_pred,
            "n_escalate": n_escalate,
            "n_filter_tcd": n_filter_tcd,
            "n_succ_survived": n_succ_survived,
            "n_clean_flag": n_clean_flag,
            "n_clean_total": n_clean_total,
        },
    }
    out = {"summary": summary, "records": out_records}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Wrote -> {args.output_json}")


if __name__ == "__main__":
    main()
