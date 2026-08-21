#!/usr/bin/env python3
"""
Compute FNR and FPR for PPL / PPL-W defenses on a PreferAttack result file.

Definitions (matching paper "FNR and FPR for PPL and PPL-W Detection"):
  Positive (adversarial) = every record in the attack file (each has best_suffix appended)
  Negative (clean)       = the corresponding *_test.json (same samples, pre-attack)
  Threshold              = FPR-based threshold computed on *_train.json (carried over
                           from results_asr_w_*_llama3b.json, same PPL model)
  PPL defense : FNR = mean(PPL     <= thr_ppl  over attack samples)
                FPR = mean(PPL     >  thr_ppl  over clean  samples)
  PPL-W defense: same with thr_pplw and PPL_W score.

Default score = windowed/full PPL computed on instruction+suffix+responses (matches
the ASR-W main results in the handoff). Also reports instruction-only variants.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from compute_asr_w_cpu import (
    load_model, calculate_ppl, calculate_windowed_ppl, concat_parts,
    get_best_suffix,
)


def load_attack_scores(asr_w_json: str):
    with open(asr_w_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    s = d["summary"]
    recs = d["records"]
    return {
        "threshold_ppl": s["threshold_ppl_full"],
        "threshold_pplw": s["threshold_pplw_full"],
        "ppl_model": s["ppl_model"],
        "records": recs,
    }


def compute_clean_scores(tokenizer, model, clean_json: str, key: str = "full"):
    """key in {full, instr}: which text to score."""
    with open(clean_json, "r", encoding="utf-8") as f:
        clean = json.load(f)
    out = []
    t0 = time.time()
    for i, ex in enumerate(clean):
        instr = ex.get("instruction", "")
        ra = ex.get("response_a", "")
        rb = ex.get("response_b", "")
        text_full = concat_parts([instr, ra, rb])
        if key == "full":
            text = text_full
        elif key == "instr":
            text = instr if instr.strip() else text_full
        else:
            raise ValueError(key)
        ppl = calculate_ppl(tokenizer, model, text)
        ppl_w = calculate_windowed_ppl(tokenizer, model, text)
        out.append({"id": ex.get("question_id", i), "ppl": ppl, "ppl_w": ppl_w})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  clean {key} {i+1}/{len(clean)}  elapsed={el:.0f}s avg={el/(i+1):.2f}s")
    return out


def fnr_fpr(scores: list, thr: float, key: str):
    arr = np.array([r[key] for r in scores], dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    detected = int(np.sum(arr > thr))      # positive classed as attack
    missed = int(np.sum(arr <= thr))       # positive classed as clean (= FN for adversarial)
    if key.startswith("ppl"):
        # for FNR (attack side): FN rate = (arr <= thr).mean
        fnr = missed * 100.0 / n
        # for FPR (clean side): FP rate = (arr > thr).mean
        fpr = detected * 100.0 / n
    return fnr, fpr, missed, detected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[
        "code_judge_bench:code_judge_bench",
        "alpaca_eval:alpaca_eval",
        "arena_hard:arena_hard",
    ], help="name:results_prefix pairs (results file = results_asr_w_<prefix>_llama3b.json)")
    ap.add_argument("--results_dir", default="/root/PreferAttack")
    ap.add_argument("--clean_dir", default="/root/PreferAttack/data/split")
    args = ap.parse_args()

    print("=" * 78)
    print(f"{'Dataset':<20}{'Defense':<10}{'FNR':>10}{'FPR':>10}  (threshold)")
    print("=" * 78)

    rows = []
    seen_model = False
    tokenizer = model = None
    for spec in args.datasets:
        name, prefix = spec.split(":")
        asr_w_json = Path(args.results_dir) / f"results_asr_w_{prefix}_llama3b.json"
        clean_json = Path(args.clean_dir) / f"{name}_test.json"
        if not asr_w_json.exists():
            print(f"[skip] {asr_w_json} not found")
            continue
        if not clean_json.exists():
            print(f"[skip] {clean_json} not found")
            continue

        a = load_attack_scores(asr_w_json)
        if not seen_model:
            print(f"[INFO] Loading PPL model: {a['ppl_model']}")
            tokenizer, model = load_model(a["ppl_model"])
            seen_model = True
        elif model is None:
            tokenizer, model = load_model(a["ppl_model"])

        thr_ppl = a["threshold_ppl"]
        thr_pplw = a["threshold_pplw"]

        # attack scores already on file (full + instr variants)
        attack = a["records"]
        clean = compute_clean_scores(tokenizer, model, str(clean_json), key="full")

        # PPL defense (full text)
        fnr_atk_ppl, _, _, _ = fnr_fpr(attack, thr_ppl, "ppl_full")
        _, fpr_cln_ppl, _, _ = fnr_fpr(clean, thr_ppl, "ppl")
        # PPL-W defense (full text)
        fnr_atk_pw, _, _, _ = fnr_fpr(attack, thr_pplw, "ppl_w_full")
        _, fpr_cln_pw, _, _ = fnr_fpr(clean, thr_pplw, "ppl_w")

        print(f"{name:<20}{'PPL':<10}{fnr_atk_ppl:>9.2f}%{fpr_cln_ppl:>9.2f}%   (thr={thr_ppl:.2f})")
        print(f"{'':<20}{'PPL_W':<10}{fnr_atk_pw:>9.2f}%{fpr_cln_pw:>9.2f}%   (thr={thr_pplw:.2f})")
        print("-" * 78)

        rows.append({
            "dataset": name,
            "n_attack": len(attack),
            "n_clean": len(clean),
            "ppl_threshold": thr_ppl,
            "pplw_threshold": thr_pplw,
            "ppl_fnr_pct": fnr_atk_ppl,
            "ppl_fpr_pct": fpr_cln_ppl,
            "pplw_fnr_pct": fnr_atk_pw,
            "pplw_fpr_pct": fpr_cln_pw,
        })

        # save per-dataset clean scores for the record
        out = {
            "dataset": name,
            "ppl_model": a["ppl_model"],
            "thresholds": {"ppl": thr_ppl, "ppl_w": thr_pplw},
            "attack_records": attack,
            "clean_records": clean,
        }
        with open(Path(args.results_dir) / f"results_fnr_fpr_{prefix}_llama3b.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    out_summary = Path(args.results_dir) / "results_fnr_fpr_summary_llama3b.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote summary -> {out_summary}")


if __name__ == "__main__":
    main()
