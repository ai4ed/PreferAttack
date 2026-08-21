"""
PRED: PREference-Robust Ensemble Defense

Three independent detection gates, each calibrated on clean data to FPR <= alpha,
combined under either OR (permissive filtering, higher FPR) or AND (conservative).

Gates:
  S_w  : absolute windowed PPL of the judged text  (legacy PPL-W baseline)
  S_l  : local anomaly factor = ppl_w / ppl  (max-window NLL / mean NLL)
  S_r  : relative ratio = ppl_w(attacked) / ppl_w(clean-baseline-matched)

Calibration: per-gate thresholds on clean data with FPR-based selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PREDThresholds:
    thr_w: float          # S_w threshold
    thr_l: float          # S_l threshold
    thr_r: float          # S_r threshold
    target_fpr: float
    clean_fpr_w: float
    clean_fpr_l: float
    clean_fpr_r: float


def _safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return float("nan")


def fpr_threshold(scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Return (threshold, actual_fpr) s.t. mean(scores >= threshold) <= target_fpr."""
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        return float("inf"), 0.0
    unique = np.unique(valid)
    candidates = [float("inf")] + sorted(unique.tolist(), reverse=True) + [float("-inf")]
    chosen_thr = candidates[0]
    chosen_fpr = 0.0
    for thr in candidates:
        fpr = float(np.mean(valid >= thr))
        if fpr <= target_fpr:
            chosen_thr = float(thr)
            chosen_fpr = fpr
        else:
            break
    return chosen_thr, chosen_fpr


def calibrate(clean_w: np.ndarray, clean_ppl: np.ndarray, *, target_fpr: float) -> PREDThresholds:
    """Calibrate the three PRED gates on clean samples.

    S_w uses clean_w directly (per-sample windowed PPL of the judged text).
    S_l uses clean_w / clean_ppl (local anomaly factor on clean text).
    S_r is set to 1.0 by default (no matched baseline in pure clean calibration);
        if matched clean-vs-clean ratio is provided externally, override thr_r.
    """
    clean_w = np.asarray(clean_w, dtype=float)
    clean_ppl = np.asarray(clean_ppl, dtype=float)
    mask = np.isfinite(clean_w) & np.isfinite(clean_ppl) & (clean_ppl > 0) & (clean_w > 0)
    clean_w = clean_w[mask]
    clean_ppl = clean_ppl[mask]
    clean_lafl = clean_w / clean_ppl

    thr_w, fpr_w = fpr_threshold(clean_w, target_fpr)
    thr_l, fpr_l = fpr_threshold(clean_lafl, target_fpr)
    # S_r ratio: 1.0 by default, since clean-vs-clean ratio centers at ~1
    thr_r = 1.0
    fpr_r = 0.0
    return PREDThresholds(
        thr_w=thr_w, thr_l=thr_l, thr_r=thr_r,
        target_fpr=target_fpr,
        clean_fpr_w=fpr_w, clean_fpr_l=fpr_l, clean_fpr_r=fpr_r,
    )


def predict_gate(score: float, thr: float) -> bool:
    """Return True if the gate is triggered (sample flagged as adversarial)."""
    if not np.isfinite(score) or score <= 0:
        return False
    return score > thr


def predict(pred_thresholds: PREDThresholds, *,
            s_w: float, s_l: float, s_r: Optional[float] = None,
            mode: str = "or") -> bool:
    """Return True if the sample is flagged as adversarial under the chosen mode.

    mode = "or"  : any gate triggers -> flag (catches more attacks, higher FPR)
    mode = "and" : all gates trigger -> flag (lower FPR, lower catch rate)
    mode = "w_or_l" : S_w OR S_l (skip S_r, since S_r needs matched baseline)
    mode = "w_only" : S_w alone (legacy PPL-W baseline, for head-to-head comparison)
    mode = "l_only" : S_l alone
    mode = "r_only" : S_r alone
    mode = "w_or_l" : S_w OR S_l
    mode = "w_or_r" : S_w OR S_r
    mode = "l_or_r" : S_l OR S_r
    mode = "majority" : >=2 gates trigger
    """
    g_w = predict_gate(s_w, pred_thresholds.thr_w)
    g_l = predict_gate(s_l, pred_thresholds.thr_l)
    if s_r is None or not np.isfinite(s_r) or s_r <= 0:
        g_r = False
    else:
        g_r = predict_gate(s_r, pred_thresholds.thr_r)

    if mode == "or":
        return g_w or g_l or g_r
    if mode == "and":
        return g_w and g_l and g_r
    if mode == "w_only":
        return g_w
    if mode == "l_only":
        return g_l
    if mode == "r_only":
        return g_r
    if mode == "w_or_l":
        return g_w or g_l
    if mode == "w_or_r":
        return g_w or g_r
    if mode == "l_or_r":
        return g_l or g_r
    if mode == "majority":
        return sum([g_w, g_l, g_r]) >= 2
    raise ValueError(f"unknown mode: {mode}")


def compute_signals(attack_rec: dict, clean_rec: dict) -> dict[str, float]:
    """Compute the three PRED signals from one (attack, clean) matched pair.

    Assumes the JSON layout produced by compute_fnr_fpr.py:
        attack_rec keys: ppl_full, ppl_w_full, ppl_instr, ppl_w_instr
        clean_rec  keys: ppl,      ppl_w
    """
    s_w = _safe_float(attack_rec.get("ppl_w_full"))
    ppl_w_full = s_w
    ppl_full = _safe_float(attack_rec.get("ppl_full"))
    s_l = ppl_w_full / ppl_full if np.isfinite(ppl_w_full) and np.isfinite(ppl_full) and ppl_full > 0 else float("nan")
    clean_w = _safe_float(clean_rec.get("ppl_w"))
    s_r = ppl_w_full / clean_w if np.isfinite(ppl_w_full) and np.isfinite(clean_w) and clean_w > 0 else float("nan")
    return {"S_w": s_w, "S_l": s_l, "S_r": s_r}


def evaluate(fnr_fpr_json: str, *,
             target_fpr: float = 0.01,
             modes: tuple[str, ...] = ("w_only", "l_only", "r_only", "or", "and", "w_or_l", "w_or_r", "l_or_r", "majority"),
             asr_w_json: Optional[str] = None) -> dict:
    """Run PRED evaluation on a single fnr_fpr_*.json file.

    If asr_w_json is given (pointing to the matching results_asr_w_*_llama3b.json),
    the S_w threshold is taken from that file (100-sample train calibration in the paper)
    so that 'w_only' reproduces the paper's ASR-W exactly. S_l and S_r are still
    calibrated on the 421-sample clean test set inside fnr_fpr_json.
    """
    with open(fnr_fpr_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    attack = d["attack_records"]
    clean = d["clean_records"]
    n_attack = len(attack)
    n_clean = len(clean)
    n_pairs = min(n_attack, n_clean)
    base_success = sum(1 for r in attack[:n_pairs] if str(r.get("success_no_defense")).lower() == "true")
    asr = base_success / n_pairs if n_pairs else 0.0

    # Build per-pair signals
    rows = []
    for i in range(n_pairs):
        a = attack[i]
        c = clean[i]
        sig = compute_signals(a, c)
        rows.append({"i": i, "success": str(a.get("success_no_defense")).lower() == "true", **sig})

    # Calibrate on clean (S_w, S_l from clean directly; S_r ratio defaults to 1.0)
    clean_w_arr = np.array([_safe_float(r.get("ppl_w")) for r in clean[:n_pairs]], dtype=float)
    clean_ppl_arr = np.array([_safe_float(r.get("ppl")) for r in clean[:n_pairs]], dtype=float)
    thresholds = calibrate(clean_w_arr, clean_ppl_arr, target_fpr=target_fpr)

    # For S_r on clean: approximate as clean_w_i / median(clean_w) to capture whether
    # a sample's windowed PPL deviates far from the typical clean level (matched-baseline proxy)
    median_clean_w = float(np.median(clean_w_arr[np.isfinite(clean_w_arr)])) if np.any(np.isfinite(clean_w_arr)) else 1.0
    # Override thr_r: calibrate on clean using clean_w / median_clean_w to keep clean FPR <= target
    clean_s_r_proxy = clean_w_arr / max(median_clean_w, 1e-9)
    thr_r, fpr_r = fpr_threshold(clean_s_r_proxy[np.isfinite(clean_s_r_proxy)], target_fpr)
    thresholds.thr_r = thr_r
    thresholds.clean_fpr_r = fpr_r

    # If a matching asr_w_json is provided, override thr_w with the paper's calibration
    paper_thr_w = None
    paper_thr_ppl = None
    if asr_w_json:
        with open(asr_w_json, "r", encoding="utf-8") as f:
            asw = json.load(f)
        s = asw.get("summary", {})
        paper_thr_w = s.get("threshold_pplw_full")
        paper_thr_ppl = s.get("threshold_ppl_full")
        if paper_thr_w and np.isfinite(float(paper_thr_w)):
            thresholds.thr_w = float(paper_thr_w)
        # We do NOT override thr_l (it's a ratio); S_r stays at its clean-test calibration.

    # Compute per-mode ASR-PRED and FPR
    out = {
        "file": str(Path(fnr_fpr_json).name),
        "dataset": d.get("dataset"),
        "n_attack": n_attack,
        "n_clean": n_clean,
        "n_pairs": n_pairs,
        "base_success": base_success,
        "asr_no_defense_pct": asr * 100.0,
        "thresholds": {
            "thr_w": thresholds.thr_w,
            "thr_l": thresholds.thr_l,
            "thr_r": thresholds.thr_r,
            "median_clean_w": median_clean_w,
            "paper_thr_w": paper_thr_w,
            "paper_thr_ppl": paper_thr_ppl,
            "target_fpr": target_fpr,
            "clean_fpr_w_pct": thresholds.clean_fpr_w * 100.0,
            "clean_fpr_l_pct": thresholds.clean_fpr_l * 100.0,
            "clean_fpr_r_pct": thresholds.clean_fpr_r * 100.0,
        },
        "modes": {},
    }

    for mode in modes:
        # Attack side: among base_success samples, how many survive (not flagged)?
        survived = 0
        flagged_atk = 0
        for row in rows:
            if not row["success"]:
                continue
            triggered = predict(thresholds, s_w=row["S_w"], s_l=row["S_l"], s_r=row["S_r"], mode=mode)
            if triggered:
                flagged_atk += 1
            else:
                survived += 1
        asr_pred = survived / n_pairs if n_pairs else 0.0
        asr_reduction = (asr - asr_pred) / asr * 100.0 if asr > 0 else 0.0

        # Clean side FPR
        clean_flagged = 0
        for i in range(n_pairs):
            cw = _safe_float(clean[i].get("ppl_w"))
            cp = _safe_float(clean[i].get("ppl"))
            cl = cw / cp if np.isfinite(cw) and np.isfinite(cp) and cp > 0 else float("nan")
            cr = cw / max(median_clean_w, 1e-9) if np.isfinite(cw) else float("nan")
            if predict(thresholds, s_w=cw, s_l=cl, s_r=cr, mode=mode):
                clean_flagged += 1
        clean_fpr = clean_flagged / n_pairs if n_pairs else 0.0

        out["modes"][mode] = {
            "asr_pred_pct": asr_pred * 100.0,
            "asr_reduction_pct": asr_reduction,
            "attack_flagged": flagged_atk,
            "attack_survived": survived,
            "clean_fpr_pct": clean_fpr * 100.0,
        }

    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="paths to results_fnr_fpr_*_llama3b.json")
    ap.add_argument("--target_fpr", type=float, default=0.01)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--asr_w_dir", default="/root/PreferAttack",
                    help="directory containing results_asr_w_*_llama3b.json; "
                         "filename mapping: fnr_fpr_<prefix>_llama3b.json -> asr_w_<prefix>_llama3b.json")
    args = ap.parse_args()

    all_out = []
    for path in args.inputs:
        print(f"\n=== PRED evaluation: {path} ===")
        # Try to find matching asr_w json (paper calibration)
        asr_w_json = None
        fname = Path(path).name
        if fname.startswith("results_fnr_fpr_") and fname.endswith("_llama3b.json"):
            prefix = fname[len("results_fnr_fpr_"):-len("_llama3b.json")]
            cand = Path(args.asr_w_dir) / f"results_asr_w_{prefix}_llama3b.json"
            if cand.exists():
                asr_w_json = str(cand)
                print(f"  Matched paper-calibration asr_w file: {asr_w_json}")
        res = evaluate(path, target_fpr=args.target_fpr, asr_w_json=asr_w_json)
        all_out.append(res)

        print(f"  Dataset: {res['dataset']}, pairs: {res['n_pairs']}, "
              f"base_success: {res['base_success']}, ASR: {res['asr_no_defense_pct']:.2f}%")
        print(f"  Thresholds: thr_w={res['thresholds']['thr_w']:.2f}"
              f"{'(paper)' if res['thresholds'].get('paper_thr_w') else ''}, "
              f"thr_l={res['thresholds']['thr_l']:.2f}, "
              f"thr_r={res['thresholds']['thr_r']:.3f}, "
              f"median_clean_w={res['thresholds']['median_clean_w']:.2f}")
        print(f"  Per-gate clean FPR: S_w={res['thresholds']['clean_fpr_w_pct']:.2f}%, "
              f"S_l={res['thresholds']['clean_fpr_l_pct']:.2f}%, "
              f"S_r={res['thresholds']['clean_fpr_r_pct']:.2f}%")
        asr = res["asr_no_defense_pct"]
        print(f"  {'mode':<10}{'ASR-PRED':>12}{'ASR-Red':>12}{'atk_flagged':>14}{'clean_FPR':>12}")
        for mode, mres in res["modes"].items():
            print(f"  {mode:<10}{mres['asr_pred_pct']:>11.2f}%{mres['asr_reduction_pct']:>11.2f}%"
                  f"{mres['attack_flagged']:>14}{mres['clean_fpr_pct']:>11.2f}%")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_out, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Wrote PRED results -> {args.output_json}")


if __name__ == "__main__":
    main()
