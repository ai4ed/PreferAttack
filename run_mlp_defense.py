"""
Standalone runner for the MLP adversarial-suffix detector.

Mirrors the classifier portion of ``run_stealth_defenses.py`` (the LR baseline)
but uses ``MLPClassifierDefense``. The LR script is left untouched; this script
writes its own result file and prints an LR-vs-MLP comparison at the end by
reading the LR summary JSON produced by ``run_stealth_defenses.py``.

Protocol is identical to the LR experiment so the two are directly comparable:
  * A1 -- 5-fold CV on PreferAttack-vs-clean (strongest position for defender)
  * A2 -- transfer detector trained on GCG+AutoDAN, tested on PreferAttack
Both evaluated on two text views: isolated "suffix" (oracle) and full
"attacked_instr" (what the judge actually sees).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.defense.model_defense import (  # noqa: E402
    Embedder, asr_under_defense, fnr_fpr,
    load_clean_instructions, load_gcg_records, load_records,
)
from src.defense.mlp_defense import MLPClassifierDefense  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths (same data as the LR experiment)                                      #
# --------------------------------------------------------------------------- #
PREFERATTACK_FILES = [
    HERE / "results" / "multi_agent_pairwise_eval_origin_opti_code_judge_bench_20260628_203045.json",
    HERE / "results" / "multi_agent_pairwise_eval_origin_opti_alpaca_eval_20260427_134843.json",
]
GCG_FILE = Path("/root/autodan/results/gcg_baseline_arena_hard_20251024_150918.json")
AUTODAN_GLOB = "/root/AutoDAN-main/results/pairwise_ga_*.json"
AUTODAN_SKIP = ("tts", "checkpoint")

OUT_JSON = HERE / "results" / "mlp_defense_summary.json"
LR_JSON = HERE / "results" / "stealth_defenses_summary.json"   # for comparison printout
TARGET_FPR = 0.01
SEED = 0


def _load_autodan():
    out = []
    for f in sorted(glob.glob(AUTODAN_GLOB)):
        if any(s in f for s in AUTODAN_SKIP):
            continue
        out.extend(load_records(f, source="autodan"))
    return out


def _texts(records, mode):
    if mode == "suffix":
        return [r.suffix for r in records]
    return [r.attacked_instruction for r in records]


def main():
    rng = np.random.default_rng(SEED)

    # ----- load data (identical to LR experiment) -------------------------- #
    pa_records = []
    for p in PREFERATTACK_FILES:
        if p.exists():
            pa_records.extend(load_records(p, source=f"preferattack:{p.stem}"))
    qid2instr = {}
    for split in ["arena_hard_test.json", "arena_hard_train.json"]:
        p = HERE / "data" / "split" / split
        if p.exists():
            for r in json.load(open(p)):
                if isinstance(r, dict) and r.get("question_id") and r.get("instruction"):
                    qid2instr[r["question_id"]] = r["instruction"]
    gcg_records = []
    if GCG_FILE.exists():
        gcg_records = load_gcg_records(str(GCG_FILE), instruction_lookup=qid2instr)
        gcg_records = [r for r in gcg_records if r.instruction.strip()]
    ad_records = _load_autodan()
    clean_paths = [str(p) for p in PREFERATTACK_FILES if p.exists()] \
                  + ([str(GCG_FILE)] if GCG_FILE.exists() else [])
    clean_instr = load_clean_instructions(clean_paths, cap=2000, seed=SEED)

    print("=" * 78)
    print("DATA LOADED")
    print("=" * 78)
    print(f"  PreferAttack records : {len(pa_records)}  "
          f"(success={sum(r.success for r in pa_records)})")
    print(f"  GCG records          : {len(gcg_records)}")
    print(f"  AutoDAN records      : {len(ad_records)}")
    print(f"  Clean instructions   : {len(clean_instr)}")
    if not pa_records:
        print("No PreferAttack records found -- aborting.")
        return

    pa_success = [r for r in pa_records if r.success]
    pa_all = pa_records
    embedder = Embedder()
    results = {"classifier": "MLP", "data": {
        "n_preferattack": len(pa_records), "n_preferattack_success": len(pa_success),
        "n_gcg": len(gcg_records), "n_autodan": len(ad_records),
        "n_clean": len(clean_instr), "target_fpr": TARGET_FPR,
        "hidden_layers": [128, 64], "alpha": 1e-3,
    }}

    print()
    print("=" * 78)
    print("MLP CLASSIFIER DEFENSE  (2-layer MLP over MiniLM embeddings)")
    print("=" * 78)

    for text_mode in ["suffix", "attacked_instr"]:
        pa_texts = _texts(pa_all, text_mode)

        # ---- A1: in-distribution 5-fold CV -------------------------------- #
        n = min(len(pa_texts), len(clean_instr))
        cv = MLPClassifierDefense(embedder, target_fpr=TARGET_FPR).cv_auc(
            pa_texts[:n], clean_instr[:n])

        # ---- A2: transfer detector --------------------------------------- #
        other = [r.suffix for r in (gcg_records + ad_records) if r.suffix.strip()]
        idx = rng.choice(len(clean_instr), size=min(len(other), len(clean_instr)),
                         replace=False)
        train_neg = [clean_instr[i] for i in idx]
        transfer = MLPClassifierDefense(embedder, target_fpr=TARGET_FPR)
        transfer.fit(other, train_neg)
        transfer.calibrate(clean_instr)

        pa_flagged = transfer.flag(pa_texts)
        det = fnr_fpr([1] * len(pa_texts), pa_flagged)
        n2 = min(len(other), len(clean_instr))
        train_det = fnr_fpr([1] * n2, transfer.flag(other[:n2])) if other else \
            {"fnr": float("nan"), "n_flagged_adv": 0, "n_adv": 0}

        pa_succ_texts = _texts(pa_success, text_mode)
        pa_succ_flagged = transfer.flag(pa_succ_texts) if pa_succ_texts else np.array([], bool)
        asr_a2 = asr_under_defense(pa_success, pa_succ_flagged)

        print(f"\n  [text = {text_mode}]")
        print(f"    A1 in-dist 5-fold CV  : AUC = {cv['auc_mean']:.3f} "
              f"(+/-{cv['auc_std']:.3f}) | FNR@1%FPR = {cv['fnr@1%fpr_mean']:.3f}")
        print(f"    A2 transfer detector  : trained on GCG+AutoDAN ({len(other)} pos), "
              f"FPR on clean <= {TARGET_FPR:.0%} by construction")
        print(f"      -> on PreferAttack  : FNR@1%FPR = {det['fnr']:.3f}  "
              f"(caught {det['n_flagged_adv']}/{det['n_adv']})")
        print(f"      -> on GCG+AutoDAN   : FNR = {train_det['fnr']:.3f}  "
              f"(caught {train_det.get('n_flagged_adv',0)}/{train_det.get('n_adv',0)})")
        print(f"    ASR under A2 defense  : ASR={asr_a2['asr']:.4f}  "
              f"ASR-W={asr_a2['asr_w']:.4f}  ASR-Reduction={asr_a2['asr_reduction']:.4f}")

        results[f"classifier_{text_mode}"] = {
            "A1_indist_cv": cv,
            "A2_transfer": {
                "train_positives": len(other), "train_clean": len(train_neg),
                "on_preferattack": {**det, "asr_under_defense": asr_a2},
                "on_gcg_autodan": train_det, "threshold": transfer.threshold,
            },
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*78}\nWROTE  {OUT_JSON}\n{'='*78}")

    # ----- LR vs MLP comparison (read-only on the LR result) --------------- #
    if LR_JSON.exists():
        lr = json.load(open(LR_JSON))
        print()
        print("=" * 78)
        print("LR vs MLP  (FNR@1%FPR -- higher = more PreferAttack evasion; "
              "AUC -- lower = harder to detect)")
        print("=" * 78)
        hdr = f"  {'text view':<14} {'metric':<22} {'LR':>10} {'MLP':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for tm in ["suffix", "attacked_instr"]:
            lrblk = lr.get(f"classifier_{tm}", {})
            mlblk = results.get(f"classifier_{tm}", {})
            lr_a1 = lrblk.get("A1_indist_cv", {})
            ml_a1 = mlblk.get("A1_indist_cv", {})
            lr_a2 = lrblk.get("A2_transfer", {}).get("on_preferattack", {})
            ml_a2 = mlblk.get("A2_transfer", {}).get("on_preferattack", {})
            lr_asr = lr_a2.get("asr_under_defense", {})
            ml_asr = ml_a2.get("asr_under_defense", {})
            print(f"  {tm:<14} {'A1 CV AUC':<22} "
                  f"{lr_a1.get('auc_mean',float('nan')):>10.3f} "
                  f"{ml_a1.get('auc_mean',float('nan')):>10.3f}")
            print(f"  {tm:<14} {'A1 CV FNR@1%FPR':<22} "
                  f"{lr_a1.get('fnr@1%fpr_mean',float('nan')):>10.3f} "
                  f"{ml_a1.get('fnr@1%fpr_mean',float('nan')):>10.3f}")
            print(f"  {tm:<14} {'A2 transfer FNR':<22} "
                  f"{lr_a2.get('fnr',float('nan')):>10.3f} "
                  f"{ml_a2.get('fnr',float('nan')):>10.3f}")
            print(f"  {tm:<14} {'A2 ASR-Reduction':<22} "
                  f"{lr_asr.get('asr_reduction',float('nan')):>10.4f} "
                  f"{ml_asr.get('asr_reduction',float('nan')):>10.4f}")
            print()
    else:
        print(f"\n(skipping LR-vs-MLP comparison: {LR_JSON} not found -- "
              f"run run_stealth_defenses.py first to produce it)")


if __name__ == "__main__":
    main()
