"""
Run model-based and similarity-based defenses against PreferAttack.

This script directly supports (or challenges) the paper's stealthiness claim by
asking: can a defender who deploys a *learned* detector or a *semantic*
detector filter PreferAttack's adversarial prompts?

It reproduces the metric protocol of the paper's PPL-defense tables (Tables 9
and 10): ASR / ASR-W / ASR-Reduction for attack effectiveness under defense, and
FNR / FPR for detector quality.

Inputs (all already present in the repo):
  * PreferAttack attack results (code_judge_bench, Qwen3-VL-8B) -- the attack
    under test.
  * GCG + AutoDAN baseline results -- used to train the transfer detector.
  * Clean instructions -- used for FPR calibration and as negative class.

Outputs:
  * results/stealth_defenses_summary.json   -- machine-readable numbers
  * stdout                                  -- a paper-style summary table
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.defense.model_defense import (  # noqa: E402
    AttackRecord, ClassifierDefense, Embedder, SimilarityDefense,
    asr_under_defense, fnr_fpr,
    load_clean_instructions, load_gcg_records, load_records,
)

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #

PREFERATTACK_FILES = [
    HERE / "results" / "multi_agent_pairwise_eval_origin_opti_code_judge_bench_20260628_203045.json",
    HERE / "results" / "multi_agent_pairwise_eval_origin_opti_alpaca_eval_20260427_134843.json",
]
GCG_FILE = Path("/root/autodan/results/gcg_baseline_arena_hard_20251024_150918.json")
AUTODAN_GLOB = "/root/AutoDAN-main/results/pairwise_ga_*.json"
# checkpoint files duplicate the final results; keep only the canonical ones
AUTODAN_SKIP = ("tts", "checkpoint")

OUT_JSON = HERE / "results" / "stealth_defenses_summary.json"
TARGET_FPR = 0.01
SEED = 0


def _load_autodan() -> list[AttackRecord]:
    import glob
    out: list[AttackRecord] = []
    for f in sorted(glob.glob(AUTODAN_GLOB)):
        if any(s in f for s in AUTODAN_SKIP):
            continue
        out.extend(load_records(f, source="autodan"))
    return out


def _texts(records: list[AttackRecord], mode: str) -> list[str]:
    """Render records into raw text used for classification.

    mode:
      "suffix"             -- just the suffix (what a suffix-inspector sees)
      "attacked_instr"     -- instruction + suffix (what the judge sees)
    """
    if mode == "suffix":
        return [r.suffix for r in records]
    if mode == "attacked_instr":
        return [r.attacked_instruction for r in records]
    raise ValueError(mode)


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ----- load data -------------------------------------------------------- #
    pa_records: list[AttackRecord] = []
    for p in PREFERATTACK_FILES:
        if p.exists():
            pa_records.extend(load_records(p, source=f"preferattack:{p.stem}"))
    gcg_records = load_gcg_records(str(GCG_FILE)) if GCG_FILE.exists() else []
    # GCG records only carry question_id -- join with arena_hard instructions so
    # the similarity contrast uses a real instruction instead of the suffix itself.
    qid2instr: dict = {}
    for split in ["arena_hard_test.json", "arena_hard_train.json"]:
        p = HERE / "data" / "split" / split
        if p.exists():
            for r in json.load(open(p)):
                if isinstance(r, dict) and r.get("question_id") and r.get("instruction"):
                    qid2instr[r["question_id"]] = r["instruction"]
    if gcg_records and qid2instr:
        gcg_records = load_gcg_records(str(GCG_FILE), instruction_lookup=qid2instr)
        gcg_records = [r for r in gcg_records if r.instruction.strip()]
    ad_records = _load_autodan()

    # clean instructions (negative class / FPR calibration)
    clean_paths = [str(p) for p in PREFERATTACK_FILES if p.exists()] \
                  + [f for f in [str(GCG_FILE)] if os.path.exists(f)]
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

    # only successful PreferAttack attacks matter for ASR-under-defense
    pa_success = [r for r in pa_records if r.success]
    # for detector evaluation we use ALL preferattack suffixes (success or not)
    # but report ASR-under-defense on the successful subset.
    pa_all = pa_records

    embedder = Embedder()
    results: dict = {"data": {
        "n_preferattack": len(pa_records),
        "n_preferattack_success": len(pa_success),
        "n_gcg": len(gcg_records), "n_autodan": len(ad_records),
        "n_clean": len(clean_instr), "target_fpr": TARGET_FPR,
    }}

    # ======================================================================= #
    # EXPERIMENT A -- lightweight classifier defense                           #
    # ======================================================================= #
    print()
    print("=" * 78)
    print("EXPERIMENT A -- Classifier defense (LR over MiniLM embeddings)")
    print("=" * 78)

    # We evaluate on the SUFFIX text, since the user's scenario is a "classifier
    # trained to detect adversarial suffixes". We also report attacked_instr.
    for text_mode in ["suffix", "attacked_instr"]:
        pa_texts = _texts(pa_all, text_mode)
        clean_texts_eval = clean_instr  # clean has no suffix -> instruction text

        # ---- A1: in-distribution (strongest defense) ------------------------
        # Train AND test on PreferAttack via 5-fold CV. Best case for defender.
        n = min(len(pa_texts), len(clean_texts_eval))
        pa_cv = pa_texts[:n]
        clean_cv = clean_texts_eval[:n]
        clf_cv = ClassifierDefense(embedder, variant="indist", target_fpr=TARGET_FPR)
        cv = clf_cv.cv_auc(pa_cv, clean_cv)

        # ---- A2: transfer detector (realistic deployment) -------------------
        # Train on GCG + AutoDAN suffixes; test on PreferAttack suffixes.
        other = [r.suffix for r in (gcg_records + ad_records) if r.suffix.strip()]
        # balance training negatives to positives
        n_train_pos = len(other)
        idx = rng.choice(len(clean_instr), size=min(n_train_pos, len(clean_instr)), replace=False)
        train_neg = [clean_instr[i] for i in idx]
        transfer = ClassifierDefense(embedder, variant="transfer", target_fpr=TARGET_FPR)
        transfer.fit(other, train_neg)
        transfer.calibrate(clean_instr)  # set threshold at FPR=1% on clean

        # detector quality on PreferAttack
        pa_proba = transfer.score(pa_texts)
        pa_flagged = transfer.flag(pa_texts)
        det = fnr_fpr([1] * len(pa_texts), pa_flagged)
        # contrast: how well does the same transfer detector catch GCG/AutoDAN?
        if other:
            other_proba = transfer.score(other)
            # rebuild clean-aligned other set for FPR/FNR on the training-domain attack
            n2 = min(len(other), len(clean_instr))
            other_eval = other[:n2]
            other_flagged = transfer.flag(other_eval)
            train_det = fnr_fpr([1] * len(other_eval), other_flagged)
        else:
            train_det = {"fnr": float("nan"), "fpr": float("nan"), "n_adv": 0, "n_clean": 0}

        # ASR under defense: use the same flag decision but on the successful subset
        # (re-score to keep alignment with pa_success)
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
                "on_gcg_autodan": train_det,
                "threshold": transfer.threshold,
            },
        }

    # ======================================================================= #
    # EXPERIMENT B -- semantic similarity defense                              #
    # ======================================================================= #
    print()
    print("=" * 78)
    print("EXPERIMENT B -- Semantic similarity defense")
    print("=" * 78)

    # A benign modification a user might legitimately append to their instruction.
    BENIGN_SUFFIX = " Please answer the above concisely and in English."
    # For suffix_vs_instr (which assumes the suffix can be isolated), the right
    # "clean" anchor is the similarity between a clean instruction and *generic
    # unrelated text*. We approximate that by pairing each clean instruction
    # with a different, shuffled clean instruction.
    rng_cross = np.random.default_rng(SEED + 1)
    shuffled = list(clean_instr)
    rng_cross.shuffle(shuffled)
    cross_pairs = shuffled
    for variant in ["prompt_drift", "suffix_vs_instr"]:
        sim = SimilarityDefense(embedder, variant=variant, target_fpr=TARGET_FPR)
        if variant == "prompt_drift":
            sim.calibrate_benign(clean_instr, BENIGN_SUFFIX)
        else:
            sim.calibrate_benign(clean_instr, cross_pairs)

        pa_instr = [r.instruction for r in pa_all]
        pa_suf = [r.suffix for r in pa_all]
        pa_score = sim.score(pa_instr, pa_suf)
        pa_flagged = sim.flag(pa_instr, pa_suf)
        det = fnr_fpr([1] * len(pa_instr), pa_flagged)
        auc_b = sim.auc_against_clean(pa_instr, pa_suf, clean_instr, BENIGN_SUFFIX)

        # contrast on GCG (gibberish suffixes -> should be very dissimilar)
        if gcg_records:
            g_instr = [r.instruction or r.suffix for r in gcg_records]
            g_suf = [r.suffix for r in gcg_records]
            g_flagged = sim.flag(g_instr, g_suf)
            g_det = fnr_fpr([1] * len(g_instr), g_flagged)
        else:
            g_det = {"fnr": float("nan"), "fpr": float("nan"), "n_adv": 0, "n_clean": 0,
                     "n_flagged_adv": 0}

        # ASR under defense on successful PreferAttack subset
        ps_instr = [r.instruction for r in pa_success]
        ps_suf = [r.suffix for r in pa_success]
        ps_flagged = sim.flag(ps_instr, ps_suf) if ps_suf else np.array([], bool)
        asr_b = asr_under_defense(pa_success, ps_flagged)

        print(f"\n  [variant = {variant}]  (threshold={sim.threshold:.4f}, FPR on benign appends <= {TARGET_FPR:.0%} by construction)")
        print(f"    PreferAttack : FNR@1%FPR = {det['fnr']:.3f}  "
              f"AUC(vs benign append) = {auc_b:.3f}  (caught {det['n_flagged_adv']}/{det['n_adv']})")
        print(f"    GCG contrast : FNR = {g_det['fnr']:.3f}  "
              f"(caught {g_det.get('n_flagged_adv',0)}/{g_det.get('n_adv',0)})")
        print(f"    median anomaly score : PreferAttack={np.median(pa_score):.4f}")
        print(f"    ASR under defense : ASR={asr_b['asr']:.4f}  ASR-W={asr_b['asr_w']:.4f}  "
              f"ASR-Reduction={asr_b['asr_reduction']:.4f}")

        results[f"similarity_{variant}"] = {
            "threshold": sim.threshold, "benign_suffix": BENIGN_SUFFIX,
            "on_preferattack": {**det, "auc": auc_b, "asr_under_defense": asr_b,
                                "median_anomaly_score": float(np.median(pa_score))},
            "on_gcg_contrast": g_det,
        }

    # ----- persist ---------------------------------------------------------- #
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("=" * 78)
    print(f"WROTE  {OUT_JSON}")
    print("=" * 78)


if __name__ == "__main__":
    main()
