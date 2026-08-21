"""
Tail-Consistency Defense (TCD) for PreferAttack.

Core idea (see TCD_DEFENSE_DESIGN.md): instead of asking "does this suffix look
adversarial?" (anomaly detection -- which all prior defenses do, and which fails
because PreferAttack explicitly optimizes for stealth), TCD asks "does the
appended text have *causal* influence over the judge's preference?"

Mechanism: re-query the judge after truncating the last K tokens of the
attacked instruction. For a benign instruction, the preference is determined by
(A, B) quality and is largely truncation-robust. For a PreferAttack-style
append-only attack, the entire attack payload lives at the tail, so any
sufficiently large K reverts the preference to the baseline.

This module is judge-agnostic: it works with any object exposing the
PairwiseExample-based judge_pairwise / judge_examples interface (VLLMJudge or
the OpenAI-compatible judge). Token-level truncation uses the judge's own
tokenizer so the truncation is consistent with what the judge actually sees.

Public API:
    TailConsistencyDefense
        .evaluate_sample(instruction, suffix, response_a, response_b)
            -> dict with pref_full, prefs_trunc, agreement, flag
        .calibrate_threshold(clean_pairs)
            -> sets self.threshold at FPR <= target_fpr on clean data

Reuses asr_under_defense / fnr_fpr from src.defense.model_defense so the
metric protocol matches the paper's §5.11 PPL-defense tables.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Shared utilities (inlined to keep TCD dependency-light: no sklearn, no       #
# sentence-transformers required). Mirrors src.defense.model_defense exactly. #
# --------------------------------------------------------------------------- #

def build_attacked_instruction(instruction: str, suffix: str) -> str:
    """Mirror PreferAttack's suffix-appending rule (Multi_Agent_Framework.py:20)."""
    return instruction.rstrip() + " " + suffix


@dataclass
class AttackRecord:
    """A single (instruction, suffix) pair with its attack outcome.

    Kept structurally identical to src.defense.model_defense.AttackRecord so
    metric computations are interchangeable.
    """
    instruction: str
    suffix: str
    success: bool
    orig_choice: Optional[int] = None
    new_choice: Optional[int] = None
    source: str = ""

    @property
    def attacked_instruction(self) -> str:
        return build_attacked_instruction(self.instruction, self.suffix)


def asr_under_defense(records: Sequence[AttackRecord],
                      flagged: Sequence[bool]) -> Dict[str, float]:
    """Replicate the paper's ASR / ASR-W / ASR-Reduction protocol.

    A successful attack that gets flagged by the defense is assumed to be
    reverted to its original preference (i.e. the attack fails under defense).
    Mirrors the PPL-defense evaluation in the paper (Table 9).
    """
    n = len(records)
    if n == 0:
        return {"asr": 0.0, "asr_w": 0.0, "asr_reduction": 0.0, "n": 0}
    base_success = np.array([r.success for r in records], dtype=bool)
    flagged = np.asarray(flagged, dtype=bool)
    survived = base_success & (~flagged)
    asr = float(base_success.mean())
    asr_w = float(survived.mean())
    return {
        "n": n,
        "asr": asr,
        "asr_w": asr_w,
        "asr_reduction": asr - asr_w,
        "n_success": int(base_success.sum()),
        "n_flagged_of_success": int((flagged & base_success).sum()),
    }


def fnr_fpr(y_true: Sequence[int], flagged: Sequence[bool]) -> Dict[str, float]:
    """FNR (attacks missed) / FPR (clean flagged). y_true: 1=adversarial, 0=clean."""
    y = np.asarray(y_true, dtype=int)
    f = np.asarray(flagged, dtype=bool)
    adv = y == 1
    clean = y == 0
    fnr = float(1.0 - f[adv].mean()) if adv.any() else float("nan")
    fpr = float(f[clean].mean()) if clean.any() else float("nan")
    return {
        "fnr": fnr, "fpr": fpr,
        "n_adv": int(adv.sum()), "n_clean": int(clean.sum()),
        "n_flagged_adv": int((f & adv).sum()),
        "n_flagged_clean": int((f & clean).sum()),
    }


@dataclass
class CleanPair:
    """A clean pairwise query (instruction, A, B) with no adversarial suffix."""
    instruction: str
    response_a: str
    response_b: str
    question_id: str = ""


@dataclass
class TailConsistencyDefense:
    """
    TCD defense: flag a pairwise query as adversarial if the judge's preference
    is unstable under tail-truncation of the (attacked) instruction.

    Parameters
    ----------
    judge : object
        Anything exposing ``judge_pairwise(example)`` and ``judge_examples(
        examples, batch_size=, ...)`` returning a JudgeResponse with a
        ``preference`` field (0=A, 1=B). Both VLLMJudge and the OpenAI judge
        satisfy this.
    K_set : tuple[int, ...]
        Truncation lengths (in tokens) tried per sample. Default covers short,
        medium, and long suffixes; multi-K gives robustness against unknown
        suffix length.
    target_fpr : float
        Desired false-positive rate on clean data; threshold is calibrated to
        satisfy this (paper protocol: FPR <= 1%).
    min_len_tokens : int
        Hard floor on the truncated instruction length. Skip any K that would
        cut below this.
    min_len_ratio : float
        Relative floor: never truncate below min_len_ratio * original_len.
    threshold : float
        Agreement threshold below which a sample is flagged. Calibrated by
        ``calibrate_threshold`` on clean data.
    """

    judge: object
    K_set: Tuple[int, ...] = (30, 60, 120, 200)
    target_fpr: float = 0.01
    min_len_tokens: int = 20
    min_len_ratio: float = 0.3
    threshold: float = 0.5  # default; overwritten by calibrate_threshold
    tokenizer: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = self._resolve_tokenizer()

    # ------------------------------------------------------------------ #
    # Tokenizer                                                          #
    # ------------------------------------------------------------------ #

    def _resolve_tokenizer(self):
        """Best-effort: pull the tokenizer out of the judge object."""
        # VLLMJudge path
        tok = getattr(self.judge, "_get_tokenizer", None)
        if callable(tok):
            try:
                t = tok()
                if t is not None:
                    return t
            except Exception:
                pass
        # Direct attribute
        for attr in ("tokenizer", "_tokenizer"):
            t = getattr(self.judge, attr, None)
            if t is not None:
                return t
        return None

    def _encode(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("TCD needs a tokenizer to do token-level truncation.")
        # HF tokenizer path
        if hasattr(self.tokenizer, "encode"):
            return self.tokenizer.encode(text, add_special_tokens=False)
        # tiktoken-style
        if hasattr(self.tokenizer, "encode_batch"):
            return self.tokenizer.encode_batch([text])[0]
        raise RuntimeError("Unrecognized tokenizer interface.")

    def _decode(self, ids: Sequence[int]) -> str:
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(list(ids), skip_special_tokens=True)
        # tiktoken
        if hasattr(self.tokenizer, "decode_batch"):
            return self.tokenizer.decode_batch([list(ids)])[0]
        raise RuntimeError("Unrecognized tokenizer interface.")

    def _truncate_text(self, text: str, K: int) -> Optional[str]:
        """Drop the last K tokens; return None if result would be too short."""
        try:
            ids = self._encode(text)
        except Exception:
            return None
        if len(ids) <= K:
            return None
        keep_n = len(ids) - K
        # Respect both absolute and relative floors
        floor = max(self.min_len_tokens, int(len(ids) * self.min_len_ratio))
        if keep_n < floor:
            keep_n = floor
            if keep_n >= len(ids):
                return None
        try:
            return self._decode(ids[:keep_n])
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Judge wrapper                                                      #
    # ------------------------------------------------------------------ #

    def _build_example(self, instruction: str, response_a: str, response_b: str,
                       qid: str = ""):
        """Build a PairwiseExample. Imports locally to avoid hard dep at import time."""
        from utils.data_types import PairwiseExample
        return PairwiseExample(
            question_id=qid,
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
            model_a="",
            model_b="",
        )

    def _judge_batch(self, examples) -> List[Optional[int]]:
        """Run judge on a batch; return list of preference (0/1) or None on failure.

        Uses judge_examples if available (much faster on vLLM), else falls back
        to per-sample judge_pairwise.
        """
        if not examples:
            return []
        judge_examples = getattr(self.judge, "judge_examples", None)
        if callable(judge_examples):
            try:
                resps = judge_examples(examples, batch_size=min(8, len(examples)))
                return [getattr(r, "preference", None) for r in resps]
            except Exception:
                pass
        # Fallback: per-sample (slower)
        judge_pairwise = getattr(self.judge, "judge_pairwise", None)
        out: List[Optional[int]] = []
        for ex in examples:
            try:
                r = judge_pairwise(ex)
                out.append(getattr(r, "preference", None))
            except Exception:
                out.append(None)
        return out

    # ------------------------------------------------------------------ #
    # Core evaluation                                                    #
    # ------------------------------------------------------------------ #

    def evaluate_sample(self, instruction: str, suffix: str,
                        response_a: str, response_b: str,
                        qid: str = "") -> Dict:
        """
        Run TCD on one (instruction, suffix, A, B) tuple.

        Returns a dict:
            pref_full     : int or None  -- judge preference on full attacked prompt
            prefs_trunc   : list[int|None] -- preference at each K (None if K skipped)
            K_used        : list[int]       -- K values actually evaluated
            agreement     : float in [0,1] -- fraction of truncs matching pref_full
            flag          : bool           -- agreement < threshold (causal inconsistency)
            trunc_texts   : list[str]       -- the actual truncated instructions
                                              (for debugging / analysis)
        """
        # Reconstruct the attacked instruction exactly as PreferAttack does
        if suffix and suffix.strip():
            attacked = build_attacked_instruction(instruction, suffix)
        else:
            attacked = instruction

        # Build the full + truncated examples
        ex_full = self._build_example(attacked, response_a, response_b, qid=qid)
        trunc_examples = []
        K_used = []
        trunc_texts = []
        for K in self.K_set:
            trunc = self._truncate_text(attacked, K)
            if trunc is None or len(trunc.strip()) < 5:
                continue
            trunc_examples.append(self._build_example(trunc, response_a, response_b, qid=qid))
            K_used.append(K)
            trunc_texts.append(trunc)

        # Judge full + truncs in one batch for speed
        all_examples = [ex_full] + trunc_examples
        all_prefs = self._judge_batch(all_examples)
        pref_full = all_prefs[0] if all_prefs else None
        prefs_trunc = all_prefs[1:]

        # Compute agreement (only over K's that produced a non-None preference)
        if pref_full is None:
            agreement = 0.0
            matches = []
        else:
            matches = [int(p == pref_full) for p in prefs_trunc if p is not None]
            agreement = float(np.mean(matches)) if matches else 1.0

        return {
            "pref_full": pref_full,
            "prefs_trunc": prefs_trunc,
            "K_used": K_used,
            "agreement": agreement,
            # Use <= so threshold pinned at 0 (when some cleans also hit 0) still
            # flags attack samples at agreement=0. With strict <, a threshold of
            # 0 never flags anything (agreement is non-negative), losing all signal.
            "flag": bool(agreement <= self.threshold) if agreement < 1.0 else False,
            "trunc_texts": trunc_texts,
            "n_trunc_evaluated": len(matches) if pref_full is not None else 0,
        }

    def evaluate_sample_v2(self, instruction_clean: str, suffix: str,
                           response_a: str, response_b: str,
                           qid: str = "") -> Dict:
        """v2: Position-bias-corrected TCD via truncation-baseline correction.

        For each K in K_set, queries the judge on BOTH:
          - trunc(attacked_instruction, K)  -- the attacked prompt truncated
          - trunc(clean_instruction, K)     -- the clean prompt truncated
        and computes persistence_rate = mean over K of
        [judge(trunc_attacked_K) != judge(trunc_clean_K)].

        For a clean sample (suffix=""), trunc_attacked_K == trunc_clean_K by
        construction, so persistence_rate == 0 strictly (modulo vLLM
        nondeterminism). For an attacked sample where the suffix causally
        drives the preference at some K, persistence_rate > 0. Crucially,
        any position bias that affects trunc_attacked_K also affects
        trunc_clean_K identically (same length, same content modulo
        suffix), so it cancels out -- this is the v2 correction.

        Cost: 2 + 2*|K_set| judge calls per sample (vs 1 + |K_set| for v1).

        Returns dict with:
          pref_full_attacked : int or None -- judge on full attacked prompt
          pref_full_clean    : int or None -- judge on full clean prompt
          prefs_trunc_attacked : list[int|None]
          prefs_trunc_clean    : list[int|None]
          K_used              : list[int]
          persistence_rate    : float in [0,1]
          suffix_changed_full : bool -- pref_full_attacked != pref_full_clean
          flag                : bool -- persistence_rate > threshold
          n_pairs_evaluated   : int -- # K's contributing to persistence_rate
        """
        if suffix and suffix.strip():
            attacked = build_attacked_instruction(instruction_clean, suffix)
            is_clean = False
        else:
            attacked = instruction_clean
            is_clean = True

        ex_full_attacked = self._build_example(attacked, response_a, response_b, qid=qid)
        ex_full_clean = self._build_example(instruction_clean, response_a, response_b, qid=qid)

        trunc_examples_attacked: List = []
        trunc_examples_clean: List = []
        K_used: List[int] = []
        for K in self.K_set:
            trunc_attacked = self._truncate_text(attacked, K)
            trunc_clean = self._truncate_text(instruction_clean, K)
            if trunc_attacked is None or len(trunc_attacked.strip()) < 5:
                continue
            if trunc_clean is None or len(trunc_clean.strip()) < 5:
                continue
            trunc_examples_attacked.append(self._build_example(trunc_attacked, response_a, response_b, qid=qid))
            trunc_examples_clean.append(self._build_example(trunc_clean, response_a, response_b, qid=qid))
            K_used.append(K)

        # For clean samples (suffix empty), attacked == instruction_clean, so
        # trunc_attacked and trunc_clean are byte-identical strings. Calling
        # the judge twice on identical inputs exposes us to vLLM numerical
        # noise across batch positions, which would spuriously push up the
        # clean persistence_rate. Deduplicate: judge each unique input once
        # and reuse the result. (For attack samples, both sides are genuinely
        # different inputs, so this branch is a no-op.)
        if is_clean:
            unique_examples = [ex_full_attacked] + trunc_examples_attacked
            unique_prefs = self._judge_batch(unique_examples)
            pref_full_attacked = unique_prefs[0] if unique_prefs else None
            pref_full_clean = pref_full_attacked
            nK = len(K_used)
            prefs_trunc_attacked = unique_prefs[1 : 1 + nK]
            prefs_trunc_clean = list(prefs_trunc_attacked)
        else:
            all_examples = [ex_full_attacked, ex_full_clean] + trunc_examples_attacked + trunc_examples_clean
            all_prefs = self._judge_batch(all_examples)
            pref_full_attacked = all_prefs[0] if all_prefs else None
            pref_full_clean = all_prefs[1] if len(all_prefs) > 1 else None
            nK = len(K_used)
            prefs_trunc_attacked = all_prefs[2 : 2 + nK]
            prefs_trunc_clean = all_prefs[2 + nK : 2 + 2 * nK]

        matches = []
        for pa, pc in zip(prefs_trunc_attacked, prefs_trunc_clean):
            if pa is None or pc is None:
                continue
            matches.append(int(pa != pc))
        persistence_rate = float(np.mean(matches)) if matches else 0.0

        suffix_changed_full = (
            pref_full_attacked is not None
            and pref_full_clean is not None
            and pref_full_attacked != pref_full_clean
        )

        return {
            "pref_full_attacked": pref_full_attacked,
            "pref_full_clean": pref_full_clean,
            "prefs_trunc_attacked": prefs_trunc_attacked,
            "prefs_trunc_clean": prefs_trunc_clean,
            "K_used": K_used,
            "persistence_rate": persistence_rate,
            "suffix_changed_full": suffix_changed_full,
            "flag": bool(persistence_rate > self.threshold),
            "n_pairs_evaluated": len(matches),
        }

    # ------------------------------------------------------------------ #
    # Threshold calibration                                              #
    # ------------------------------------------------------------------ #

    def calibrate_threshold(self, clean_pairs: Sequence[CleanPair]) -> float:
        """
        Set self.threshold so FPR <= target_fpr on clean (no-suffix) pairwise
        queries. On clean data, no truncation should change the preference, so
        agreement should be ~1.0; we flag anything below the 1st-percentile.
        """
        agreements = []
        for cp in clean_pairs:
            try:
                r = self.evaluate_sample(
                    cp.instruction, suffix="", response_a=cp.response_a,
                    response_b=cp.response_b, qid=cp.question_id,
                )
                if r["n_trunc_evaluated"] > 0:
                    agreements.append(r["agreement"])
            except Exception:
                continue
        if not agreements:
            self.threshold = 0.5
            return self.threshold
        # Low agreement = anomalous. Threshold at target_fpr percentile.
        # agreements are in [0,1]; flag if agreement < threshold.
        # We want flag rate on clean <= target_fpr, so threshold = target_fpr percentile.
        self.threshold = float(np.percentile(agreements, self.target_fpr * 100.0))
        return self.threshold

    def calibrate_threshold_v2(self, clean_pairs: Sequence[CleanPair]) -> float:
        """Calibrate v2 threshold on clean pairs.

        Clean samples should have persistence_rate ≈ 0 (theoretically exactly
        0, modulo vLLM nondeterminism). We threshold at the
        (1-target_fpr)*100 percentile of clean persistence_rates so that
        empirical FPR <= target_fpr.
        """
        rates: List[float] = []
        for cp in clean_pairs:
            try:
                r = self.evaluate_sample_v2(
                    cp.instruction, suffix="", response_a=cp.response_a,
                    response_b=cp.response_b, qid=cp.question_id,
                )
                if r["n_pairs_evaluated"] > 0:
                    rates.append(r["persistence_rate"])
            except Exception:
                continue
        if not rates:
            self.threshold = 0.0
            return self.threshold
        # flag if persistence_rate > threshold.
        # Want FPR on clean <= target_fpr, so threshold = (1-target_fpr)*100 percentile.
        self.threshold = float(np.percentile(rates, 100.0 * (1.0 - self.target_fpr)))
        return self.threshold


# --------------------------------------------------------------------------- #
# Clean-pair loading                                                           #
# --------------------------------------------------------------------------- #

def load_clean_pairs_from_attack_json(path: str, cap: Optional[int] = None,
                                      seed: int = 0) -> List[CleanPair]:
    """Build clean pairwise queries from an attack results JSON.

    The attack JSON stores the original (instruction, A, B) per sample -- these
    are clean (no suffix). We use them as the negative class for FPR
    calibration. The same instructions are what the existing LR/MLP/similarity
    defenses use for FPR calibration, so this stays directly comparable.
    """
    with open(path) as f:
        d = json.load(f)
    records = d.get("records", []) if isinstance(d, dict) else d
    out: List[CleanPair] = []
    seen = set()
    for r in records:
        if not isinstance(r, dict):
            continue
        instr = r.get("instruction")
        a = r.get("response_a")
        b = r.get("response_b")
        if not instr or not a or not b:
            continue
        key = instr[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(CleanPair(
            instruction=str(instr),
            response_a=str(a),
            response_b=str(b),
            question_id=str(r.get("id", "")),
        ))
        if cap and len(out) >= cap:
            break
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def load_attack_pairs_from_json(path: str,
                                require_success: bool = False) -> List[dict]:
    """Load attack records as flat dicts for the runner."""
    with open(path) as f:
        d = json.load(f)
    records = d.get("records", []) if isinstance(d, dict) else d
    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if require_success and not r.get("success"):
            continue
        out.append(r)
    return out
