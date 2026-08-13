"""
MLP-based adversarial-suffix detector -- a standalone companion to the
logistic-regression detector in ``model_defense.py``.

This module is intentionally separate from the LR baseline so that the original
``ClassifierDefense`` and its runner (``run_stealth_defenses.py``) stay exactly
as they are. Both detectors share the *same* frozen MiniLM features and the
*same* FPR=1% threshold-calibration protocol, so any gap in detectability comes
purely from classifier capacity (linear vs. non-linear).

Architecture (sklearn MLPClassifier):
    384-dim MiniLM embedding
        -> StandardScaler
        -> Linear(384, 128) -> ReLU
        -> Linear(128, 64)  -> ReLU
        -> Linear(64, 1)    -> softmax over {clean, adversarial}

Regularisation aimed at the small training set (~1.3k samples):
    * L2 weight decay  alpha = 1e-3
    * early stopping on a 10% internal validation split, patience = 15 epochs
    * Adam, lr = 1e-3, max 300 epochs

The public API mirrors ClassifierDefense (fit / score / calibrate / flag /
cv_auc) so the two can be driven by identical experiment code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Public utilities reused from the shared defense toolbox (these are NOT the LR
# classifier -- just embedding / data-loading helpers).
from src.defense.model_defense import Embedder


@dataclass
class MLPClassifierDefense:
    """2-layer MLP detector over MiniLM sentence embeddings."""
    embedder: Embedder
    variant: str = "indist"      # "indist" | "transfer" (semantic only, see runner)
    target_fpr: float = 0.01
    hidden: tuple = (128, 64)
    alpha: float = 1e-3
    scaler: Optional[StandardScaler] = field(default=None, repr=False)
    clf: Optional[MLPClassifier] = field(default=None, repr=False)
    threshold: float = 0.5

    def _make_clf(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.hidden, activation="relu", solver="adam",
            alpha=self.alpha, batch_size=64, learning_rate_init=1e-3, max_iter=300,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
            random_state=0,
        )

    def _features(self, texts: Sequence[str]) -> np.ndarray:
        return self.scaler.transform(self.embedder.encode(texts)) if self.scaler \
            else self.embedder.encode(texts)

    def fit(self, pos_texts: Sequence[str], neg_texts: Sequence[str]) -> "MLPClassifierDefense":
        X = self.embedder.encode(list(pos_texts) + list(neg_texts))
        y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
        self.scaler = StandardScaler().fit(X)
        self.clf = self._make_clf().fit(self.scaler.transform(X), y)
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Probability of 'adversarial' class."""
        return self.clf.predict_proba(self._features(texts))[:, 1]

    def calibrate(self, clean_texts: Sequence[str]) -> "MLPClassifierDefense":
        """Threshold on clean data so FPR <= target_fpr (paper's protocol)."""
        s = self.score(clean_texts)
        self.threshold = float(np.percentile(s, (1.0 - self.target_fpr) * 100.0))
        return self

    def flag(self, texts: Sequence[str]) -> np.ndarray:
        return self.score(texts) > self.threshold

    def cv_auc(self, pos_texts: Sequence[str], neg_texts: Sequence[str],
               n_splits: int = 5) -> Dict[str, float]:
        """5-fold stratified-CV AUC + mean evasion rate at FPR=1%."""
        X = self.embedder.encode(list(pos_texts) + list(neg_texts))
        y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        aucs, fnrs = [], []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = self._make_clf().fit(sc.transform(X[tr]), y[tr])
            proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], proba))
            thr = np.percentile(proba[y[te] == 0], (1.0 - self.target_fpr) * 100.0)
            fnrs.append(float(np.mean(proba[y[te] == 1] <= thr)))
        return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "fnr@1%fpr_mean": float(np.mean(fnrs)), "n_splits": n_splits}
