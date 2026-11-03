"""Evaluation metrics (training/eval only; not imported by the deployed app).

Conventions: ``y`` is 1 for AI-generated, ``p`` is the calibrated probability of AI.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_score,
                             roc_auc_score)

from .calibrate import expected_calibration_error


def _safe(fn, *a, **k) -> float:
    try:
        return float(fn(*a, **k))
    except ValueError:  # e.g. AUC with a single class present
        return float("nan")


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Threshold and ranking metrics for one split."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "n": int(len(y)), "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": _safe(precision_score, y, pred, zero_division=0),
        "recall_tpr": tpr, "specificity_tnr": tnr,
        "f1": _safe(f1_score, y, pred, zero_division=0),
        "roc_auc": _safe(roc_auc_score, y, p),
        "average_precision": _safe(average_precision_score, y, p),
        "ece": expected_calibration_error(p, y),
        "brier": _safe(brier_score_loss, y, p),
        "log_loss": _safe(log_loss, y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str = "roc_auc", threshold: float = 0.5,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI (stratified by class) for one metric of :func:`binary_metrics`."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) if len(pos) and len(neg) \
            else rng.choice(len(y), len(y))
        yy, pp = y[idx], p[idx]
        if metric == "roc_auc":
            v = _safe(roc_auc_score, yy, pp)
        elif metric == "accuracy":
            v = float(((pp >= threshold).astype(int) == yy).mean())
        elif metric == "average_precision":
            v = _safe(average_precision_score, yy, pp)
        else:
            v = binary_metrics(yy, pp, threshold)[metric]
        vals.append(v)
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def per_source_metrics(y: np.ndarray, p: np.ndarray, sources: np.ndarray, threshold: float = 0.5) -> dict:
    """Per-source breakdown within one split.

    For an AI source: detection rate (TPR) and ROC-AUC of that source's images against
    all real images of the same split. For a real source: specificity (TNR).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    sources = np.asarray(sources).astype(str)
    real = y == 0
    out: dict[str, dict] = {}
    for s in sorted(set(sources)):
        m = sources == s
        lab = int(round(y[m].mean()))
        row = {"n": int(m.sum()), "label": lab, "mean_prob_ai": float(p[m].mean())}
        if lab == 1:
            row["detection_rate"] = float((p[m] >= threshold).mean())
            sel = m | real
            row["roc_auc_vs_real"] = _safe(roc_auc_score, y[sel], p[sel])
            row["average_precision_vs_real"] = _safe(average_precision_score, y[sel], p[sel])
        else:
            row["specificity"] = float((p[m] < threshold).mean())
        out[s] = row
    return out
