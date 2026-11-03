"""Calibration, ensembling and the decision rule.

Everything here is fitted on the VAL split only and serialised to plain JSON
(``models/config.json``), so inference needs only numpy.

* :class:`Calibrator`: identity / temperature / Platt / isotonic maps from logits to probabilities.
* :class:`Ensemble`: combines member logits (calibrated-logit averaging or a stacked
  logistic regression) and applies a final calibrator.
* :func:`choose_uncertain_band`: selective-prediction rule. A verdict is issued only when
  the calibrated probability of the leaning class reaches ``min_confidence``; the value is
  chosen on VAL as the smallest one whose selective accuracy meets a target, with a floor.
* :func:`expected_calibration_error` / :func:`reliability_curve`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-7


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -50, 50)))


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _fit_affine_logistic(z: np.ndarray, y: np.ndarray, fit_bias: bool, l2: float = 1e-2,
                         iters: int = 100) -> tuple[float, float]:
    """Damped Newton (backtracking line search) for ``p = sigmoid(a*z + b)`` (Platt) or
    ``sigmoid(a*z)`` (temperature), minimising NLL + ``l2/2 * ||w||^2``. The line search
    guarantees monotone descent, so near-separable VAL sets cannot make it diverge."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.stack([z, np.ones_like(z)], 1) if fit_bias else z[:, None]
    w = np.zeros(X.shape[1])
    w[0] = 1.0

    def nll(v: np.ndarray) -> float:
        s = X @ v
        return float(np.sum(np.logaddexp(0.0, s) - y * s) + 0.5 * l2 * v @ v)

    f = nll(w)
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-10:
            f_new = nll(w - t * step)
            if f_new <= f - 1e-4 * t * float(g @ step):
                break
            t *= 0.5
        w = w - t * step
        converged = abs(f - f_new) < 1e-12 * max(1.0, abs(f)) or np.abs(t * step).max() < 1e-10
        f = f_new
        if converged:
            break
    return float(w[0]), float(w[1]) if fit_bias else 0.0


@dataclass
class Calibrator:
    """Maps raw logits to calibrated probabilities."""

    method: str = "identity"  # identity | temperature | platt | isotonic
    a: float = 1.0
    b: float = 0.0
    x: list[float] = field(default_factory=list)  # isotonic knots (logit space)
    y: list[float] = field(default_factory=list)  # isotonic values (probabilities)

    def __call__(self, z: np.ndarray | float) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if self.method == "isotonic":
            return np.interp(z, self.x, self.y)
        return sigmoid(self.a * z + self.b)

    def calibrated_logit(self, z: np.ndarray | float) -> np.ndarray:
        return logit(self(z))

    def to_dict(self) -> dict:
        d = {"method": self.method, "a": self.a, "b": self.b}
        if self.method == "isotonic":
            d.update(x=list(map(float, self.x)), y=list(map(float, self.y)))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(d.get("method", "identity"), d.get("a", 1.0), d.get("b", 0.0), d.get("x", []), d.get("y", []))


def fit_calibrator(z: np.ndarray, y: np.ndarray, method: str = "platt") -> Calibrator:
    """Fit a calibrator on (logits, labels) from the VAL split."""
    if method == "identity":
        return Calibrator()
    if method == "temperature":
        a, _ = _fit_affine_logistic(z, y, fit_bias=False)
        return Calibrator("temperature", a, 0.0)
    if method == "platt":
        a, b = _fit_affine_logistic(z, y, fit_bias=True)
        return Calibrator("platt", a, b)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(z, y)
        return Calibrator("isotonic", x=list(iso.X_thresholds_), y=list(iso.y_thresholds_))
    raise ValueError(method)


# --------------------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------------------


@dataclass
class Ensemble:
    """``prob = final( sum_k w_k * calibrated_logit_k + bias )``."""

    members: list[str]
    weights: dict[str, float]
    bias: float = 0.0
    member_calibrators: dict[str, Calibrator] = field(default_factory=dict)
    final: Calibrator = field(default_factory=Calibrator)
    method: str = "mean_logit"  # mean_logit | stacked_lr

    def member_probs(self, member_logits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: self.member_calibrators[k](member_logits[k]) for k in self.members}

    def combined_logit(self, member_logits: dict[str, np.ndarray]) -> np.ndarray:
        z = sum(self.weights[k] * self.member_calibrators[k].calibrated_logit(member_logits[k]) for k in self.members)
        return z + self.bias

    def prob(self, member_logits: dict[str, np.ndarray]) -> np.ndarray:
        return self.final(self.combined_logit(member_logits))

    def to_dict(self) -> dict:
        return {"method": self.method, "members": self.members, "weights": self.weights, "bias": self.bias,
                "member_calibrators": {k: c.to_dict() for k, c in self.member_calibrators.items()},
                "final_calibrator": self.final.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Ensemble":
        return cls(d["members"], d["weights"], d.get("bias", 0.0),
                   {k: Calibrator.from_dict(v) for k, v in d["member_calibrators"].items()},
                   Calibrator.from_dict(d.get("final_calibrator", {})), d.get("method", "mean_logit"))


def fit_ensemble(member_logits: dict[str, np.ndarray], y: np.ndarray, members: list[str],
                 method: str = "mean_logit", member_cal: str = "platt", final_cal: str = "platt") -> Ensemble:
    """Fit member calibrators, combination weights and the final calibrator on VAL.

    ``mean_logit`` (default) averages calibrated member logits with equal weights: weights
    learned on VAL (same generators as train) would over-trust whichever member overfits
    the seen generators. ``stacked_lr`` is kept as a comparison row.
    """
    cals = {k: fit_calibrator(member_logits[k], y, member_cal) for k in members}
    Z = np.stack([cals[k].calibrated_logit(member_logits[k]) for k in members], 1)
    if method == "mean_logit":
        weights = {k: 1.0 / len(members) for k in members}
        bias = 0.0
    elif method == "stacked_lr":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)
        weights = {k: float(w) for k, w in zip(members, lr.coef_[0])}
        bias = float(lr.intercept_[0])
    else:
        raise ValueError(method)
    ens = Ensemble(members, weights, bias, cals, Calibrator(), method)
    ens.final = fit_calibrator(ens.combined_logit(member_logits), y, final_cal)
    return ens


# --------------------------------------------------------------------------------------
# Decision rule / selective prediction
# --------------------------------------------------------------------------------------


def verdict_from_prob(p: float, threshold: float, min_confidence: float) -> tuple[str, float]:
    """Map a calibrated probability to ``(verdict, confidence)``.

    ``confidence`` is the calibrated probability of the leaning class, ``max(p, 1 - p)``
    (relative to ``threshold``); the verdict is "Uncertain" when it is below ``min_confidence``.
    """
    leaning_ai = p >= threshold
    conf = float(p if leaning_ai else 1.0 - p)
    if conf < min_confidence:
        return "Uncertain", conf
    return ("AI-generated" if leaning_ai else "Real"), conf


def selective_stats(p: np.ndarray, y: np.ndarray, threshold: float, min_confidence: float) -> dict:
    """Coverage and accuracy on the samples that receive a verdict."""
    p = np.asarray(p)
    y = np.asarray(y)
    conf = np.where(p >= threshold, p, 1 - p)
    keep = conf >= min_confidence
    pred = (p >= threshold).astype(int)
    acc = float((pred[keep] == y[keep]).mean()) if keep.any() else float("nan")
    return {"coverage": float(keep.mean()), "selective_accuracy": acc, "n_covered": int(keep.sum())}


def choose_uncertain_band(p: np.ndarray, y: np.ndarray, threshold: float = 0.5, target_accuracy: float = 0.95,
                          floor: float = 0.65, ceiling: float = 0.9) -> dict:
    """Smallest ``min_confidence`` in [floor, ceiling] with VAL selective accuracy >= target."""
    grid = np.round(np.arange(floor, ceiling + 1e-9, 0.01), 4)
    curve = [{"min_confidence": float(c), **selective_stats(p, y, threshold, c)} for c in grid]
    chosen = next((r for r in curve if r["selective_accuracy"] >= target_accuracy), curve[-1])
    return {"threshold": threshold, "min_confidence": chosen["min_confidence"],
            "uncertain_low": round(1 - chosen["min_confidence"], 4), "uncertain_high": chosen["min_confidence"],
            "target_accuracy": target_accuracy, "val_coverage": chosen["coverage"],
            "val_selective_accuracy": chosen["selective_accuracy"], "curve": curve}


# --------------------------------------------------------------------------------------
# Calibration metrics
# --------------------------------------------------------------------------------------


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Per-bin mean predicted probability, observed frequency of the AI class and counts."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins) / np.maximum(count, 1)
    freq = np.bincount(idx, weights=y, minlength=n_bins) / np.maximum(count, 1)
    return {"edges": edges.tolist(), "mean_pred": conf.tolist(), "frac_ai": freq.tolist(), "count": count.tolist()}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Binary ECE with equal-width bins on P(AI)."""
    rc = reliability_curve(p, y, n_bins)
    count = np.array(rc["count"], dtype=np.float64)
    gap = np.abs(np.array(rc["mean_pred"]) - np.array(rc["frac_ai"]))
    return float((count * gap).sum() / max(count.sum(), 1))
