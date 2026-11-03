"""Visual evidence and human-readable explanations.

Every sentence produced here is computed from measured quantities (calibrated member
probabilities, the CNN's class-activation map, spectral statistics compared with
reference percentiles from the TRAIN split). Nothing is templated without evidence.

Heatmap semantics: for the CNN (global-average-pool + linear head) the class-activation
map is an exact additive decomposition of the logit over image regions
(``logit = mean(cam) + bias``). We show its positive part ("regions pushing towards
AI-generated"), divided by a *fixed* scale calibrated on VAL rather than per-image
max-normalisation, so a real photo with no AI evidence renders as a dark map instead of
amplified noise.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_GRID_NAMES = [["upper-left", "top", "upper-right"],
               ["left", "centre", "right"],
               ["lower-left", "bottom", "lower-right"]]


def cam_to_heatmap(cam: np.ndarray, box: tuple[int, int, int, int], image_size: tuple[int, int],
                   scale: float) -> np.ndarray:
    """Positive CAM / ``scale``, clipped to [0, 1], resized to ``box`` and placed in an image-sized map.

    ``image_size`` is PIL ``(W, H)``; returns ``[H, W]`` float32.
    """
    W, H = image_size
    x0, y0, x1, y1 = box
    pos = np.clip(np.asarray(cam, dtype=np.float32) / max(scale, 1e-6), 0.0, 1.0)
    up = np.asarray(Image.fromarray(pos, mode="F").resize((x1 - x0, y1 - y0), Image.BILINEAR), dtype=np.float32)
    out = np.zeros((H, W), dtype=np.float32)
    out[y0:y1, x0:x1] = np.clip(up, 0.0, 1.0)
    return out


def heatmap_summary(heatmap: np.ndarray) -> dict:
    """Where and how concentrated the evidence is."""
    h, w = heatmap.shape
    total = float(heatmap.sum())
    peak = float(heatmap.max()) if heatmap.size else 0.0
    if total <= 0:
        return {"peak": 0.0, "concentration": 0.0, "region": None, "hot_fraction": 0.0}
    flat = np.sort(heatmap.ravel())[::-1]
    k = max(1, int(0.1 * flat.size))
    concentration = float(flat[:k].sum() / total)  # 0.1 = uniform, 1.0 = all heat in 10% of pixels
    cells = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cells[i, j] = heatmap[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3].mean()
    i, j = np.unravel_index(int(cells.argmax()), cells.shape)
    return {"peak": peak, "concentration": concentration, "region": _GRID_NAMES[i][j],
            "hot_fraction": float((heatmap > 0.5).mean())}


def _pct(p: float) -> str:
    if 0 < p < 0.005:
        return "<1%"
    if 0.995 < p < 1:
        return ">99%"
    return f"{100 * p:.0f}%"


def _percentile_rank(value: float, ref: dict) -> float | None:
    """Approximate percentile of ``value`` within a reference given as {"q": [...], "v": [...]}."""
    if not ref or "q" not in ref:
        return None
    return float(np.interp(value, ref["v"], ref["q"]))


def build_explanation(prob_ai: float, verdict: str, decision: dict, member_probs: dict[str, float],
                      heat: dict | None, analysed_box: tuple[int, int, int, int] | None,
                      image_size: tuple[int, int], spectral: dict | None, spectral_ref: dict | None,
                      notes: list[str] | None = None) -> list[str]:
    """Return 2-4 evidence-backed sentences (caution ``notes`` take priority over the spectral one)."""
    lo, hi = decision["uncertain_low"], decision["uncertain_high"]
    out: list[str] = []
    if verdict == "Uncertain" and lo < prob_ai < hi:
        out.append(f"Calibrated probability of AI generation is {_pct(prob_ai)}, inside the uncertainty band "
                   f"({_pct(lo)}-{_pct(hi)}), so TruthLens does not issue a verdict.")
    elif verdict == "Uncertain":  # withheld for an input-quality reason given in ``notes``
        out.append(f"Calibrated probability of AI generation is {_pct(prob_ai)}, but this input is outside the "
                   "conditions the model was validated on.")
    elif verdict == "AI-generated":
        out.append(f"Calibrated probability of AI generation is {_pct(prob_ai)} (verdicts need at least {_pct(hi)}).")
    else:
        out.append(f"Calibrated probability of AI generation is only {_pct(prob_ai)} "
                   f"(a 'Real' verdict needs at most {_pct(lo)}).")

    if len(member_probs) >= 2:
        parts = ", ".join(f"{k} {_pct(v)}" for k, v in member_probs.items())
        votes = [v >= 0.5 for v in member_probs.values()]
        spread = max(member_probs.values()) - min(member_probs.values())
        if all(votes) or not any(votes):
            side = "AI-generated" if all(votes) else "real"
            out.append(f"All detectors lean {side}: {parts}.")
        else:
            out.append(f"The detectors disagree ({parts}; spread {_pct(spread)}), which lowers confidence.")

    if heat is not None:
        W, H = image_size
        crop_note = ""
        if analysed_box is not None:
            x0, y0, x1, y1 = analysed_box
            if (x1 - x0) * (y1 - y0) < W * H:
                crop_note = f"; only the central {x1 - x0}x{y1 - y0} square is analysed"
        if heat["peak"] < 0.2:
            out.append(f"The artifact CNN found no strong localised generation artifacts{crop_note}.")
        elif heat["concentration"] >= 0.35:
            out.append(f"The artifact CNN's evidence is concentrated in the {heat['region']} of the image "
                       f"({_pct(heat['concentration'])} of the heat in 10% of the pixels){crop_note}.")
        else:
            out.append(f"The artifact CNN's evidence is spread across the image, strongest in the "
                       f"{heat['region']}{crop_note}.")

    for note in notes or []:
        if len(out) < 4:
            out.append(note)
    if spectral and spectral_ref and len(out) < 4:
        sentence = _spectral_sentence(spectral, spectral_ref)
        if sentence:
            out.append(sentence)
    return out[:4]


def _spectral_sentence(spectral: dict, ref: dict) -> str | None:
    """Compare spectral statistics with percentiles of REAL training photos."""
    real = ref.get("real", {})
    hf_rank = _percentile_rank(spectral["hf_ratio"], real.get("hf_ratio", {}))
    pk_rank = _percentile_rank(spectral["peak_score"], real.get("peak_score", {}))
    if pk_rank is not None and pk_rank >= 95:
        return (f"Periodic peaks in the noise spectrum are stronger than in {pk_rank:.0f}% of real training photos, "
                "a pattern left by upsampling layers (JPEG blocking can also cause it).")
    if hf_rank is not None and hf_rank <= 10:
        return (f"Fine-detail (high-frequency) energy is lower than in {100 - hf_rank:.0f}% of real training photos, "
                "consistent with generator smoothing (or heavy compression/blur).")
    if hf_rank is not None and hf_rank >= 90:
        return (f"High-frequency energy is higher than in {hf_rank:.0f}% of real training photos "
                "(sensor-like noise or sharpening).")
    if hf_rank is not None:
        return "Frequency-domain statistics fall within the typical range of real training photos."
    return None
