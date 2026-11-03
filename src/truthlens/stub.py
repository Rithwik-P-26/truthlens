"""StubDetector: a stand-in for ``truthlens.inference.Detector`` (Contract 2).

Used by the app when trained weights are not available ("demo mode"). It has the
same constructor and ``predict`` signature as the real detector and returns the
same output dict. Values are fake but deterministic per image (seeded from the
pixels), so the same upload always gives the same answer. The frequency spectrum
is real: a log-magnitude FFT computed with numpy.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from PIL import Image

# Uncertainty band on prob_ai: inside [LOW, HIGH] the verdict is "Uncertain".
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65

MODEL_NAMES = ("CLIP probe", "CNN artifacts")
MODEL_WEIGHTS = (0.55, 0.45)


def log_spectrum(image: Image.Image) -> np.ndarray:
    """Centered log-magnitude FFT of the grayscale image, scaled to [0, 1] (HxW float32)."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gray = gray - gray.mean()
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    spec = np.log1p(mag)
    lo, hi = np.percentile(spec, [1.0, 99.9])
    spec = np.clip((spec - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return spec.astype(np.float32)


def _seed(image: Image.Image) -> int:
    thumb = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    return int.from_bytes(hashlib.sha1(thumb.tobytes()).digest()[:8], "little")


def _fake_heatmap(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    heat = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.integers(2, 4)):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        sy, sx = rng.uniform(0.08, 0.22) * h, rng.uniform(0.08, 0.22) * w
        amp = rng.uniform(0.5, 1.0)
        heat += amp * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2.0)
    heat -= heat.min()
    heat /= max(float(heat.max()), 1e-8)
    return heat


def _high_freq_ratio(spec: np.ndarray) -> float:
    h, w = spec.shape
    yy, xx = np.ogrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    outer = spec[r > 0.6].mean() if (r > 0.6).any() else 0.0
    inner = spec[r <= 0.6].mean() if (r <= 0.6).any() else 1.0
    return float(outer / max(inner, 1e-8))


class StubDetector:
    """Same interface as ``truthlens.inference.Detector``; returns simulated predictions."""

    is_stub = True
    version = "stub-0"

    def __init__(self, models_dir: str = "models", device: str | None = None):
        self.models_dir = models_dir
        self.device = device or "cpu"

    def predict(self, image: Image.Image) -> dict:
        t0 = time.perf_counter()
        image = image.convert("RGB")
        w, h = image.size
        rng = np.random.default_rng(_seed(image))

        # Push the base probability toward the extremes most of the time, with some
        # mass in the middle so the "Uncertain" state shows up too.
        base = float(rng.beta(0.6, 0.6))
        per_model = {}
        for name in MODEL_NAMES:
            per_model[name] = float(np.clip(base + rng.normal(0, 0.08), 0.01, 0.99))
        prob_ai = float(sum(wt * per_model[n] for wt, n in zip(MODEL_WEIGHTS, MODEL_NAMES)))

        if prob_ai >= UNCERTAIN_HIGH:
            verdict = "AI-generated"
        elif prob_ai <= UNCERTAIN_LOW:
            verdict = "Real"
        else:
            verdict = "Uncertain"
        confidence = float(max(prob_ai, 1.0 - prob_ai))

        side = min(w, h)  # like the real detector: spectrum of the central square at 224x224
        left, top = (w - side) // 2, (h - side) // 2
        view = image.crop((left, top, left + side, top + side)).resize((224, 224), Image.Resampling.LANCZOS)
        spectrum = log_spectrum(view)
        heatmap = _fake_heatmap(h, w, rng)

        clip_p, cnn_p = per_model[MODEL_NAMES[0]], per_model[MODEL_NAMES[1]]
        explanation = []
        if clip_p >= 0.5:
            explanation.append(
                f"Semantic probe (CLIP) finds the content statistically closer to generated images ({clip_p:.0%})."
            )
        else:
            explanation.append(
                f"Semantic probe (CLIP) finds the content consistent with camera photos ({1 - clip_p:.0%})."
            )
        if cnn_p >= 0.5:
            explanation.append("Artifact detector responds to fine texture typical of generator upsampling.")
        else:
            explanation.append("Artifact detector finds sensor-like noise and few generator artifacts.")
        hf = _high_freq_ratio(spectrum)
        explanation.append(
            "Frequency spectrum shows "
            + ("strong high-frequency energy." if hf > 0.75 else "a smooth natural fall-off toward high frequencies.")
        )
        if abs(clip_p - cnn_p) > 0.25 or verdict == "Uncertain":
            explanation.append("The two models disagree or sit near the decision boundary, so treat this result with caution.")

        return {
            "prob_ai": prob_ai,
            "verdict": verdict,
            "confidence": confidence,
            "models": per_model,
            "heatmap": heatmap,
            "spectrum": spectrum,
            "explanation": explanation[:4],
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
