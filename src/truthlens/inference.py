"""TruthLens inference API (Contracts 2 and 3).

>>> from truthlens.inference import Detector
>>> det = Detector("models")            # reads models/config.json; det.device == "cpu" on a Space
>>> out = det.predict(pil_rgb_image)     # dict, see Detector.predict

One preprocessing policy end to end: every upload goes through :func:`normalize_image`, the
exact normalisation used to build the training/test data (centre square crop, Lanczos to
224x224, JPEG q95). All members analyse that 224 px view, so the reported test metrics describe
what the app computes. The cost is that high-frequency evidence in large photos is discarded.

Members (whichever ``models/config.json`` enables):

* ``clip``: frozen open_clip image tower + linear probe on final and intermediate CLS tokens,
* ``cnn``: fine-tuned timm CNN (source of the heatmap, an exact class-activation map),
* ``freq``: logistic regression on spectral features.

Member logits are calibrated individually and combined by :class:`truthlens.calibrate.Ensemble`
(all fitted on VAL). Runtime deps: torch, open_clip_torch, timm, safetensors, numpy, pillow.
The image orientation is used as given (no EXIF transpose) so the heatmap aligns with the caller's view.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .calibrate import Ensemble, verdict_from_prob
from .explain import build_explanation, cam_to_heatmap, heatmap_summary
from .features import NORM_SIZE, ClipEmbedder, log_spectrum_display, normalize_image, pick_device, spectral_features
from .models import LinearHead, cnn_input, load_cnn, stack_features


class Detector:
    """Calibrated ensemble detector. Stateless per call (safe to share across requests)."""

    def __init__(self, models_dir: str = "models", device: str | None = None, config: str = "config.json") -> None:
        """``config`` selects an alternative config in ``models_dir`` (e.g. ``config_fast.json``,
        the ViT-B/16 fallback for slow CPUs); the default is the Contract-2 behaviour."""
        self.models_dir = Path(models_dir)
        self.config = json.loads((self.models_dir / config).read_text())
        self._dev = pick_device(device)
        self.device: str = str(self._dev)  # Contract 3: shown by the app
        self.version: str = str(self.config.get("version", "unknown"))
        mcfg = self.config["members"]
        self.ensemble = Ensemble.from_dict(self.config["ensemble"])
        self.decision = self.config["decision"]
        self.display = {k: v.get("display_name", k) for k, v in mcfg.items()}
        self.spectral_ref = self.config.get("spectral_reference")
        active = set(self.ensemble.members)

        self.clip = self.clip_head = None
        if "clip" in active:
            self.clip = ClipEmbedder(mcfg["clip"]["backbone"], device=self._dev)
            self.clip_head = LinearHead.load(self.models_dir / mcfg["clip"]["head"])
        self.cnn = None
        self.cam_scale = 1.0
        if "cnn" in mcfg:  # loaded even if not in the ensemble: it provides the heatmap
            c = mcfg["cnn"]
            self.cnn = load_cnn(self.models_dir / c["weights"], c.get("arch"), self._dev)
            self.cam_scale = float(c.get("cam_scale", 1.0))
        self.freq_head = None
        if "freq" in active:
            self.freq_head = LinearHead.load(self.models_dir / mcfg["freq"]["head"])
        self._warmup()

    def _warmup(self) -> None:
        """One dummy pass so the first user request does not pay one-off allocation costs."""
        self.predict(Image.new("RGB", (256, 256), (127, 127, 127)))

    @torch.inference_mode()
    def member_logits(self, norm: Image.Image) -> tuple[dict[str, float], dict]:
        """Raw member logits for an already-normalised 224x224 image, plus evidence."""
        logits: dict[str, float] = {}
        evidence: dict = {}
        if self.clip is not None:
            feats = self.clip.embed_images([norm])
            logits["clip"] = float(self.clip_head.logits(stack_features(feats, self.clip_head.feature_keys))[0])
        if self.cnn is not None:
            x, _ = cnn_input(norm)
            lg, cam = self.cnn.forward_with_cam(x.unsqueeze(0).to(self._dev))
            logits["cnn"] = float(lg[0])
            evidence["cam"] = cam[0].float().cpu().numpy()
        vec, stats = spectral_features(norm)
        evidence["spectral"] = {"hf_ratio": stats.hf_ratio, "peak_score": stats.peak_score,
                                "noise_level": stats.noise_level}
        if self.freq_head is not None:
            logits["freq"] = float(self.freq_head.logits(vec[None, :])[0])
        return logits, evidence

    def predict(self, image: Image.Image) -> dict:
        """Classify one image.

        Returns ``prob_ai`` (calibrated), ``verdict`` ("AI-generated" | "Real" | "Uncertain"),
        ``confidence`` (calibrated probability of the leaning class, ``max(p, 1 - p)``),
        ``models`` (calibrated probability per ensemble member, keyed by display name),
        ``heatmap`` (``[H, W]`` float32 in [0, 1] at the ORIGINAL upload size: CNN evidence for
        "AI" on the analysed central square, 0 outside it; or None), ``spectrum`` (224x224
        log-magnitude FFT of the analysed image, float32 in [0, 1]), ``explanation``
        (2-4 evidence-based sentences) and ``latency_ms``.
        """
        t0 = time.perf_counter()
        norm, box, _ = normalize_image(image)
        logits, ev = self.member_logits(norm)
        z = {k: np.array([logits[k]]) for k in self.ensemble.members}
        prob = float(self.ensemble.prob(z)[0])
        member_probs = {self.display[k]: float(v[0]) for k, v in self.ensemble.member_probs(z).items()}
        verdict, conf = verdict_from_prob(prob, self.decision["threshold"], self.decision["min_confidence"])
        side = box[2] - box[0]
        upsampled = side < NORM_SIZE
        if upsampled and self.decision.get("uncertain_if_upsampled", True):
            verdict = "Uncertain"  # out-of-distribution input: see eval_robustness (downscale_0.5x)
        heatmap = heat = None
        if "cam" in ev:
            heatmap = cam_to_heatmap(ev["cam"], box, image.size, self.cam_scale)
            heat = heatmap_summary(heatmap[box[1]:box[3], box[0]:box[2]])
        notes = []
        if upsampled:
            notes.append(f"The image is only {side} px on its shorter side, so it was upsampled to {NORM_SIZE} px. "
                         "Training images were never upsampled, and in our robustness tests upsampled real photos "
                         "were often mistaken for AI-generated ones, so no verdict is given.")
        explanation = build_explanation(prob, verdict, self.decision, member_probs, heat, box, image.size,
                                        ev.get("spectral"), self.spectral_ref, notes)
        return {
            "prob_ai": prob,
            "verdict": verdict,
            "confidence": conf,
            "models": member_probs,
            "heatmap": heatmap,
            "spectrum": log_spectrum_display(norm),
            "explanation": explanation,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
