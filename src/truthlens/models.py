"""Model definitions for TruthLens.

* :class:`LinearHead`: standardise -> linear -> logit, stored as plain numpy arrays
  (``.npz``), so deployment needs neither scikit-learn nor pickle.
* :class:`CnnDetector`: a timm CNN with a single-logit head whose class-activation map
  is computed in the same forward pass. For a global-average-pool + linear head
  (EfficientNet, ResNet), Grad-CAM reduces exactly to CAM, and the spatial mean of the
  (pre-ReLU) map plus the bias *equals the logit*, so the heatmap is an additive
  decomposition of the decision over image regions (verified in the smoke test).
* :func:`cnn_input`: the single preprocessing path for the CNN at eval *and* inference
  (native resolution, centre crop to a pixel budget, never resized).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# --------------------------------------------------------------------------------------
# Linear heads (CLIP probe, spectral model, metadata baseline)
# --------------------------------------------------------------------------------------


@dataclass
class LinearHead:
    """``logit = ((x - mean) / scale) @ coef + intercept`` over concatenated feature blocks."""

    feature_keys: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    meta: dict = field(default_factory=dict)

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Raw (uncalibrated) logits for a ``[N, D]`` feature matrix."""
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        return Z @ self.coef + self.intercept

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef,
                 intercept=np.array(self.intercept),
                 header=np.array(json.dumps({"feature_keys": self.feature_keys, "meta": self.meta})))

    @classmethod
    def load(cls, path: str | Path) -> "LinearHead":
        z = np.load(path, allow_pickle=False)
        header = json.loads(str(z["header"]))
        return cls(header["feature_keys"], z["mean"], z["scale"], z["coef"], float(z["intercept"]),
                   header.get("meta", {}))

    @classmethod
    def from_sklearn(cls, scaler, lr, feature_keys: list[str], meta: dict | None = None) -> "LinearHead":
        """Convert a fitted ``StandardScaler`` + binary ``LogisticRegression``."""
        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        return cls(list(feature_keys), scaler.mean_.astype(np.float64), scale.astype(np.float64),
                   lr.coef_[0].astype(np.float64), float(lr.intercept_[0]), meta or {})


def stack_features(feats: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    """Concatenate named feature blocks (each ``[N, d_k]``) into ``[N, sum d_k]``."""
    return np.concatenate([np.asarray(feats[k], dtype=np.float32) for k in keys], axis=1)


# --------------------------------------------------------------------------------------
# CNN artifact detector
# --------------------------------------------------------------------------------------

#: architectures whose head is global-average-pool + linear (CAM == Grad-CAM exactly)
GAP_LINEAR_PREFIXES = ("efficientnet", "tf_efficientnet", "resnet", "resnext", "regnet", "mobilenetv3")


class CnnDetector(nn.Module):
    """timm backbone with one output logit (AI-generated) and an exact CAM."""

    def __init__(self, arch: str = "efficientnet_b0", pretrained: bool = False, drop_rate: float = 0.2) -> None:
        super().__init__()
        import timm

        if not arch.startswith(GAP_LINEAR_PREFIXES):
            raise ValueError(f"{arch}: only GAP+linear heads are supported (exact CAM)")
        self.arch = arch
        self.net = timm.create_model(arch, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)
        cfg = self.net.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1),
                             persistent=False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[0, 1]`` RGB tensors to the backbone's input normalisation."""
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, 3, H, W]`` in ``[0, 1]``. Returns ``[B]`` logits."""
        return self.net(self.normalize(x)).squeeze(1)

    def forward_with_cam(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits [B], cam [B, h, w])`` from one forward pass.

        ``cam[b, i, j] = w . A[b, :, i, j]``, so ``cam.mean((1, 2)) + bias == logits``.
        """
        feats = self.net.forward_features(self.normalize(x))
        logits = self.net.forward_head(feats).squeeze(1)
        clf = self.net.get_classifier()
        cam = torch.einsum("bchw,c->bhw", feats, clf.weight[0])
        return logits, cam

    @property
    def bias(self) -> float:
        b = self.net.get_classifier().bias
        return float(b[0].detach()) if b is not None else 0.0


def cnn_input(img: Image.Image, max_pixels: int = 1024 * 1024, min_side: int = 64
              ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Prepare an RGB image for :class:`CnnDetector` without resampling.

    Images above ``max_pixels`` are centre-cropped to that budget (square if possible).
    Tiny images (< ``min_side``) are upsampled. Returns ``([3, h, w] in [0, 1], box)`` where
    ``box = (x0, y0, x1, y1)`` locates the analysed region in the original image.
    """
    w, h = img.size
    if min(w, h) < min_side:  # tiny image: upsample; the analysed region is the whole image
        s = min_side / min(w, h)
        up = img.resize((max(min_side, round(w * s)), max(min_side, round(h * s))), Image.BICUBIC)
        arr = np.asarray(up, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), (0, 0, w, h)
    if w * h > max_pixels:
        side = int(max_pixels ** 0.5)
        if min(w, h) >= side:
            cw, ch = side, side
        elif w < h:
            cw, ch = w, min(h, max_pixels // w)
        else:
            cw, ch = min(w, max_pixels // h), h
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        box = (x0, y0, x0 + cw, y0 + ch)
    else:
        box = (0, 0, w, h)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), box


def save_cnn(model: CnnDetector, path: str | Path) -> None:
    """Save weights as safetensors (no pickle)."""
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={"arch": model.arch})


def load_cnn(path: str | Path, arch: str | None = None, device: str | torch.device = "cpu") -> CnnDetector:
    """Load a :class:`CnnDetector` saved by :func:`save_cnn`."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if arch is None:
        with safe_open(str(path), framework="pt") as f:
            arch = (f.metadata() or {}).get("arch", "efficientnet_b0")
    model = CnnDetector(arch, pretrained=False, drop_rate=0.0)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval().to(device)
