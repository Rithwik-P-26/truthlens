"""Feature extraction for TruthLens.

Everything that turns pixels (or files) into numbers lives here:

* device / seed helpers shared by the whole package,
* robust image loading (EXIF orientation, alpha, odd modes),
* :class:`ClipEmbedder`: a frozen open_clip image tower that returns the final
  image embedding *and* the CLS token of selected intermediate blocks
  (UnivFD uses only the former; RINE-style probes use the latter too),
* hand-crafted spectral features (azimuthally averaged power spectra of the image
  and of its high-pass noise residual, periodic-peak score, high-frequency ratio),
* file-metadata features used only by the *shortcut-audit* baseline,
* deterministic perturbations used by the robustness evaluation.

Runtime dependencies are limited to numpy, PIL, torch and open_clip so the module
can be imported by the deployed app.
"""
from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

# --------------------------------------------------------------------------------------
# Device / reproducibility helpers
# --------------------------------------------------------------------------------------


def pick_device(prefer: str | None = None) -> torch.device:
    """Return ``prefer`` if given, else the best available device (cuda > mps > cpu)."""
    if prefer:
        return torch.device(prefer)
    env = os.environ.get("TRUTHLENS_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------------------


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL image to 8-bit RGB, compositing transparency over white."""
    if img.mode == "RGB":
        return img
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if img.mode.startswith("I;16") or img.mode == "I":
        arr = np.asarray(img, dtype=np.float32)
        hi = float(arr.max()) or 1.0
        img = Image.fromarray(np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8), mode="L")
    return img.convert("RGB")


def load_image(path: str | os.PathLike) -> Image.Image:
    """Load an image file as RGB with EXIF orientation applied."""
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im) or im
        return to_rgb(im)


def gray_float(img: Image.Image) -> np.ndarray:
    """ITU-R 601 luma of an image as float32 in [0, 1]."""
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


# --------------------------------------------------------------------------------------
# Dataset normalisation (must match scripts/prepare_data.py::normalize_one exactly)
# --------------------------------------------------------------------------------------

NORM_SIZE = 224  # every training image is 224x224
NORM_MIN_SIDE = 256  # training data never contains images whose shorter side was below this
NORM_JPEG_Q = 95  # JPEG quality, 4:4:4 chroma, optimize=True, no EXIF/ICC


def _flatten_like_dataset(im: Image.Image) -> Image.Image:
    """RGB conversion exactly as the data pipeline does it (alpha composited onto white)."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def _square_resize_jpeg(im: Image.Image) -> tuple[bytes, tuple[int, int, int, int]]:
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    box = (left, top, left + side, top + side)
    im = im.crop(box).resize((NORM_SIZE, NORM_SIZE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=NORM_JPEG_Q, subsampling=0, optimize=True)
    return buf.getvalue(), box


def normalize_image(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], bool]:
    """Apply the training-data normalisation to a decoded image.

    Centre square crop (side = shorter side), Lanczos resize to 224x224, JPEG q95 (4:4:4)
    round trip. Returns ``(normalised RGB image, crop box in input coordinates, too_small)``;
    ``too_small`` flags inputs whose shorter side is below 256 px, which the training data
    never contains. Inputs below 224 px are upscaled here, and the detector calls them Uncertain.
    No EXIF transpose: the caller's orientation is kept so heatmaps align with its display.
    """
    data, box = _square_resize_jpeg(_flatten_like_dataset(img))
    with Image.open(io.BytesIO(data)) as out:
        return out.convert("RGB"), box, (box[2] - box[0]) < NORM_MIN_SIDE


def normalize_bytes(raw: bytes) -> bytes:
    """Byte-level mirror of ``normalize_one`` (EXIF transpose when EXIF exists); for equivalence tests."""
    im = Image.open(io.BytesIO(raw))
    has_exif = bool(im.info.get("exif")) or bool(getattr(im, "getexif", lambda: {})())
    im = ImageOps.exif_transpose(im) if has_exif else im
    return _square_resize_jpeg(_flatten_like_dataset(im))[0]


# --------------------------------------------------------------------------------------
# CLIP embeddings (frozen backbone)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipSpec:
    """An open_clip image tower and the transformer blocks whose CLS token we keep."""

    arch: str
    pretrained: str
    blocks: tuple[int, ...]  # 0-based block indices for intermediate CLS tokens
    quick_gelu: bool = True  # OpenAI CLIP weights were trained with QuickGELU


CLIP_SPECS: dict[str, ClipSpec] = {
    # UnivFD (Ojha et al., CVPR 2023) backbone. 24 blocks, width 1024, embed 768.
    "clip_vitl14": ClipSpec("ViT-L-14", "openai", (7, 11, 15, 19, 23)),
    # 4.6x cheaper alternative. 12 blocks, width 768, embed 512.
    "clip_vitb16": ClipSpec("ViT-B-16", "openai", (3, 5, 7, 9, 11)),
    # Fastest option. 12 blocks, width 768, embed 512.
    "clip_vitb32": ClipSpec("ViT-B-32", "openai", (3, 5, 7, 9, 11)),
}


class ClipEmbedder:
    """Frozen open_clip image tower.

    ``embed`` returns a dict of float32 arrays::

        {"final": [B, E]  (projected image embedding, what UnivFD uses),
         "b07":   [B, W]  (CLS token after block 8), ...}

    The text tower is discarded right after loading to save memory.
    """

    def __init__(self, key: str = "clip_vitl14", device: str | torch.device | None = None) -> None:
        import open_clip  # local import keeps module import cheap

        if key not in CLIP_SPECS:
            raise KeyError(f"unknown CLIP backbone {key!r}; options: {sorted(CLIP_SPECS)}")
        self.key = key
        self.spec = CLIP_SPECS[key]
        self.device = pick_device(str(device) if device is not None else None)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.spec.arch, pretrained=self.spec.pretrained, force_quick_gelu=self.spec.quick_gelu
        )
        self.visual = model.visual.eval().to(self.device)
        for p in self.visual.parameters():
            p.requires_grad_(False)
        del model
        self.preprocess = preprocess  # Resize(224, bicubic) + CenterCrop(224) + Normalize
        self.feature_names = ["final"] + [f"b{b:02d}" for b in self.spec.blocks]

    @torch.inference_mode()
    def embed(self, batch: torch.Tensor, amp: bool = False) -> dict[str, np.ndarray]:
        """Embed a preprocessed batch ``[B, 3, 224, 224]``."""
        x = batch.to(self.device, non_blocking=True)
        ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if amp and self.device.type in ("cuda", "mps")
            else _nullcontext()
        )
        with ctx:
            out = self.visual.forward_intermediates(
                x,
                indices=list(self.spec.blocks),
                output_fmt="NLC",
                output_extra_tokens=True,
            )
        feats = {"final": out["image_features"].float().cpu().numpy()}
        for b, cls in zip(self.spec.blocks, out["image_intermediates_prefix"]):
            feats[f"b{b:02d}"] = cls[:, 0].float().cpu().numpy()
        return feats

    def embed_images(self, images: Sequence[Image.Image], amp: bool = False) -> dict[str, np.ndarray]:
        """Preprocess and embed a list of PIL images."""
        batch = torch.stack([self.preprocess(to_rgb(im)) for im in images])
        return self.embed(batch, amp=amp)


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


# --------------------------------------------------------------------------------------
# Spectral (frequency-domain) features
# --------------------------------------------------------------------------------------

SPECTRAL_BINS = 32
SPECTRAL_CROP = 256


def _box_blur3(a: np.ndarray) -> np.ndarray:
    """3x3 mean filter with edge replication (pure numpy)."""
    p = np.pad(a, 1, mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _crop_positions(h: int, w: int, s: int) -> list[tuple[int, int]]:
    """Centre crop plus the centres of the four quadrants (deterministic)."""
    cy, cx = (h - s) // 2, (w - s) // 2
    pos = [(cy, cx)]
    if h >= 2 * s and w >= 2 * s:
        for fy in (0.25, 0.75):
            for fx in (0.25, 0.75):
                y = int(np.clip(fy * h - s / 2, 0, h - s))
                x = int(np.clip(fx * w - s / 2, 0, w - s))
                pos.append((y, x))
    return pos


def _radial_index(h: int, w: int, nbins: int, rmax: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx**2 + fy**2)
    idx = np.floor(r / rmax * nbins).astype(np.int64)
    idx[r < 1.0 / min(h, w)] = -1  # drop DC
    idx[idx >= nbins] = -1  # drop corners beyond Nyquist circle
    return idx, r


def _radial_mean(vals: np.ndarray, idx: np.ndarray, nbins: int) -> np.ndarray:
    m = idx >= 0
    s = np.bincount(idx[m], weights=vals[m], minlength=nbins)
    c = np.bincount(idx[m], minlength=nbins)
    return (s / np.maximum(c, 1)).astype(np.float32)


@dataclass
class SpectralStats:
    """Named scalar statistics used for human-readable explanations."""

    hf_ratio: float  # fraction of AC power above 0.25 cycles/px (image spectrum)
    peak_score: float  # strength of isolated periodic peaks in the residual spectrum (log units)
    noise_level: float  # std of the 3x3 high-pass residual (0..1 intensity units)


def spectral_features(img: Image.Image, nbins: int = SPECTRAL_BINS, crop: int = SPECTRAL_CROP
                      ) -> tuple[np.ndarray, SpectralStats]:
    """Frequency-domain descriptor of an image at native resolution.

    Up to five ``crop x crop`` windows are analysed (no resampling unless the image is
    smaller than the window). Returns a ``(2*nbins + 3,)`` float32 vector:
    shape-normalised log radial power profile of the image, the same for its 3x3
    high-pass residual, then ``log10(hf_ratio)``, ``peak_score``, ``log10(noise_level)``.
    """
    g = gray_float(img)
    h, w = g.shape
    if min(h, w) < 64:  # tiny images: upsample so the FFT is meaningful
        scale = 64 / min(h, w)
        g = gray_float(img.resize((max(64, round(w * scale)), max(64, round(h * scale))), Image.BICUBIC))
        h, w = g.shape
    s = int(min(crop, h, w)) // 2 * 2
    idx, r = _radial_index(s, s, nbins)
    fine_bins = 64
    fidx, _ = _radial_index(s, s, fine_bins)
    fy = np.abs(np.fft.fftfreq(s))[:, None] * np.ones((1, s))
    fx = np.abs(np.fft.fftfreq(s))[None, :] * np.ones((s, 1))
    peak_mask = (r >= 0.05) & (r <= 0.5) & (fx >= 2.0 / s) & (fy >= 2.0 / s) & (fidx >= 0)
    win = _hann2d(s, s)
    hf_mask = (r > 0.25) & (r <= 0.5)
    ac_mask = (r > 0.02) & (r <= 0.5)

    prof_img, prof_res, hf, peaks, noise = [], [], [], [], []
    for (y, x) in _crop_positions(h, w, s):
        c = g[y:y + s, x:x + s]
        res = c - _box_blur3(c)
        p_img = np.abs(np.fft.fft2((c - c.mean()) * win)) ** 2
        p_res = np.abs(np.fft.fft2((res - res.mean()) * win)) ** 2
        lp_img = np.log(p_img + 1e-10)
        lp_res = np.log(p_res + 1e-10)
        a = _radial_mean(lp_img, idx, nbins)
        b = _radial_mean(lp_res, idx, nbins)
        prof_img.append(a - a.mean())
        prof_res.append(b - b.mean())
        hf.append(p_img[hf_mask].sum() / max(p_img[ac_mask].sum(), 1e-12))
        # periodic peaks: deviation of each frequency from the radial median of its ring
        med = np.zeros(fine_bins, dtype=np.float32)
        for k in range(fine_bins):
            ring = lp_res[fidx == k]
            med[k] = np.median(ring) if ring.size else 0.0
        dev = lp_res - med[np.clip(fidx, 0, fine_bins - 1)]
        top = np.sort(dev[peak_mask])[-8:]
        peaks.append(float(top.mean()) if top.size else 0.0)
        noise.append(float(res.std()))

    stats = SpectralStats(float(np.mean(hf)), float(np.mean(peaks)), float(np.mean(noise)))
    vec = np.concatenate([
        np.mean(prof_img, axis=0),
        np.mean(prof_res, axis=0),
        np.array([np.log10(stats.hf_ratio + 1e-8), stats.peak_score, np.log10(stats.noise_level + 1e-8)],
                 dtype=np.float32),
    ]).astype(np.float32)
    return vec, stats


def log_spectrum_display(img: Image.Image, max_side: int = 1024) -> np.ndarray:
    """Log-magnitude FFT of the grayscale image for display, float32 in [0, 1].

    Computed at native resolution; images larger than ``max_side`` on a side are
    centre-cropped (never resized, which would erase the high frequencies we want to show).
    A Hann window suppresses the edge-discontinuity cross. DC is at the centre.
    """
    g = gray_float(img)
    h, w = g.shape
    ch, cw = min(h, max_side), min(w, max_side)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    g = g[y0:y0 + ch, x0:x0 + cw]
    g = (g - g.mean()) * _hann2d(ch, cw)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    lo, hi = np.percentile(mag, [1.0, 99.9])
    return np.clip((mag - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# File-metadata features (shortcut-audit baseline only; never used by the detector)
# --------------------------------------------------------------------------------------

META_FEATURES = [
    "log_width", "log_height", "aspect", "log_pixels", "is_jpeg", "is_png", "is_webp",
    "bits_per_pixel", "jpeg_quality", "has_exif", "is_square", "side_multiple_64",
]

_STD_LUMA_Q50 = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def _jpeg_quality(im: Image.Image) -> float:
    """Estimate libjpeg quality (1..100) from the luminance quantisation table."""
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return 0.0
    table = np.array(q[0], dtype=np.float64)
    if table.size != 64:
        return 0.0
    scale = np.mean(table / _STD_LUMA_Q50) * 100.0  # libjpeg scale factor in percent
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(np.clip(quality, 1.0, 100.0))


def metadata_features(path: str | os.PathLike) -> np.ndarray:
    """Container/format statistics of an image file (see ``META_FEATURES``)."""
    nbytes = os.path.getsize(path)
    with Image.open(path) as im:
        w, h = im.size
        fmt = (im.format or "").upper()
        quality = _jpeg_quality(im) if fmt == "JPEG" else 0.0
        has_exif = float(bool(im.info.get("exif")))
    px = max(w * h, 1)
    return np.array([
        np.log(w), np.log(h), w / h, np.log(px), fmt == "JPEG", fmt == "PNG", fmt == "WEBP",
        8.0 * nbytes / px, quality, has_exif, float(w == h), float(w % 64 == 0 and h % 64 == 0),
    ], dtype=np.float32)


# --------------------------------------------------------------------------------------
# Perturbations for the robustness evaluation (deterministic)
# --------------------------------------------------------------------------------------

PERTURBATIONS: dict[str, tuple[str, float]] = {
    "clean": ("none", 0.0),
    "jpeg_q90": ("jpeg", 90),
    "jpeg_q75": ("jpeg", 75),
    "jpeg_q50": ("jpeg", 50),
    "downscale_0.5x": ("downscale", 0.5),
    "blur_sigma1": ("blur", 1.0),
}


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode/decode an image as JPEG at the given quality (4:2:0 subsampling, libjpeg default)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def apply_perturbation(img: Image.Image, name: str) -> Image.Image:
    """Apply one named perturbation from ``PERTURBATIONS``."""
    kind, level = PERTURBATIONS[name]
    if kind == "none":
        return img
    if kind == "jpeg":
        return jpeg_roundtrip(img, int(level))
    if kind == "downscale":
        w, h = img.size
        return img.resize((max(1, round(w * level)), max(1, round(h * level))), Image.BICUBIC, reducing_gap=None)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(level)))
    raise ValueError(f"unknown perturbation kind {kind!r}")


def resolve_path(path: str, root: str | os.PathLike) -> Path:
    """Resolve a manifest path (relative to the repo root) to an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p
