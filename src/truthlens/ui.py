"""Presentation helpers for the TruthLens Gradio app (HTML fragments, colormaps, metrics rendering).

Everything here is pure Python/numpy so it can be tested without launching Gradio.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_BAND = (0.35, 0.65)
MIN_SIDE = 224  # the detector's input size; smaller uploads are upsampled and get "Uncertain"

# --------------------------------------------------------------------------- images


def load_image(path: str, max_side: int = 4096) -> Image.Image:
    """Open an upload as RGB like the data pipeline: EXIF rotation, alpha onto white, cap huge images."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode.startswith("I;16") or im.mode in ("I", "F"):
        # 16-bit and float images: convert("RGB") would clip them to white, so rescale to 8 bit first
        arr = np.asarray(im, dtype=np.float32)
        mx = float(arr.max()) if arr.size else 0.0
        if im.mode == "F" or mx > 65535:
            hi = mx
        else:
            hi = 255.0 if mx <= 255 else 65535.0
        im = Image.fromarray(np.clip(arr / (hi or 1.0) * 255.0, 0, 255).astype(np.uint8))
    if im.mode != "RGB":
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def downscale(im: Image.Image, max_side: int) -> Image.Image:
    if max(im.size) <= max_side:
        return im
    im = im.copy()
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def analysed_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The central square the detector analyses (same rule as ``normalize_image``)."""
    w, h = size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return left, top, left + side, top + side


def resize_map(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a float map in [0,1] to (w, h)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if (arr.shape[1], arr.shape[0]) == size:
        return arr
    im = Image.fromarray(arr, mode="F").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _lut(anchors: list[tuple[int, int, int]]) -> np.ndarray:
    a = np.asarray(anchors, dtype=np.float32)
    xs = np.linspace(0, 1, len(a))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, xs, a[:, c]) for c in range(3)], axis=1).astype(np.uint8)


TURBO = _lut([(48, 18, 59), (70, 107, 227), (40, 187, 236), (49, 242, 153), (162, 252, 60),
              (237, 208, 58), (251, 128, 34), (208, 47, 6), (122, 4, 3)])
VIRIDIS = _lut([(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
                (31, 158, 137), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)])


def colorize(arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.clip(np.nan_to_num(arr) * 255.0, 0, 255).astype(np.uint8)
    return lut[idx]


def overlay(img: np.ndarray, heat: np.ndarray | None, opacity: float,
            box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Blend a [0,1] heatmap over an RGB uint8 image; cold regions stay mostly clear.

    ``box`` (display coordinates) marks the analysed square: the area outside it is dimmed.
    """
    out = img.astype(np.float32)
    if heat is not None:
        heat = np.clip(np.nan_to_num(np.asarray(heat, dtype=np.float32)), 0, 1)
        color = colorize(heat, TURBO).astype(np.float32)
        alpha = (float(opacity) * np.clip(heat * 1.25, 0, 1) ** 0.8)[..., None]
        out = out * (1 - alpha) + color * alpha
    if box is not None:
        h, w = out.shape[:2]
        x0, y0, x1, y1 = box
        if (x1 - x0) < w or (y1 - y0) < h:
            mask = np.ones((h, w), dtype=bool)
            mask[y0:y1, x0:x1] = False
            out[mask] *= 0.4
            t = max(1, round(min(w, h) / 200))
            edge = np.zeros((h, w), dtype=bool)
            edge[y0:y1, x0:min(x0 + t, w)] = edge[y0:y1, max(x1 - t, 0):x1] = True
            edge[y0:min(y0 + t, h), x0:x1] = edge[max(y1 - t, 0):y1, x0:x1] = True
            out[edge] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def spectrum_image(spec: np.ndarray | None, display_side: int = 448) -> np.ndarray | None:
    """Colourise the [0,1] spectrum and scale it to a comfortable display size."""
    if spec is None:
        return None
    spec = np.asarray(spec, dtype=np.float32)
    h, w = spec.shape[:2]
    scale = display_side / max(h, w)
    if abs(scale - 1.0) > 0.01:
        spec = resize_map(spec, (max(1, round(w * scale)), max(1, round(h * scale))))
    return colorize(spec, VIRIDIS)


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- names

_BRANDS = {"gpt": "GPT", "flux": "FLUX", "sd": "SD", "sdxl": "SDXL", "dalle": "DALL·E", "adm": "ADM", "glide": "GLIDE",
           "vqdm": "VQDM", "docci": "DOCCI", "laion": "LAION", "imagenet": "ImageNet", "pexels": "Pexels",
           "coco": "COCO", "xl": "XL", "ilv140": "ILv140", "t2v": "T2V", "vae": "VAE"}
_SPECIAL = {"sd15": "Stable Diffusion 1.5", "sd_1_5": "Stable Diffusion 1.5", "midjourney_v5": "Midjourney v5",
            "dalle_3": "DALL·E 3", "wukong": "Wukong", "gpt_image_1_5": "GPT Image 1.5"}


def pretty_source(s: str) -> str:
    """Human-readable generator / dataset name, e.g. ``seedream_v4_5`` -> ``Seedream v4.5``."""
    s = str(s or "").strip()
    if not s:
        return ""
    if s.lower() in _SPECIAL:
        return _SPECIAL[s.lower()]
    words: list[str] = []
    for tok in re.split(r"[_\-\s]+", s):
        if not tok:
            continue
        if tok.isdigit() and words and (words[-1][-1:].isdigit() or words[-1] == "FLUX"):
            words[-1] += "." + tok
            continue
        low = tok.lower()
        if low in _BRANDS:
            words.append(_BRANDS[low])
        elif re.fullmatch(r"v\d+", low):
            words.append(low)
        elif re.fullmatch(r"\d+b", low):
            words.append(low.upper())
        else:
            words.append(tok[:1].upper() + tok[1:])
    return " ".join(words)


# --------------------------------------------------------------------------- examples


def _norm_label(v) -> str | None:
    s = str(v).strip().lower()
    if s in {"1", "fake", "ai", "ai-generated", "ai_generated", "synthetic", "generated"}:
        return "AI-generated"
    if s in {"0", "real", "authentic", "photo"}:
        return "Real"
    return None


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in {"1", "yes", "true", "y", "hard"}


def _sidecar(examples_dir: Path) -> list[dict]:
    """Optional metadata rows from examples/*.csv or *.json (file + label/source/split/hard_case), in file order."""
    rows: list[dict] = []
    for p in sorted(examples_dir.glob("*.csv")):
        try:
            with open(p, newline="", encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
        except Exception:
            pass
    for p in sorted(examples_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows += [r for r in data if isinstance(r, dict)]
            elif isinstance(data, dict):
                for k, v in data.items():
                    rows.append({"file": k, **v} if isinstance(v, dict) else {"file": k, "label": v})
        except Exception:
            pass
    out = []
    for r in rows:
        r = {str(k).lower(): v for k, v in r.items()}
        key = r.get("file") or r.get("filename") or r.get("path") or r.get("name")
        if key:
            r["_name"] = Path(str(key)).name
            out.append(r)
    return out


def discover_examples(examples_dir: str | Path, limit: int = 16) -> list[dict]:
    """Return ``[{path, caption, truth, source, split, hard, sha1}]`` for images in examples/.

    Order follows the sidecar (e.g. ``labels.csv``) when present, then the remaining files by name.
    """
    d = Path(examples_dir)
    if not d.is_dir():
        return []
    files = {p.name: p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS
             and not any(part.startswith(".") for part in p.relative_to(d).parts)}
    meta = {}
    order = []
    for r in _sidecar(d):
        if r["_name"] in files and r["_name"] not in meta:
            meta[r["_name"]] = r
            order.append(r["_name"])
    order += [n for n in files if n not in meta]
    out = []
    for name in order[:limit]:
        p = files[name]
        m = meta.get(name, {})
        truth = _norm_label(m.get("label", "")) if m else None
        if truth is None:
            tokens = {t.lower() for part in p.relative_to(d).parts for t in part.replace("-", "_").split("_")}
            if tokens & {"fake", "ai", "synthetic", "generated"}:
                truth = "AI-generated"
            elif "real" in tokens:
                truth = "Real"
        source = str(m.get("source") or m.get("generator") or "").strip()
        hard = _truthy(m.get("hard_case") or m.get("hard")) or name.startswith("hard_")
        caption = str(m.get("caption") or m.get("title") or "").strip()
        if not caption:
            # thumbnails are small: label only; the generator is revealed in the result card
            caption = {"AI-generated": "AI", "Real": "Real"}.get(truth or "", "") or pretty_source(p.stem)
        try:
            sha = file_sha1(str(p))
        except OSError:
            continue
        out.append({"path": str(p), "caption": caption, "truth": truth, "source": source,
                    "split": str(m.get("split") or "").strip(), "hard": hard, "sha1": sha})
    return out


# --------------------------------------------------------------------------- result card

VERDICTS = {
    "AI-generated": ("ai", "Likely AI-generated",
                     '<path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'),
    "Real": ("real", "Likely a real photo",
             '<circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/>'),
    "Uncertain": ("unsure", "Uncertain",
                  '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3m.1 4h.01"/>'),
}


def _icon(paths: str, size: int = 28) -> str:
    return (f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')


def _pt(p: float, r: float = 80.0, cx: float = 100.0, cy: float = 100.0) -> tuple[float, float]:
    a = math.pi * (1 - p)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def _arc(p1: float, p2: float, cls: str) -> str:
    x1, y1 = _pt(p1)
    x2, y2 = _pt(p2)
    return f'<path class="{cls}" d="M{x1:.2f} {y1:.2f} A80 80 0 0 1 {x2:.2f} {y2:.2f}"/>'


def gauge_svg(prob: float, band: tuple[float, float], muted: bool = False) -> str:
    lo, hi = band
    p = min(max(prob, 0.0), 1.0)
    mx, my = _pt(p)
    ix, iy = _pt(p, r=62)
    return (
        f'<svg class="tl-gauge{" muted" if muted else ""}" viewBox="0 0 200 122" role="img" '
        f'aria-label="Calibrated probability of AI generation: {p:.0%}">'
        + _arc(0.0, lo, "seg seg-real") + _arc(lo, hi, "seg seg-unsure") + _arc(hi, 1.0, "seg seg-ai")
        + f'<line class="needle" x1="{ix:.2f}" y1="{iy:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
        f'<circle class="marker" cx="{mx:.2f}" cy="{my:.2f}" r="9"/>'
        f'<text class="gv" x="100" y="92" text-anchor="middle">{p:.0%}</text>'
        '<text class="gs" x="100" y="110" text-anchor="middle">P(AI)</text>'
        '<text class="gl" x="20" y="118" text-anchor="middle">Real</text>'
        '<text class="gl" x="180" y="118" text-anchor="middle">AI</text>'
        "</svg>"
    )


def _zone(p: float, band: tuple[float, float]) -> str:
    return "ai" if p >= band[1] else "real" if p <= band[0] else "unsure"


def empty_card() -> str:
    return (
        '<div class="tl-card tl-empty">'
        + _icon('<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>', 40)
        + "<h3>No image analyzed yet</h3>"
        "<p>Upload, paste or drop an image, or pick one of the examples, to get a verdict with visual evidence.</p>"
        "</div>"
    )


def error_card(msg: str) -> str:
    return f'<div class="tl-card tl-empty tl-error" role="alert"><h3>Could not analyze this file</h3><p>{html.escape(msg)}</p></div>'


def _example_html(ex: dict, verdict: str, gen_stats: dict | None) -> str:
    truth = ex.get("truth")
    if not truth:
        return ""
    ok = truth == verdict
    cls = "match" if ok else ("neutral" if verdict == "Uncertain" else "miss")
    note = "prediction matches" if ok else ("TruthLens abstained" if verdict == "Uncertain" else "prediction is wrong")
    src, split = ex.get("source", ""), ex.get("split", "")
    bits = []
    if src:
        bits.append(pretty_source(src))
    if truth == "AI-generated" and split == "test_unseen":
        bits.append("a generator never seen in training")
    elif truth == "AI-generated" and split == "test":
        bits.append("a generator family seen in training; held-out image")
    elif split:
        bits.append("held-out photo")
    detail = f" ({html.escape(', '.join(bits))})" if bits else ""
    out = (f'<div class="tl-truth {cls}"><span class="tl-tag">Example</span> Ground truth: <b>{html.escape(truth)}</b>'
           f"{detail} · {note}</div>")
    if ex.get("hard"):
        stat = (gen_stats or {}).get((split, src))
        if stat and stat.get("n"):
            acc, n = float(stat.get("accuracy", 0)), int(stat["n"])
            what = "detects" if truth == "AI-generated" else "correctly passes"
            what_n = "images from this generator" if truth == "AI-generated" else f"{pretty_source(src)} photos"
            rate = f"On {n:,} held-out {what_n}, TruthLens {what} {acc:.0%}."
        else:
            rate = ""
        out += ('<div class="tl-hard"><b>Hard case.</b> Shown on purpose: this is where the detector struggles. '
                f"{html.escape(rate)}</div>")
    return out


def result_card(res: dict, band: tuple[float, float], *, example: dict | None = None,
                gen_stats: dict | None = None, image_size: tuple[int, int] | None = None, mode: str = "",
                heatmap_available: bool = True) -> str:
    verdict = str(res.get("verdict", "Uncertain"))
    key, title, icon = VERDICTS.get(verdict, VERDICTS["Uncertain"])
    prob = float(res.get("prob_ai", 0.5))
    conf = float(res.get("confidence", max(prob, 1 - prob)))
    lo, hi = band
    short_side = min(image_size) if image_size else None
    too_small = verdict == "Uncertain" and short_side is not None and short_side < MIN_SIDE
    if too_small:
        sub = "No verdict · image too small to judge"
    elif verdict == "Uncertain":
        sub = f"No verdict · {prob:.0%} is inside the {lo:.0%}–{hi:.0%} band"
    else:
        conf_word = "High" if conf >= 0.85 else "Moderate" if conf >= 0.65 else "Low"
        sub = f"{conf_word} confidence · {conf:.0%}"

    callout = ""
    if too_small:
        callout = (f'<div class="tl-callout" role="note"><b>Image too small to judge.</b> Its shorter side is only '
                   f"{short_side} px; TruthLens analyses a {MIN_SIDE}×{MIN_SIDE} px view, so this image had to be "
                   "upscaled. Upscaled photos often look AI-generated to detectors, so no verdict is given. "
                   "Try the original, full-resolution file.</div>")

    models = res.get("models") or {}
    rows = []
    for name, p in models.items():
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        z = _zone(p, band)
        rows.append(
            f'<div class="tl-model"><span class="mname">{html.escape(str(name))}</span>'
            f'<span class="mbar" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{p * 100:.0f}" '
            f'aria-label="{html.escape(str(name))}: {p:.0%} AI"><span class="fill z-{z}" style="width:{p * 100:.1f}%"></span>'
            f'<span class="mid" style="left:50%"></span></span>'
            f'<span class="mval">{p:.0%}</span></div>'
        )
    models_html = (
        '<div class="tl-section"><div class="tl-h">Per-model probability of AI generation</div>'
        + "".join(rows) + "</div>"
    ) if rows else ""

    bullets = [str(b) for b in (res.get("explanation") or []) if str(b).strip()]
    if too_small:  # the callout already says this, in fewer words
        bullets = [b for b in bullets if not b.startswith("The image is only")]
    bullets = bullets[:4]
    if not heatmap_available:
        bullets.append("Heatmap is not available for this prediction.")
    bullets_html = (
        '<div class="tl-section"><div class="tl-h">Why</div><ul class="tl-why">'
        + "".join(f"<li>{html.escape(b)}</li>" for b in bullets) + "</ul></div>"
    ) if bullets else ""

    meta = [f"Analyzed in {float(res.get('latency_ms', 0.0)):.0f} ms"]
    if image_size:
        meta.append(f"{image_size[0]}×{image_size[1]} px")
        side = min(image_size)
        if image_size[0] != image_size[1]:
            meta.append(f"central {side}×{side} square analysed")
    if mode:
        meta.append(html.escape(mode))

    return (
        f'<div class="tl-card verdict-{key}">'
        '<div class="tl-top">'
        f'<div class="tl-badge" role="status" aria-live="polite">'
        f'<span class="tl-badge-icon">{_icon(icon)}</span>'
        f'<div><div class="tl-badge-title">{html.escape(title)}</div>'
        f'<div class="tl-badge-sub">{html.escape(sub)}</div></div></div>'
        f'<div class="tl-gauge-wrap">{gauge_svg(prob, band, muted=too_small)}'
        '<div class="tl-gauge-cap">Calibrated probability of AI generation</div></div>'
        "</div>"
        + callout + (_example_html(example, verdict, gen_stats) if example else "") + models_html + bullets_html
        + f'<div class="tl-meta">{" · ".join(meta)}</div>'
        "</div>"
    )


# --------------------------------------------------------------------------- how it works


def pipeline_html(config: dict | None) -> str:
    cfg = config or {}
    bb = cfg.get("backbones") if isinstance(cfg.get("backbones"), dict) else {}
    th = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), dict) else {}
    lo, hi = th.get("uncertain_low", DEFAULT_BAND[0]), th.get("uncertain_high", DEFAULT_BAND[1])
    clip_d = str(bb.get("clip") or "a frozen CLIP vision encoder")
    cnn_d = str(bb.get("cnn") or "a fine-tuned convolutional network")
    steps = [
        ("1", "Normalize", "Centre square crop, resized to 224×224 and re-encoded exactly like the training data, "
         "so test metrics describe what the app does."),
        ("2a", "CLIP semantic probe", f"A linear probe on {clip_d.split(';')[0]} features from several layers. "
         "Captures what generated content looks like and generalizes across generators."),
        ("2b", "CNN artifact detector", f"{cnn_d[:1].upper() + cnn_d[1:]}. Picks up low-level texture and "
         "upsampling traces, and provides the heatmap."),
        ("3", "Calibration", "Each member is calibrated on held-out data (Platt scaling), so 80% means about 80% "
         "of such images are AI-generated."),
        ("4", "Ensemble + abstain", f"Calibrated scores are combined. Between {lo:.0%} and {hi:.0%} the app says "
         "'Uncertain' instead of guessing; tiny images always get 'Uncertain'."),
        ("5", "Evidence", "A class-activation heatmap shows where the CNN saw generation artifacts; the FFT "
         "spectrum shows the frequency fingerprint the models analysed."),
    ]
    cards = "".join(
        f'<div class="tl-step"><div class="tl-step-n">{n}</div><div class="tl-step-t">{html.escape(t)}</div>'
        f'<div class="tl-step-d">{html.escape(d)}</div></div>'
        for n, t, d in steps
    )
    ver = cfg.get("version")
    extra = f'<div class="tl-meta"><b>Model version:</b> {html.escape(str(ver))}</div>' if ver else ""
    return f'<div class="tl-pipeline">{cards}</div>{extra}'


# ----- metrics: dedicated layout for the model agent's schema (v2), generic fallback otherwise

_PERTURB = {"clean": "No edit (clean)", "jpeg_q90": "JPEG quality 90", "jpeg_q75": "JPEG quality 75",
            "jpeg_q50": "JPEG quality 50", "downscale_0.5x": "Downscaled to 50%", "blur_sigma1": "Gaussian blur (σ=1)"}
_KIND_ORDER = {"deployed": 0, "member": 1, "baseline": 2, "ablation": 3}


def _pct(v, digits: int = 1) -> str:
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "–"


def _f3(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "–"


def _bar(v: float, cls: str) -> str:
    v = max(0.0, min(1.0, float(v)))
    return f'<span class="tl-mini"><span class="fill {cls}" style="width:{v * 100:.0f}%"></span></span>'


def generator_stats(metrics: dict | None) -> dict:
    """``{(split, source): row}`` from ``per_generator`` (used for the hard-case notes)."""
    out = {}
    for r in (metrics or {}).get("per_generator") or []:
        if isinstance(r, dict) and r.get("source"):
            out[(str(r.get("split", "")), str(r["source"]))] = r
    return out


def read_metrics(path: str | Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data else None
    except Exception:
        return None


def _v2_html(m: dict) -> str:
    sp = m.get("splits") or {}
    seen, unseen = sp.get("test") or {}, sp.get("test_unseen") or {}
    gens = [r for r in m.get("per_generator") or [] if isinstance(r, dict)]
    unseen_ai = [r for r in gens if r.get("split") == "test_unseen" and str(r.get("label")).lower() in ("ai", "1")]
    unseen_real = [r for r in gens if r.get("split") == "test_unseen" and str(r.get("label")).lower() in ("real", "0")]
    n_gen = len(unseen_ai)
    lat = m.get("latency_ms_cpu") or (m.get("latency") or {}).get("latency_ms_cpu")
    head = m.get("headline_split", "test_unseen")
    hs = sp.get(head) or {}
    get = lambda k: m.get(k, hs.get(k))  # noqa: E731

    parts = []
    # 1) takeaways
    take = []
    if unseen:
        take.append(f"On <b>{n_gen or 'many'} generators never seen in training</b>, TruthLens ranks an AI image above a "
                    f"real one {_pct(unseen.get('roc_auc'))} of the time (ROC-AUC {_f3(unseen.get('roc_auc'))}), "
                    f"wrongly flags only <b>{_pct(unseen.get('false_positive_rate'))}</b> of real photos and catches "
                    f"<b>{_pct(unseen.get('detection_rate'), 0)}</b> of fakes.")
    if seen:
        take.append(f"On held-out images from generator families it was trained on: <b>{_pct(seen.get('accuracy'))}</b> "
                    f"accuracy, ROC-AUC {_f3(seen.get('roc_auc'))}.")
    if unseen.get("coverage") is not None:
        take.append(f"It says <i>Uncertain</i> on {_pct(1 - float(unseen['coverage']))} of unseen-generator images "
                    "instead of guessing. Missing a new generator is its main weakness; see the per-generator list below.")
    if take:
        parts.append('<div class="tl-takeaways"><div class="tl-h">In 20 seconds</div><ul>'
                     + "".join(f"<li>{t}</li>" for t in take) + "</ul></div>")

    # 2) headline cards (headline split)
    label = (f"On {n_gen} generators never seen in training ({int(hs.get('n', 0)):,} test images)"
             if head == "test_unseen" else f"Headline split: {html.escape(str(head))}")
    cards = [("ROC-AUC", _f3(get("roc_auc")), "ranking quality, 1.0 is perfect"),
             ("Accuracy", _pct(get("accuracy")), "balanced real / AI"),
             ("F1", _f3(get("f1")), "AI class"),
             ("Fakes caught", _pct(get("detection_rate")), "detection rate at the default threshold"),
             ("Real photos flagged", _pct(get("false_positive_rate")), "false-positive rate"),
             ("CPU latency", f"{float(lat):.0f} ms" if lat else "–", "per image, median")]
    parts.append(f'<div class="tl-h tl-h-lg">{html.escape(label)}</div><div class="tl-stats">' + "".join(
        f'<div class="tl-stat"><div class="v">{v}</div><div class="k">{k}</div><div class="d">{d}</div></div>'
        for k, v, d in cards) + "</div>")

    # 3) seen vs unseen
    if seen or unseen:
        def row(name, s, hl=False):
            ci = (s.get("ci95") or {}).get("roc_auc")
            ci_s = f'<span class="ci"> ({_f3(ci[0])}–{_f3(ci[1])})</span>' if isinstance(ci, list) and len(ci) == 2 else ""
            abst = _pct(1 - float(s["coverage"])) if s.get("coverage") is not None else "–"
            return (f'<tr{" class=hl" if hl else ""}><th>{name}</th><td>{int(s.get("n", 0)):,}</td>'
                    f"<td>{_pct(s.get('accuracy'))}</td><td>{_f3(s.get('f1'))}</td><td>{_f3(s.get('roc_auc'))}{ci_s}</td>"
                    f"<td>{_pct(s.get('detection_rate'))}</td><td>{_pct(s.get('false_positive_rate'))}</td>"
                    f"<td>{abst}</td><td>{_f3(s.get('ece'))}</td></tr>")
        parts.append(
            '<div class="tl-h">Seen vs never-seen generators</div><div class="tl-table-wrap"><table class="tl-table">'
            "<tr><th></th><th>Images</th><th>Accuracy</th><th>F1</th><th>ROC-AUC (95% CI)</th><th>Fakes caught</th>"
            "<th>Real flagged</th><th>Abstained</th><th>ECE</th></tr>"
            + (row("Seen generator families (test)", seen) if seen else "")
            + (row(f"Never-seen generators (test-unseen{f', {n_gen}' if n_gen else ''})", unseen, True) if unseen else "")
            + '</table></div><div class="tl-note">ECE = calibration error (lower is better). Calibration was fitted on '
              "seen generators, so probabilities are less reliable on never-seen ones.</div>")

    # 4) members and baselines
    models = m.get("models") or {}
    if models:
        items = sorted(models.items(), key=lambda kv: (_KIND_ORDER.get(kv[1].get("kind"), 9),
                                                       -float((kv[1].get("test_unseen") or {}).get("roc_auc") or 0)))

        def mrow(key, v):
            t, u = v.get("test") or {}, v.get("test_unseen") or {}
            kind = str(v.get("kind", ""))
            return (f'<tr{" class=hl" if kind == "deployed" else ""}><th>{html.escape(str(v.get("label", key)))}</th>'
                    f'<td><span class="tl-kind k-{html.escape(kind)}">{html.escape(kind)}</span></td>'
                    f"<td>{_f3(t.get('roc_auc'))}</td><td>{_f3(u.get('roc_auc'))}</td><td>{_pct(u.get('accuracy'))}</td>"
                    f"<td>{_pct(u.get('detection_rate'))}</td><td>{_pct(u.get('false_positive_rate'))}</td></tr>")
        hdr = ("<tr><th>System</th><th>Role</th><th>Seen AUC</th><th>Unseen AUC</th><th>Unseen acc.</th>"
               "<th>Fakes caught</th><th>Real flagged</th></tr>")
        main = [mrow(k, v) for k, v in items if v.get("kind") != "ablation"]
        abl = [mrow(k, v) for k, v in items if v.get("kind") == "ablation"]
        parts.append('<div class="tl-h">TruthLens vs its members and baselines</div><div class="tl-table-wrap">'
                     f'<table class="tl-table">{hdr}{"".join(main)}</table></div>')
        if abl:
            parts.append(f'<details class="tl-details"><summary>Show {len(abl)} ablations</summary>'
                         f'<div class="tl-table-wrap"><table class="tl-table">{hdr}{"".join(abl)}</table></div></details>')

    # 5) robustness
    rob = m.get("robustness") or {}
    if rob:
        rrows = []
        for k, v in rob.items():
            if not isinstance(v, dict):
                continue
            seen_auc = (((v.get("by_split") or {}).get("test") or {}).get("ensemble") or {}).get("roc_auc")
            fpr = v.get("false_positive_rate")
            bad = ' class="warn"' if fpr is not None and float(fpr) > 0.25 else ""
            rrows.append(f"<tr><th>{html.escape(_PERTURB.get(k, k))}</th><td>{_f3(v.get('roc_auc'))}</td>"
                         f"<td>{_pct(v.get('detection_rate'))}</td><td{bad}>{_pct(fpr)}</td><td>{_f3(seen_auc)}</td></tr>")
        n_rob = (m.get("robustness_meta") or {}).get("n_per_split")
        parts.append(
            '<div class="tl-h">Robustness to everyday edits</div><div class="tl-table-wrap"><table class="tl-table">'
            "<tr><th>Edit applied before analysis</th><th>Unseen AUC</th><th>Fakes caught</th><th>Real flagged</th>"
            "<th>Seen AUC</th></tr>" + "".join(rrows) + "</table></div>"
            '<div class="tl-note">JPEG recompression hides generator traces (more missed fakes); blur and downscaling '
            "make real photos look generated (more false alarms), which is why tiny uploads get <i>Uncertain</i>."
            + (f" Stratified subsample of {int(n_rob):,} images per split." if n_rob else "") + "</div>")

    # 6) per generator highlights
    big = [r for r in unseen_ai if int(r.get("n", 0)) >= 20]
    if big:
        big.sort(key=lambda r: float(r.get("accuracy", 0)), reverse=True)

        def glist(rs, cls):
            return "".join(
                f'<div class="tl-gen"><span class="g">{html.escape(pretty_source(r["source"]))}</span>'
                f'{_bar(r.get("accuracy", 0), cls)}<span class="v">{_pct(r.get("accuracy"), 0)}</span>'
                f'<span class="n">n={int(r.get("n", 0))}</span></div>' for r in rs)
        reals = sorted(unseen_real, key=lambda r: -float(r.get("accuracy", 0)))
        parts.append(
            '<div class="tl-h">Never-seen generators: share of fakes caught (n ≥ 20)</div><div class="tl-gens">'
            f'<div><div class="tl-sub">Easiest</div>{glist(big[:5], "z-real")}</div>'
            f'<div><div class="tl-sub">Hardest</div>{glist(big[::-1][:5], "z-ai")}</div>'
            + (f'<div><div class="tl-sub">Real photos correctly passed</div>{glist(reals, "z-real")}</div>' if reals else "")
            + "</div>")
    return '<div class="tl-metrics">' + "".join(parts) + "</div>"


_LABELS = {"auc": "AUC", "roc_auc": "ROC-AUC", "f1": "F1", "ece": "ECE", "acc": "Accuracy",
           "accuracy": "Accuracy", "ap": "AP", "tpr": "TPR", "fpr": "FPR", "eer": "EER",
           "test_unseen": "Unseen generators", "test": "Seen generators (test)", "val": "Validation"}
_ABBR = {"auc", "roc", "f1", "ece", "ap", "tpr", "fpr", "tnr", "eer", "cpu", "gpu", "mps", "jpeg", "clip",
         "cnn", "fft", "id", "vit", "mae", "nll", "pr"}
_WORDS = {"ms": "", "acc": "accuracy", "n": "n", "num": "#", "q75": "Q75", "q50": "Q50", "q90": "Q90"}
_PCT_HINTS = ("acc", "f1", "precision", "recall", "tpr", "fpr", "tnr", "rate", "coverage", "abstain")


def _label(k: str) -> str:
    k = str(k)
    if k.lower() in _LABELS:
        return _LABELS[k.lower()]
    words = []
    for i, w in enumerate(k.replace("-", "_").split("_")):
        if not w:
            continue
        lw = w.lower()
        if lw in _ABBR:
            words.append(w.upper())
        elif lw in _WORDS:
            if _WORDS[lw]:
                words.append(_WORDS[lw])
        elif any(c.isupper() for c in w):
            words.append(w)
        else:
            words.append(w.capitalize() if i == 0 else w)
    out = " ".join(words)
    return out[:1].upper() + out[1:]


def _fmt(k: str, v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        kl = str(k).lower()
        if "latency" in kl or kl.endswith("_ms") or kl.endswith("ms"):
            return f"{v:,.0f} ms"
        if 0.0 <= v <= 1.0 and any(h in kl for h in _PCT_HINTS):
            return f"{v * 100:.1f}%"
        if abs(v) < 10:
            return f"{v:.3f}"
        return f"{v:,.1f}"
    return html.escape(str(v))


def _generic_html(d: dict) -> str:
    """Fallback for an unknown schema: top-level numbers as cards, one level of nested numbers as tables."""
    out = []
    scalars = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if scalars:
        out.append('<div class="tl-stats">' + "".join(
            f'<div class="tl-stat"><div class="v">{_fmt(k, v)}</div><div class="k">{html.escape(_label(k))}</div></div>'
            for k, v in list(scalars.items())[:8]) + "</div>")
    for k, v in d.items():
        if isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()):
            rows = "".join(f"<tr><th>{html.escape(_label(a))}</th><td>{_fmt(a, b)}</td></tr>" for a, b in v.items())
            out.append(f'<div class="tl-h">{html.escape(_label(k))}</div><div class="tl-table-wrap">'
                       f'<table class="tl-table kv">{rows}</table></div>')
    return '<div class="tl-metrics">' + "".join(out) + "</div>"


def metrics_html(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ('<div class="tl-card tl-empty"><h3>Metrics coming soon</h3>'
                "<p>Evaluation results (accuracy, F1, ROC-AUC, calibration and latency, including generators never "
                "seen in training) will appear here once the final model is evaluated.</p></div>")
    data = read_metrics(p)
    if data is None:
        return '<div class="tl-card tl-empty"><h3>Metrics coming soon</h3><p>Could not read the metrics file yet.</p></div>'
    try:
        if isinstance(data.get("splits"), dict) and ("test_unseen" in data["splits"] or "test" in data["splits"]):
            return _v2_html(data)
    except Exception:  # never break the page over a schema surprise
        pass
    return _generic_html(data)
