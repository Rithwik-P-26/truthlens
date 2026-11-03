"""TruthLens: is this image a real photo or AI-generated?

Gradio app (Hugging Face Spaces entry point). Run locally with:
    .venv/bin/python app.py            # http://127.0.0.1:7860
Environment variables:
    PORT / GRADIO_SERVER_PORT  port to listen on (default 7860)
    TRUTHLENS_FORCE_STUB=1     always use the simulated detector (UI development)
    TRUTHLENS_CONFIG           model config in models/ (default config.json; config_fast.json = ViT-B/16, faster)
    TRUTHLENS_DEVICE           torch device for the detector (default cpu locally, cuda on ZeroGPU)
    SPACES_ZERO_GPU            set by Hugging Face ZeroGPU hardware: models go to CUDA at startup and
                               predictions run inside @spaces.GPU (15 s per image, up to 60 s per batch)
    TRUTHLENS_EXAMPLES_DIR     alternative examples folder (default: examples/)
    TRUTHLENS_METRICS          alternative metrics.json (default: reports/metrics.json)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from truthlens import ui  # noqa: E402
from truthlens.stub import StubDetector  # noqa: E402

try:  # Hugging Face ZeroGPU: `spaces` is preinstalled there; locally it is usually absent
    import spaces  # noqa: E402

    GPU = spaces.GPU
except ImportError:  # no-op stand-in so the same code runs on CPU/MPS
    def GPU(fn=None, **_kwargs):
        return fn if callable(fn) else (lambda f: f)

# ZeroGPU sets SPACES_ZERO_GPU (read by the `spaces` package itself to enable the GPU decorator).
ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("truthlens.app")

MODELS_DIR = ROOT / "models"
CONFIG_NAME = os.environ.get("TRUTHLENS_CONFIG") or "config.json"  # e.g. config_fast.json (ViT-B/16, ~2x faster)
CONFIG_PATH = MODELS_DIR / CONFIG_NAME
# Local default is CPU (matches the evaluated latency); ZeroGPU places the models on CUDA at startup.
DEVICE = os.environ.get("TRUTHLENS_DEVICE") or ("cuda" if ZERO_GPU else "cpu")
METRICS_PATH = Path(os.environ.get("TRUTHLENS_METRICS") or ROOT / "reports" / "metrics.json")
FIGURES_DIR = ROOT / "reports" / "figures"
EXAMPLES_DIR = Path(os.environ.get("TRUTHLENS_EXAMPLES_DIR") or ROOT / "examples")
ASSETS_DIR = ROOT / "assets"

MAX_ANALYZE_SIDE = 4096  # longest side passed to the detector (guards memory on huge uploads)
MAX_DISPLAY_SIDE = 1024  # longest side of images sent back to the browser
MAX_BATCH = 50

# --------------------------------------------------------------------------- detector


def _read_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _band(cfg: dict) -> tuple[float, float]:
    """Uncertainty band for the gauge, read from models/config.json (decision or thresholds)."""
    for sec in (cfg.get("decision"), cfg.get("thresholds"), cfg):
        if not isinstance(sec, dict):
            continue
        try:
            lo, hi = float(sec["uncertain_low"]), float(sec["uncertain_high"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 <= lo <= hi <= 1.0:
            return lo, hi
    return ui.DEFAULT_BAND


def load_detector():
    cfg = _read_config()
    if os.environ.get("TRUTHLENS_FORCE_STUB") == "1":
        return StubDetector(), True, "Simulated results (forced by TRUTHLENS_FORCE_STUB).", cfg
    if CONFIG_PATH.is_file():
        try:
            import torch

            from truthlens.inference import Detector

            if DEVICE.startswith("cuda"):
                # fp32 everywhere for parity with the evaluated pipeline (no TF32 shortcuts on Ampere+ GPUs)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            if ZERO_GPU:
                # Outside @spaces.GPU there is no real GPU (CUDA is emulated), so skip the dummy warm-up pass.
                Detector._warmup = lambda self: None
            t0 = time.perf_counter()
            det = Detector(models_dir=str(MODELS_DIR), device=DEVICE, config=CONFIG_NAME)
            log.info("Loaded Detector (%s, %s) in %.1fs", CONFIG_NAME, det.device, time.perf_counter() - t0)
            return det, False, "", cfg
        except Exception:
            log.exception("Failed to load the real Detector; falling back to the stub")
            return StubDetector(), True, "The model failed to load, so results are simulated.", cfg
    log.warning("models/%s not found; using StubDetector (demo mode)", CONFIG_NAME)
    return StubDetector(), True, "Model weights not found, so results are simulated.", cfg


DETECTOR, IS_STUB, DEMO_NOTICE, CONFIG = load_detector()
BAND = _band(CONFIG)
_LOCK = threading.Lock()


def _mode_label() -> str:
    if IS_STUB:
        return "demo mode (simulated)"
    ver = str(getattr(DETECTOR, "version", None) or CONFIG.get("version") or "")
    label = "TruthLens " + (ver if ver.startswith("v") else f"v{ver}") if ver else "TruthLens"
    if CONFIG_NAME != "config.json":
        label += " fast" if "fast" in CONFIG_NAME else f" ({CONFIG_NAME})"
    dev = str(getattr(DETECTOR, "device", "") or "")
    return f"{label} on {dev.upper() if dev in ('cpu', 'mps', 'cuda') else dev}" if dev else label


MODE = _mode_label()
METRICS = ui.read_metrics(METRICS_PATH) or {}
GEN_STATS = ui.generator_stats(METRICS)


def _unseen_facts(m: dict) -> tuple[int, float | None, list[str]]:
    """(number of never-seen generators, their miss rate, hardest families) from metrics.json, for UI copy."""
    gens = [r for r in m.get("per_generator") or [] if isinstance(r, dict) and r.get("split") == "test_unseen"
            and str(r.get("label")).lower() in ("ai", "1")]
    det = ((m.get("splits") or {}).get("test_unseen") or {}).get("detection_rate")
    hard = sorted((r for r in gens if int(r.get("n", 0)) >= 30), key=lambda r: float(r.get("accuracy", 1)))[:3]
    return len(gens), (1 - float(det)) if det is not None else None, [ui.pretty_source(r["source"]) for r in hard]


N_UNSEEN, UNSEEN_MISS, HARDEST = _unseen_facts(METRICS)

EXAMPLES = ui.discover_examples(EXAMPLES_DIR)
EXAMPLES_BY_SHA = {e["sha1"]: e for e in EXAMPLES}
MAIN_EXAMPLES = [e for e in EXAMPLES if not e["hard"]]
HARD_EXAMPLES = [e for e in EXAMPLES if e["hard"]]
log.info("Found %d example images (%d hard cases)", len(EXAMPLES), len(HARD_EXAMPLES))

# --------------------------------------------------------------------------- handlers


@GPU(duration=15)
def _predict_gpu(im):
    return DETECTOR.predict(im)


def _batch_duration(images):
    return min(60, 10 + len(images))


@GPU(duration=_batch_duration)
def _predict_many_gpu(images):
    return [DETECTOR.predict(im) for im in images]


def _predict(im):
    """One prediction. On ZeroGPU this borrows a GPU for up to 15 s; locally it runs under a lock."""
    if ZERO_GPU and not IS_STUB:
        return _predict_gpu(im)
    with _LOCK:
        return DETECTOR.predict(im)


def _predict_many(images):
    """Batch predictions in ONE GPU allocation on ZeroGPU (cheaper on the daily quota than one call per image)."""
    if ZERO_GPU and not IS_STUB:
        return _predict_many_gpu(images)
    with _LOCK:
        return [DETECTOR.predict(im) for im in images]


def analyze(path: str | None, opacity: float):
    hidden = gr.update(visible=False)
    if not path:
        return ui.empty_card(), None, None, None, hidden
    try:
        im = ui.load_image(path, MAX_ANALYZE_SIDE)
    except Exception as e:
        log.warning("Unreadable upload: %s", e)
        return ui.error_card("This file could not be read as an image. Try a JPEG, PNG or WebP."), None, None, None, hidden
    try:
        res = _predict(im)
    except Exception as e:
        log.exception("Prediction failed")
        return ui.error_card(f"The detector raised an error ({type(e).__name__})."), None, None, None, hidden

    try:
        ex = EXAMPLES_BY_SHA.get(ui.file_sha1(path))
    except OSError:
        ex = None

    disp = ui.downscale(im, MAX_DISPLAY_SIDE)
    disp_arr = np.asarray(disp, dtype=np.uint8)
    heat = res.get("heatmap")
    heat_disp = ui.resize_map(heat, disp.size) if heat is not None else None
    box = ui.analysed_box(disp.size)
    spec_img = ui.spectrum_image(res.get("spectrum"))

    card = ui.result_card(res, BAND, example=ex, gen_stats=GEN_STATS, image_size=im.size, mode=MODE,
                          heatmap_available=heat is not None)
    state = {"img": disp_arr, "heat": heat_disp, "box": box}
    return card, ui.overlay(disp_arr, heat_disp, opacity, box), spec_img, state, gr.update(visible=True)


def update_overlay(state, opacity: float):
    if not state:
        return gr.update()
    return ui.overlay(state["img"], state.get("heat"), opacity, state.get("box"))


def _picker(items: list[dict]):
    def pick(evt: gr.SelectData):
        i = evt.index if isinstance(evt.index, int) else evt.index[0]
        return items[i]["path"]
    return pick


def run_batch(files, progress=gr.Progress()):
    if not files:
        return None, '<div class="tl-meta">Add some images first.</div>', gr.update(visible=False)
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files][:MAX_BATCH]
    rows, model_names = [], []
    counts = {"AI-generated": 0, "Real": 0, "Uncertain": 0, "Error": 0}
    loaded = []  # (name, image or None)
    for p in progress.tqdm(paths, desc="Reading images"):
        name = Path(p).name
        try:
            loaded.append((name, ui.load_image(p, MAX_ANALYZE_SIDE)))
        except Exception as e:
            log.warning("Batch item unreadable (%s): %s", name, e)
            loaded.append((name, None))
    progress(0.5, desc=f"Analyzing {sum(im is not None for _, im in loaded)} images")
    good = [im for _, im in loaded if im is not None]
    try:
        results = iter(_predict_many(good)) if good else iter(())
    except Exception:
        log.exception("Batch prediction failed")
        return None, ui.error_card("The detector raised an error on this batch.").replace("tl-card", "tl-card tl-sm"), \
            gr.update(visible=False)
    for name, im in loaded:
        if im is None:
            counts["Error"] += 1
            rows.append({"File": name, "Verdict": "Error", "Note": "unreadable file"})
            continue
        res = next(results)
        v = str(res.get("verdict", "Uncertain"))
        counts[v] = counts.get(v, 0) + 1
        row = {"File": name, "Verdict": v, "P(AI)": round(float(res["prob_ai"]), 4),
               "Confidence": round(float(res.get("confidence", 0.0)), 4)}
        for m, mp in (res.get("models") or {}).items():
            if m not in model_names:
                model_names.append(m)
            row[m] = round(float(mp), 4)
        row["Size"] = f"{im.size[0]}x{im.size[1]}"
        row["Note"] = "too small to judge" if v == "Uncertain" and min(im.size) < ui.MIN_SIDE else ""
        row["Latency (ms)"] = round(float(res.get("latency_ms", 0.0)), 1)
        rows.append(row)
    headers = ["File", "Verdict", "P(AI)", "Confidence", *model_names, "Size", "Note", "Latency (ms)"]
    table = [[r.get(h, "") for h in headers] for r in rows]

    out_dir = Path(tempfile.mkdtemp(prefix="truthlens_"))
    csv_path = out_dir / "truthlens_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(table)

    note = f" (first {MAX_BATCH} of {len(files)})" if len(files) > MAX_BATCH else ""
    summary = (
        f'<div class="tl-batch-summary"><b>{len(paths)} images{note}</b>'
        f'<span class="pill ai">{counts["AI-generated"]} AI-generated</span>'
        f'<span class="pill real">{counts["Real"]} real</span>'
        f'<span class="pill unsure">{counts["Uncertain"]} uncertain</span>'
        + (f'<span class="pill err">{counts["Error"]} unreadable</span>' if counts["Error"] else "")
        + "</div>"
    )
    return gr.update(value=table, headers=headers), summary, gr.update(value=str(csv_path), visible=True)


def load_metrics():
    return ui.metrics_html(METRICS_PATH)


def _figures() -> list[tuple[str, str]]:
    if not FIGURES_DIR.is_dir():
        return []
    order = ["roc_curves", "per_generator_unseen_auc", "robustness", "reliability_diagram", "risk_coverage",
             "confusion_matrices", "cnn_training_curves"]
    files = [p for p in FIGURES_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
             and not p.name.startswith("data_")]
    files.sort(key=lambda p: (order.index(p.stem) if p.stem in order else len(order), p.name))
    names = {"roc_curves": "ROC curves (seen vs never-seen generators)",
             "per_generator_unseen_auc": "ROC-AUC per never-seen generator",
             "robustness": "Robustness to JPEG, downscaling and blur", "reliability_diagram": "Calibration (reliability)",
             "risk_coverage": "Accuracy vs coverage when abstaining", "confusion_matrices": "Confusion matrices",
             "cnn_training_curves": "CNN training curves"}
    return [(str(p), names.get(p.stem, p.stem.replace("_", " ").capitalize())) for p in files][:12]


# --------------------------------------------------------------------------- layout

LOGO = (
    '<svg class="tl-logo" viewBox="0 0 48 48" aria-hidden="true"><defs><linearGradient id="tlg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs>'
    '<rect width="48" height="48" rx="12" fill="url(#tlg)"/>'
    '<circle cx="21" cy="21" r="10" fill="none" stroke="#fff" stroke-width="3.5"/>'
    '<path d="m28.5 28.5 8 8" stroke="#fff" stroke-width="4" stroke-linecap="round"/>'
    '<path d="M16.5 21.5l3 3 6-6.5" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

HEADER = f"""
<header class="tl-header">
  <div class="tl-brand">{LOGO}<div><h1>TruthLens</h1>
  <p>Is this image a real photo or AI-generated? A calibrated answer, with the evidence.</p></div></div>
  <div class="tl-chips"><span>Calibrated ensemble</span><span>Abstains when unsure</span><span>Tested on {f"{N_UNSEEN} " if N_UNSEEN else ""}unseen generators</span><span>Heatmap + FFT evidence</span></div>
</header>
""" + (f'<div class="tl-demo" role="note"><b>Demo mode.</b> {DEMO_NOTICE}</div>' if IS_STUB else "")

DISCLAIMER = (
    '<div class="tl-disclaimer"><b>Responsible use.</b> TruthLens gives a probability, not proof. It can be wrong, '
    "especially on generators it has not seen, heavily compressed or edited images. Do not use it as the sole basis "
    "for accusing anyone or for legal, journalistic or moderation decisions. TruthLens does not save or log your images.</div>"
)

LIMITATIONS_MD = """
### What TruthLens can't do (yet)

- **Brand-new generators.** We test on {n_unseen}generators that were never used in training, and report those numbers separately. {miss}Some families are much harder than others{hardest}. A new model can fool it.
- **Compressed, blurred or resized images.** Social-media recompression hides generator traces, so more fakes slip through. Blur and upscaling do the opposite and make real photos look generated. Images smaller than 224 px on their shorter side always get *Uncertain*.
- **Only the central square is analysed.** Every image is centre-cropped to a square and resized to 224×224 px, exactly like the training data. Content near the edges of wide or tall images is not examined (the heatmap dims it).
- **Edited real photos.** Real photos with AI inpainting, filters, upscaling or retouching sit in between. TruthLens classifies the whole image; it does not localize small edits.
- **Adversarial attacks.** Someone deliberately evading detection (adding noise, re-photographing a screen, adversarial perturbations) can often succeed.
- **Still images only.** It is not a face-swap or video deepfake detector.

### Not forensic evidence

The output is a calibrated probability from statistical models, not proof. A "Real" verdict does not certify an image as authentic, and an "AI-generated" verdict does not prove manipulation. Use it as one signal alongside provenance checks (C2PA content credentials, reverse image search, source verification), and never as the sole basis for accusing someone, for moderation or legal decisions, or for news reporting.

### How to read the result

- **Probability gauge.** The calibrated probability that the image is AI-generated. On generators like those seen in training, about 80% of images scored 80% really are AI-generated; on brand-new generators the probabilities are less reliable.
- **Uncertain.** Between 35% and 65%, or when the image is too small, TruthLens gives no verdict instead of guessing. That is a feature: look for other evidence.
- **Heatmap.** Where the artifact CNN found generation traces inside the analysed square. A hint about where the model looked, not a map of what was edited.
- **Spectrum.** The 2-D frequency spectrum of the analysed 224 px view. Bright periodic dots or grid lines away from the centre are often traces of the upsampling layers in image generators.

### Privacy

Uploaded images are processed in memory to compute the result. TruthLens does not save, log or train on them. (The hosting platform may keep temporary files for the duration of your session.)

### Fairness and misuse

Errors are not evenly distributed: performance varies with image source, camera, content and style (for example, real DOCCI photos are wrongly flagged more often than ImageNet photos). False "AI-generated" flags on real photos can hurt real people, which is why the app abstains when unsure and why a result should never be used alone to accuse anyone.
""".format(
    n_unseen=f"{N_UNSEEN} " if N_UNSEEN else "",
    miss=(f"TruthLens still misses about {UNSEEN_MISS:.0%} of their images at its default threshold. "
          if UNSEEN_MISS is not None else ""),
    hardest=f" (for example {', '.join(HARDEST)})" if HARDEST else "",
)

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#f6f7fb",
    body_background_fill_dark="#0b0f19",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.04)",
    button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_600, *secondary_600)",
    button_primary_text_color="white",
)

with gr.Blocks(title="TruthLens · AI-generated image detector", analytics_enabled=False) as demo:
    gr.HTML(HEADER)
    with gr.Tabs(elem_id="tl-tabs"):
        # ---------------------------------------------------------------- Detect
        with gr.Tab("Detect", id="detect"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=300):
                    inp = gr.Image(type="filepath", image_mode=None, format="png", sources=["upload", "clipboard"],
                                   label="Image", height=330, elem_id="tl-input",
                                   placeholder="Drop an image here, click to upload, or paste from the clipboard")
                    with gr.Row():
                        clear_btn = gr.ClearButton(value="Clear", size="md")
                        run_btn = gr.Button("Analyze image", variant="primary", size="md", scale=2)
                    galleries = []
                    if MAIN_EXAMPLES:
                        gr.HTML('<div class="tl-h tl-ex-h">Try an example</div>')
                        galleries.append((gr.Gallery(value=[(e["path"], e["caption"]) for e in MAIN_EXAMPLES],
                                                     show_label=False, columns=5, height="auto", allow_preview=False,
                                                     object_fit="cover", interactive=False, buttons=[],
                                                     elem_id="tl-examples", elem_classes="tl-gallery"), MAIN_EXAMPLES))
                    if HARD_EXAMPLES:
                        gr.HTML('<div class="tl-h tl-ex-h">Hard cases <span class="tl-h-note">where TruthLens '
                                "gets it wrong or abstains</span></div>")
                        galleries.append((gr.Gallery(value=[(e["path"], e["caption"]) for e in HARD_EXAMPLES],
                                                     show_label=False, columns=5, height="auto", allow_preview=False,
                                                     object_fit="cover", interactive=False, buttons=[],
                                                     elem_id="tl-hard", elem_classes="tl-gallery"), HARD_EXAMPLES))
                    if not EXAMPLES:
                        gr.HTML('<div class="tl-meta">Example images will appear here once added to <code>examples/</code>.</div>')
                with gr.Column(scale=7, min_width=320):
                    card = gr.HTML(ui.empty_card(), elem_id="tl-result")
                    with gr.Column(visible=False) as evidence:
                        with gr.Row(equal_height=True):
                            heat_out = gr.Image(label="Where the CNN saw generation artifacts", interactive=False,
                                                buttons=["fullscreen", "download"], height=300)
                            spec_out = gr.Image(label="Frequency spectrum of the analysed view", interactive=False,
                                                buttons=["fullscreen", "download"], height=300)
                        opacity = gr.Slider(0, 1, value=0.55, step=0.05, label="Heatmap opacity")
                    gr.HTML(DISCLAIMER)
            state = gr.State(None)
            outs = [card, heat_out, spec_out, state, evidence]
            inp.change(analyze, [inp, opacity], outs, api_name="detect", concurrency_id="detector",
                       show_progress="minimal", trigger_mode="always_last")
            run_btn.click(analyze, [inp, opacity], outs, api_visibility="private", concurrency_id="detector",
                          show_progress="minimal")
            opacity.input(update_overlay, [state, opacity], heat_out, show_progress="hidden",
                          trigger_mode="always_last", api_visibility="private")
            clear_btn.add([inp])
            for gal, items in galleries:
                gal.select(_picker(items), None, inp, api_visibility="private", show_progress="hidden")

        # ---------------------------------------------------------------- Batch
        with gr.Tab("Batch", id="batch"):
            gr.Markdown(f"Analyze up to {MAX_BATCH} images at once and download the results as CSV.",
                        elem_classes="tl-lead")
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=280):
                    files = gr.File(file_count="multiple", file_types=["image"], label="Images", height=220)
                    batch_btn = gr.Button("Analyze all", variant="primary")
                with gr.Column(scale=8, min_width=320):
                    batch_summary = gr.HTML('<div class="tl-meta">Results will appear here.</div>')
                    batch_table = gr.Dataframe(headers=["File", "Verdict", "P(AI)", "Confidence", "Latency (ms)"],
                                               interactive=False, wrap=True, max_height=420, show_label=False)
                    batch_dl = gr.DownloadButton("Download CSV", visible=False, size="md")
            batch_btn.click(run_batch, files, [batch_table, batch_summary, batch_dl], api_name="batch",
                            concurrency_id="detector")

        # ---------------------------------------------------------------- How it works
        with gr.Tab("How it works", id="how"):
            gr.Markdown("### Pipeline", elem_classes="tl-lead")
            gr.HTML(ui.pipeline_html(CONFIG))
            gr.Markdown("### Results on held-out test data", elem_classes="tl-lead")
            metrics_out = gr.HTML(load_metrics())
            figs = _figures()
            if figs:
                gr.Markdown("### Evaluation figures", elem_classes="tl-lead")
                gr.Gallery(value=figs, columns=3, height="auto", show_label=False, allow_preview=True,
                           object_fit="contain", interactive=False, elem_id="tl-figs")

        # ---------------------------------------------------------------- Limitations
        with gr.Tab("Limitations & responsible use", id="limits"):
            gr.Markdown(LIMITATIONS_MD, elem_classes="tl-prose")

    gr.HTML('<footer class="tl-footer">TruthLens · Neural Storm 2026 · FT-02 Deepfake &amp; Synthetic Media Detection · '
            "results are probabilistic, not proof</footer>")
    demo.load(load_metrics, None, metrics_out, api_visibility="private", show_progress="hidden")


if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"),
        server_port=port,
        theme=theme,
        css_paths=[ASSETS_DIR / "style.css"],
        allowed_paths=[str(EXAMPLES_DIR), str(FIGURES_DIR)],
        max_file_size="25mb",
        footer_links=["api"],
        show_error=False,
    )
