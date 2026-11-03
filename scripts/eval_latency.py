"""CPU latency of each component and of ``Detector.predict`` with ``torch.set_num_threads(2)``.

Two threads mimic a free Hugging Face Space (2 vCPU). Caveat recorded in the output: on an
Apple-silicon Mac PyTorch's GEMMs run on Apple Accelerate/AMX, so ViT-heavy components are
substantially faster here than on a 2-vCPU x86 Space; verify on the Space itself
(``predict`` returns ``latency_ms``).

Writes ``reports/latency.json`` and the ``latency`` key of ``reports/metrics.json``.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.features import log_spectrum_display, normalize_image, spectral_features  # noqa: E402
from truthlens.inference import Detector  # noqa: E402
from truthlens.models import cnn_input, stack_features  # noqa: E402
from truthlens.train import get_logger, update_json, write_json  # noqa: E402


def _image(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (max(1, h // 16), max(1, w // 16), 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((w, h), Image.BICUBIC)  # smooth content + sensor-like noise
    noisy = np.clip(np.asarray(img, dtype=np.int16) + rng.integers(-8, 9, (h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _time(fn, runs: int) -> dict:
    fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"median_ms": statistics.median(ts), "p90_ms": ts[min(len(ts) - 1, int(0.9 * len(ts)))]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--sizes", default="224x224,1024x1024,4032x3024")
    ap.add_argument("--log", default=str(ROOT / "logs" / "eval_latency.log"))
    args = ap.parse_args()

    log = get_logger(args.log)
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    det = Detector(args.models_dir, device="cpu")
    load_s = time.perf_counter() - t0
    comps: dict[str, dict] = {}
    for spec in args.sizes.split(","):
        w, h = map(int, spec.split("x"))
        img = _image(w, h, 0)
        norm = normalize_image(img)[0]
        comps[f"normalize_{spec}"] = _time(lambda: normalize_image(img), args.runs)
        if det.clip is not None:
            comps[f"clip_{spec}"] = _time(lambda: det.clip_head.logits(
                stack_features(det.clip.embed_images([norm]), det.clip_head.feature_keys)), args.runs)
        if det.cnn is not None:
            def run_cnn() -> None:
                x, _ = cnn_input(norm)
                with torch.inference_mode():
                    det.cnn.forward_with_cam(x.unsqueeze(0))
            comps[f"cnn_{spec}"] = _time(run_cnn, args.runs)
        comps[f"spectral_features_{spec}"] = _time(lambda: spectral_features(norm), args.runs)
        comps[f"display_spectrum_{spec}"] = _time(lambda: log_spectrum_display(norm), args.runs)
        comps[f"predict_total_{spec}"] = _time(lambda: det.predict(img), args.runs)
        log(f"{spec}: " + ", ".join(f"{k.rsplit('_', 1)[0]} {v['median_ms']:.0f} ms"
                                    for k, v in comps.items() if k.endswith(spec)))
    fast_cfg = Path(args.models_dir) / "config_fast.json"
    if fast_cfg.exists():  # end-to-end latency of the ViT-B/16 fallback
        del det
        fast = Detector(args.models_dir, device="cpu", config="config_fast.json")
        for spec in args.sizes.split(","):
            w, h = map(int, spec.split("x"))
            img = _image(w, h, 0)
            comps[f"fast_predict_total_{spec}"] = _time(lambda: fast.predict(img), args.runs)
            log(f"fast config {spec}: {comps[f'fast_predict_total_{spec}']['median_ms']:.0f} ms")
    blas = "accelerate" if "accelerate" in torch.__config__.show() else "mkl/openblas"
    ref = "predict_total_1024x1024"
    out = {
        "latency_ms_cpu": comps.get(ref, next(iter(comps.values())))["median_ms"],
        "latency_ms_cpu_definition": f"median Detector.predict wall time, {ref.split('_')[-1]} upload, "
                                     f"torch threads={args.threads}",
        "threads": args.threads, "device": "cpu", "machine": f"{platform.machine()} {platform.processor()}",
        "blas": blas, "model_load_s": round(load_s, 1), "components": comps,
        "note": (f"torch.set_num_threads({args.threads}) on {platform.machine()} with BLAS={blas}. "
                 "Apple AMX makes ViT GEMMs much faster than on a 2-vCPU x86 Space; Space numbers must be "
                 "measured there (predict() returns latency_ms)."),
    }
    write_json(Path(args.reports_dir) / "latency.json", out)
    update_json(Path(args.reports_dir) / "metrics.json", "latency", out)
    log(f"model load {load_s:.1f}s; wrote latency.json")


if __name__ == "__main__":
    main()
