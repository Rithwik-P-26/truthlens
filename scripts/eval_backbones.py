"""Benchmark candidate backbones for TruthLens.

For each CLIP image tower and each CNN we measure:

* parameters, fp32 weight size and GFLOPs (``torch.utils.flop_counter``; 1 MAC = 2 FLOPs),
* CPU latency at batch 1 with ``torch.set_num_threads(2)`` (mimics a free HF Space; see caveat),
* MPS throughput: CLIP feature extraction (fp32 / fp16 autocast) and CNN training steps,
* fp16-vs-fp32 and MPS-vs-CPU agreement of CLIP features (cosine similarity).

Caveat: this Mac's PyTorch uses Apple Accelerate (AMX) for GEMMs, which makes ViTs much
faster on CPU than on a 2-vCPU x86 Space. We therefore also report GFLOPs, and the
README/eval scripts extrapolate with a measured anchor (see ``reports/backbone_benchmark.md``).

Usage::

    .venv/bin/python scripts/eval_backbones.py --out reports/backbone_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.features import CLIP_SPECS, ClipEmbedder, log_spectrum_display, spectral_features  # noqa: E402


def gflops(module: torch.nn.Module, fn, *args) -> float:
    """Total forward GFLOPs of ``fn(*args)`` (counted by torch's FlopCounterMode).

    The module is put in train mode for counting because the fused inference fast path of
    ``nn.MultiheadAttention`` is invisible to the counter (CLIP has no dropout/BN, so the
    computation is identical).
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = module.training
    module.train()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        fn(*args)
    module.train(was_training)
    return fc.get_total_flops() / 1e9


def time_cpu(fn, runs: int, warmup: int = 2) -> dict:
    """Median/min wall-clock ms of ``fn()`` on CPU."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(ts), 1), "min_ms": round(min(ts), 1)}


def time_mps(fn, iters: int, items_per_iter: int, warmup: int = 2) -> float:
    """Items/second of ``fn()`` on MPS."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return round(items_per_iter * iters / (time.perf_counter() - t0), 1)


def bench_clip(key: str, threads: int, runs: int, batch: int) -> dict:
    """Benchmark one CLIP image tower (including the multi-layer CLS extraction path)."""
    print(f"[clip] {key}", flush=True)
    emb = ClipEmbedder(key, device="cpu")
    n_params = sum(p.numel() for p in emb.visual.parameters())
    x1 = torch.randn(1, 3, 224, 224)
    flops = gflops(emb.visual, lambda t: emb.visual.forward_intermediates(t, indices=list(emb.spec.blocks),
                                                             output_fmt="NLC", output_extra_tokens=True), x1)
    torch.set_num_threads(threads)
    cpu = time_cpu(lambda: emb.embed(x1), runs)
    from PIL import Image

    img = Image.fromarray((np.random.rand(768, 1024, 3) * 255).astype(np.uint8))
    pre = time_cpu(lambda: emb.preprocess(img), runs)
    ref = emb.embed(torch.randn(8, 3, 224, 224, generator=torch.Generator().manual_seed(0)))
    res = {
        "backbone": key, "arch": emb.spec.arch, "pretrained": emb.spec.pretrained,
        "params_M": round(n_params / 1e6, 1), "weights_MB_fp32": round(n_params * 4 / 2**20, 0),
        "gflops_224": round(flops, 1), "cpu_threads": threads, "cpu_latency": cpu,
        "cpu_preprocess_ms": pre["median_ms"],
    }
    if torch.backends.mps.is_available():
        emb.visual.to("mps")
        emb.device = torch.device("mps")
        xb = torch.randn(batch, 3, 224, 224)
        res["mps_img_per_s_fp32"] = time_mps(lambda: emb.embed(xb), iters=5, items_per_iter=batch)
        res["mps_img_per_s_fp16"] = time_mps(lambda: emb.embed(xb, amp=True), iters=5, items_per_iter=batch)
        xr = torch.randn(8, 3, 224, 224, generator=torch.Generator().manual_seed(0))
        for name, amp in (("fp32", False), ("fp16", True)):
            out = emb.embed(xr, amp=amp)
            cos = [float((a * b).sum(1).mean() / 1.0) for a, b in (
                (_unit(out[k]), _unit(ref[k])) for k in ref)]
            res[f"mps_{name}_vs_cpu_fp32_min_cos"] = round(min(cos), 6)
    return res


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def bench_cnn(arch: str, threads: int, runs: int, batch: int) -> dict:
    """Benchmark one timm CNN: CPU inference at 512/1024 px and MPS training throughput."""
    import timm

    print(f"[cnn] {arch}", flush=True)
    model = timm.create_model(arch, pretrained=True, num_classes=1).eval()
    n_params = sum(p.numel() for p in model.parameters())
    res = {"arch": arch, "params_M": round(n_params / 1e6, 1), "weights_MB_fp32": round(n_params * 4 / 2**20, 1)}
    for side in (256, 512, 1024):
        res[f"gflops_{side}"] = round(gflops(model, model, torch.randn(1, 3, side, side)), 1)
    torch.set_num_threads(threads)
    for side in (512, 1024):
        x = torch.randn(1, 3, side, side)
        with torch.inference_mode():
            res[f"cpu_latency_{side}"] = time_cpu(lambda: model(x), max(3, runs // 2))
    if torch.backends.mps.is_available():
        m = model.to("mps").train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        xb = torch.randn(batch, 3, 256, 256, device="mps")
        yb = torch.randint(0, 2, (batch,), device="mps").float()
        lossf = torch.nn.BCEWithLogitsLoss()

        def step(dtype: torch.dtype | None) -> None:
            opt.zero_grad(set_to_none=True)
            if dtype is None:
                loss = lossf(m(xb).squeeze(1), yb)
            else:
                with torch.autocast(device_type="mps", dtype=dtype):
                    loss = lossf(m(xb).squeeze(1).float(), yb)
            loss.backward()
            opt.step()

        res["mps_train_img_per_s_256_fp32"] = time_mps(lambda: step(None), iters=5, items_per_iter=batch)
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            try:
                res[f"mps_train_img_per_s_256_{name}"] = time_mps(lambda: step(dt), iters=5, items_per_iter=batch)
            except Exception as e:  # noqa: BLE001 - report unsupported dtypes instead of failing
                res[f"mps_train_img_per_s_256_{name}"] = f"unsupported: {type(e).__name__}"
        m.eval()
        with torch.inference_mode():
            x1 = torch.randn(1, 3, 768, 768, device="mps")
            res["mps_infer_img_per_s_768_bs1"] = time_mps(lambda: m(x1), iters=10, items_per_iter=1)
        del m, opt
        torch.mps.empty_cache()
    return res


def bench_misc(threads: int, runs: int) -> dict:
    """CPU cost of the spectral member and the display spectrum on a 1024x768 image."""
    from PIL import Image

    torch.set_num_threads(threads)
    img = Image.fromarray((np.random.rand(768, 1024, 3) * 255).astype(np.uint8))
    return {
        "spectral_features_ms": time_cpu(lambda: spectral_features(img), runs)["median_ms"],
        "log_spectrum_display_ms": time_cpu(lambda: log_spectrum_display(img), runs)["median_ms"],
    }


def to_markdown(results: dict) -> str:
    """Render the benchmark as two markdown tables."""
    lines = ["## CLIP image towers (224 px)", "",
             "| backbone | params (M) | GFLOPs | CPU ms (2 thr, bs1) | MPS img/s fp32 | MPS img/s fp16 | fp16 min cos |",
             "|---|---|---|---|---|---|---|"]
    for r in results["clip"]:
        lines.append(f"| {r['backbone']} ({r['arch']}/{r['pretrained']}) | {r['params_M']} | {r['gflops_224']} | "
                     f"{r['cpu_latency']['median_ms']} | {r.get('mps_img_per_s_fp32', '-')} | "
                     f"{r.get('mps_img_per_s_fp16', '-')} | {r.get('mps_fp16_vs_cpu_fp32_min_cos', '-')} |")
    lines += ["", "## CNN artifact detectors", "",
              "| arch | params (M) | MB fp32 | GFLOPs 512 | CPU ms 512 | CPU ms 1024 | MPS train img/s 256 (fp32/fp16/bf16) |",
              "|---|---|---|---|---|---|---|"]
    for r in results["cnn"]:
        lines.append(f"| {r['arch']} | {r['params_M']} | {r['weights_MB_fp32']} | {r['gflops_512']} | "
                     f"{r['cpu_latency_512']['median_ms']} | {r['cpu_latency_1024']['median_ms']} | "
                     f"{r.get('mps_train_img_per_s_256_fp32')}/{r.get('mps_train_img_per_s_256_fp16')}/"
                     f"{r.get('mps_train_img_per_s_256_bf16')} |")
    lines += ["", f"Spectral features: {results['misc']['spectral_features_ms']} ms; display spectrum: "
              f"{results['misc']['log_spectrum_display_ms']} ms (1024x768, CPU)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="clip_vitb32,clip_vitb16,clip_vitl14")
    ap.add_argument("--cnn", default="efficientnet_b0,resnet18,resnet50,convnext_tiny")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "reports" / "backbone_benchmark.json"))
    args = ap.parse_args()

    results: dict = {"device": {"torch": torch.__version__, "mps": torch.backends.mps.is_available(),
                                "blas": "accelerate" if "accelerate" in torch.__config__.show() else "other"},
                     "clip": [], "cnn": []}
    for key in [k for k in args.clip.split(",") if k]:
        assert key in CLIP_SPECS, key
        results["clip"].append(bench_clip(key, args.threads, args.runs, args.batch))
        print(json.dumps(results["clip"][-1]), flush=True)
    for arch in [a for a in args.cnn.split(",") if a]:
        results["cnn"].append(bench_cnn(arch, args.threads, args.runs, args.batch))
        print(json.dumps(results["cnn"][-1]), flush=True)
    results["misc"] = bench_misc(args.threads, args.runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    out.with_suffix(".md").write_text(to_markdown(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
