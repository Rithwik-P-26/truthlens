"""Verify the deployed inference contract and the invariants the reports rely on.

Checks (CPU, ``torch.set_num_threads(2)`` like a free HF Space):

1. ``Detector(models_dir, device="cpu")`` loads, ``.device == "cpu"`` and ``config.json`` has the
   Contract-3 keys (``version``, ``backbones``, ``thresholds.uncertain_low|high``).
2. ``predict`` output types/ranges (Contract 2) on test images and on awkward inputs
   (tiny, RGBA, grayscale, panorama, 12 MP): heatmap at the ORIGINAL input size, values in [0, 1].
3. CAM exactness: ``mean(cam) + bias == logit`` and Grad-CAM (autograd) == ReLU(CAM) / (h*w);
   also the correlation with the ``pytorch-grad-cam`` library's GradCAM.
4. Train/eval/deploy consistency: ``Detector`` probabilities on test files vs the cached evaluation
   predictions (differences come only from fp16-on-GPU features and the JPEG q95 re-encode
   that ``normalize_image`` applies to already-normalised files).

Writes ``reports/contract_check.json``; exits non-zero if a hard check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.features import load_image  # noqa: E402
from truthlens.inference import Detector  # noqa: E402
from truthlens.models import cnn_input  # noqa: E402
from truthlens.train import load_cached, read_manifest, write_json  # noqa: E402

VERDICTS = {"AI-generated", "Real", "Uncertain"}


def check_output(out: dict, size: tuple[int, int]) -> list[str]:
    """Return a list of contract violations for one ``predict`` output."""
    errs = []
    W, H = size
    if not 0.0 <= out["prob_ai"] <= 1.0:
        errs.append("prob_ai out of range")
    if out["verdict"] not in VERDICTS:
        errs.append(f"bad verdict {out['verdict']}")
    if not 0.5 <= out["confidence"] <= 1.0:
        errs.append("confidence out of [0.5, 1]")
    if not out["models"] or not all(0.0 <= v <= 1.0 for v in out["models"].values()):
        errs.append("models probs invalid")
    hm = out["heatmap"]
    if hm is not None and (hm.shape != (H, W) or hm.dtype != np.float32 or hm.min() < 0 or hm.max() > 1):
        errs.append(f"heatmap invalid: shape {hm.shape} vs {(H, W)}, dtype {hm.dtype}")
    sp = out["spectrum"]
    if sp.ndim != 2 or sp.dtype != np.float32 or sp.min() < 0 or sp.max() > 1:
        errs.append("spectrum invalid")
    ex = out["explanation"]
    if not (isinstance(ex, list) and 2 <= len(ex) <= 4 and all(isinstance(s, str) and s for s in ex)):
        errs.append(f"explanation invalid ({len(ex)} items)")
    if not out["latency_ms"] > 0:
        errs.append("latency_ms invalid")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--cache-dir", default=str(ROOT / "models" / "cache"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    report: dict = {"threads": args.threads}
    hard_fail: list[str] = []

    t0 = time.perf_counter()
    det = Detector(args.models_dir, device="cpu")
    report["load_s"] = round(time.perf_counter() - t0, 1)
    cfg = det.config
    c3 = {"device_is_cpu": det.device == "cpu", "version_str": isinstance(cfg.get("version"), str),
          "backbones": isinstance(cfg.get("backbones"), (dict, list)),
          "thresholds": all(isinstance(cfg.get("thresholds", {}).get(k), float) for k in ("uncertain_low", "uncertain_high"))}
    report["contract3_config"] = c3
    hard_fail += [f"contract3:{k}" for k, ok in c3.items() if not ok]

    # 2) awkward inputs
    rng = np.random.default_rng(0)
    odd = {
        "tiny_100x80": Image.fromarray(rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)),
        "rgba_300x300": Image.fromarray(rng.integers(0, 255, (300, 300, 4), dtype=np.uint8), mode="RGBA"),
        "gray_400x300": Image.fromarray(rng.integers(0, 255, (300, 400), dtype=np.uint8), mode="L"),
        "panorama_2000x500": Image.fromarray(rng.integers(0, 255, (500, 2000, 3), dtype=np.uint8)),
        "photo_4032x3024": Image.fromarray(rng.integers(0, 255, (3024, 4032, 3), dtype=np.uint8)),
    }
    report["odd_inputs"] = {}
    for name, im in odd.items():
        out = det.predict(im)
        errs = check_output(out, im.size)
        report["odd_inputs"][name] = {"errors": errs, "verdict": out["verdict"], "latency_ms": round(out["latency_ms"]),
                                      "explanation": out["explanation"]}
        hard_fail += [f"{name}:{e}" for e in errs]

    # 3) CAM exactness on a real test image
    df = read_manifest(args.manifest, args.root)
    test = df[df.split.isin(["test", "test_unseen"])]
    sample = test.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), args.n_images // 2),
                                                                              random_state=0))
    if det.cnn is not None:
        x, _ = cnn_input(load_image(sample.abspath.iloc[0]))
        x = x.unsqueeze(0)
        with torch.inference_mode():
            lg, cam = det.cnn.forward_with_cam(x)
        dec_err = float(abs(cam.mean() + det.cnn.bias - lg[0]))
        feats = det.cnn.net.forward_features(det.cnn.normalize(x)).detach().requires_grad_(True)
        y = det.cnn.net.forward_head(feats).squeeze()
        (g,) = torch.autograd.grad(y, feats)
        gradcam = torch.relu((g.mean((2, 3), keepdim=True) * feats).sum(1))[0].detach()
        ours = torch.relu(cam[0]) / cam[0].numel()
        gc_err = float((gradcam - ours).abs().max() / max(float(gradcam.abs().max()), 1e-12))
        lib_corr = None
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            class _Wrap(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, t):
                    return self.m.net(t)

            lib = GradCAM(model=_Wrap(det.cnn), target_layers=[det.cnn.net.bn2])
            lib_map = lib(input_tensor=det.cnn.normalize(x), targets=[ClassifierOutputTarget(0)])[0]
            ours_up = np.asarray(Image.fromarray(ours.numpy().astype(np.float32), mode="F").resize(
                lib_map.shape[::-1], Image.BILINEAR))
            if ours_up.std() > 0 and lib_map.std() > 0:
                lib_corr = float(np.corrcoef(ours_up.ravel(), lib_map.ravel())[0, 1])
        except Exception as e:  # noqa: BLE001 - library check is informative only
            lib_corr = f"skipped: {type(e).__name__}: {e}"
        report["cam"] = {"logit_decomposition_abs_err": dec_err, "gradcam_vs_cam_rel_err": gc_err,
                         "pytorch_grad_cam_correlation": lib_corr}
        if dec_err > 1e-3 or gc_err > 1e-4:
            hard_fail.append("cam exactness")

    # 4) consistency with cached evaluation predictions + per-image contract on real images
    cached = {}
    for m in det.ensemble.members:
        p = Path(args.cache_dir) / "preds" / f"{m}.npz"
        if p.exists():
            try:
                cached[m] = load_cached(p, sample)["logits"]
            except KeyError:
                pass
    diffs, errs_all, lat = [], [], []
    for i, row in enumerate(sample.itertuples()):
        im = load_image(row.abspath)
        out = det.predict(im)
        errs_all += check_output(out, im.size)
        lat.append(out["latency_ms"])
        if len(cached) == len(det.ensemble.members):
            z = {m: np.array([cached[m][i]]) for m in det.ensemble.members}
            diffs.append(abs(float(det.ensemble.prob(z)[0]) - out["prob_ai"]))
    report["real_images"] = {"n": len(sample), "contract_errors": sorted(set(errs_all)),
                             "median_latency_ms": float(np.median(lat))}
    if diffs:
        report["detector_vs_cached_eval"] = {"max_abs_prob_diff": float(np.max(diffs)),
                                             "mean_abs_prob_diff": float(np.mean(diffs))}
    hard_fail += [f"real:{e}" for e in set(errs_all)]
    report["hard_failures"] = hard_fail
    report["passed"] = not hard_fail
    write_json(Path(args.reports_dir) / "contract_check.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "odd_inputs"}, indent=1, default=str))
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
