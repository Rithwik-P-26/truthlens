"""Fit calibration, the ensemble and the decision rule on VAL; write ``models/config.json``.

* each member: Platt scaling of its logit on VAL (its calibrated probability is what the
  app shows per member),
* ensemble: equal-weight average of calibrated member logits followed by a final Platt
  calibration (``--method stacked_lr`` is available for comparison),
* decision: threshold 0.5 on the calibrated probability; "Uncertain" when the probability of
  the leaning class is below ``min_confidence``, the smallest value in [floor, 0.9] whose VAL
  selective accuracy reaches ``--target-acc``,
* heatmap scale: median per-image 99th percentile of positive CAM over VAL AI images,
* spectral reference percentiles from TRAIN (used only to phrase explanations).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.calibrate import choose_uncertain_band, expected_calibration_error, fit_ensemble  # noqa: E402
from truthlens.train import get_logger, load_cached, read_manifest, write_json  # noqa: E402

DISPLAY = {"clip": "CLIP probe", "cnn": "CNN artifacts", "freq": "Spectral"}
QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def _ref(values: np.ndarray) -> dict:
    v = np.percentile(values, QUANTILES)
    v = v + np.arange(len(v)) * 1e-9  # strictly increasing for np.interp
    return {"q": QUANTILES, "v": [float(x) for x in v]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--cache-dir", default=str(ROOT / "models" / "cache"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--members", default="clip,cnn")
    ap.add_argument("--method", default="mean_logit", choices=["mean_logit", "stacked_lr"])
    ap.add_argument("--target-acc", type=float, default=0.95)
    ap.add_argument("--floor", type=float, default=0.65, help="never issue a verdict below this confidence")
    ap.add_argument("--version", default="2.1.0")
    ap.add_argument("--data-version", default="v2.1")
    ap.add_argument("--clip-head", default=None,
                    help="cached head name to deploy instead of the selected one (e.g. clip_vitb16_multi)")
    ap.add_argument("--out", default="config.json", help="config file name inside --models-dir")
    ap.add_argument("--log", default=str(ROOT / "logs" / "train_ensemble.log"))
    args = ap.parse_args()

    log = get_logger(args.log)
    df = read_manifest(args.manifest, args.root)
    cache, models_dir = Path(args.cache_dir), Path(args.models_dir)
    val = df[df.split == "val"]
    yv = val["label"].to_numpy()
    members = [m for m in args.members.split(",") if m]
    pred_file = {m: m for m in members}
    if args.clip_head:
        pred_file["clip"] = args.clip_head
    zv = {m: load_cached(cache / "preds" / f"{pred_file[m]}.npz", val)["logits"] for m in members}

    ens = fit_ensemble(zv, yv, members, method=args.method)
    pv = ens.prob(zv)
    decision = choose_uncertain_band(pv, yv, threshold=0.5, target_accuracy=args.target_acc, floor=args.floor)
    curve = decision.pop("curve")
    # inputs whose shorter side is < 224 px must be upsampled; robustness tests show such real photos are
    # often called AI (downscale_0.5x), so the app withholds the verdict for them (input-based guard)
    decision["uncertain_if_upsampled"] = True
    log(f"ensemble {args.method} on {members}: VAL ECE {expected_calibration_error(pv, yv):.4f}; "
        f"min_confidence {decision['min_confidence']} -> coverage {decision['val_coverage']:.3f}, "
        f"selective acc {decision['val_selective_accuracy']:.4f}")

    head_sel = json.loads((Path(args.reports_dir) / "head_selection.json").read_text())
    clip_name, clip_file = head_sel["deployed_clip_head"], "clip_head.npz"
    if args.clip_head:
        import shutil

        clip_name, clip_file = args.clip_head, f"{args.clip_head}_head.npz"
        shutil.copy(cache / "preds" / f"{args.clip_head}_head.npz", models_dir / clip_file)
    members_cfg: dict[str, dict] = {
        "clip": {"display_name": DISPLAY["clip"], "backbone": head_sel["heads"][clip_name]["backbone"],
                 "head": clip_file, "head_name": clip_name},
        "freq": {"display_name": DISPLAY["freq"], "head": "freq_head.npz"},
    }
    cnn_info_path = Path(args.reports_dir) / "cnn_training.json"
    if cnn_info_path.exists():
        cnn_info = json.loads(cnn_info_path.read_text())
        cnn_pred = load_cached(cache / "preds" / "cnn.npz", val)
        ai = yv == 1
        scale = float(np.median(cnn_pred["cam_p99"][ai])) if ai.any() else 1.0
        members_cfg["cnn"] = {"display_name": DISPLAY["cnn"], "arch": cnn_info["config"]["arch"],
                              "weights": cnn_info["weights"], "input_size": 224,
                              "cam_scale": scale if scale > 0 else 1.0}

    train = df[df.split == "train"]
    spec = load_cached(cache / "features" / "spectral.npz", train)["stats"]
    spectral_ref = {}
    for cls, lab in (("real", 0), ("ai", 1)):
        m = train["label"].to_numpy() == lab
        spectral_ref[cls] = {name: _ref(spec[m, j]) for j, name in enumerate(["hf_ratio", "peak_score", "noise_level"])}

    from truthlens.features import CLIP_SPECS

    spec = CLIP_SPECS[members_cfg["clip"]["backbone"]]
    backbones = {"clip": f"open_clip {spec.arch} ({spec.pretrained}), frozen; probe {members_cfg['clip']['head_name']}"}
    if "cnn" in members_cfg:
        backbones["cnn"] = f"timm {members_cfg['cnn']['arch']} (ImageNet init, fine-tuned at 224 px)"
    config = {
        "version": args.version,
        "data_version": args.data_version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backbones": backbones,
        "preprocessing": "centre square crop -> Lanczos 224x224 -> JPEG q95 4:4:4 (same as the dataset)",
        "thresholds": {"uncertain_low": decision["uncertain_low"], "uncertain_high": decision["uncertain_high"],
                       "decision": decision["threshold"]},
        "members": members_cfg,
        "ensemble": ens.to_dict(),
        "decision": decision,
        "spectral_reference": spectral_ref,
        "fitted_on": {"split": "val", "n": int(len(val)), "n_ai": int(yv.sum())},
    }
    write_json(models_dir / args.out, config)
    suffix = "" if args.out == "config.json" else "_" + Path(args.out).stem
    write_json(Path(args.reports_dir) / f"decision_curve_val{suffix}.json", curve)
    log(f"wrote {models_dir / args.out}")


if __name__ == "__main__":
    main()
