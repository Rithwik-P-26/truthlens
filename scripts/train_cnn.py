"""Fine-tune the CNN artifact detector and cache its predictions.

Training: the normalised 224x224 images (the exact deployment view) with robustness
augmentations on both classes (rescale-then-back, blur, JPEG quality 60-100, flips),
pool-balanced sampling, AdamW + cosine schedule; the checkpoint with the best VAL AUC is kept.
Prediction: val/test/test_unseen at 224 px, cached with CAM statistics for heatmap scaling.

Usage::

    .venv/bin/python scripts/train_cnn.py --arch efficientnet_b0 --epochs 6 2>&1 | tee logs/train_cnn.log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.features import pick_device  # noqa: E402
from truthlens.models import load_cnn  # noqa: E402
from truthlens.train import (AugConfig, CnnTrainConfig, get_logger, predict_cnn, read_manifest, train_cnn,  # noqa: E402
                             write_json)


def plot_history(history: list[dict], out: Path) -> None:
    """Training/validation curves for the README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] + 1 for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], marker="o", label="val")
    ax[0].set(xlabel="epoch", ylabel="BCE loss", title="CNN loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_auc"] for h in history], marker="o", color="C2")
    ax[1].set(xlabel="epoch", ylabel="ROC-AUC", title="VAL AUC")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--cache-dir", default=str(ROOT / "models" / "cache"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--pool-alpha", type=float, default=0.5)
    ap.add_argument("--predict-splits", default="val,test,test_unseen")
    ap.add_argument("--skip-train", action="store_true", help="only re-run predictions with saved weights")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=str(ROOT / "logs" / "train_cnn.log"))
    args = ap.parse_args()

    log = get_logger(args.log)
    df = read_manifest(args.manifest, args.root)
    models_dir, cache, reports = Path(args.models_dir), Path(args.cache_dir), Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    (cache / "preds").mkdir(parents=True, exist_ok=True)
    weights = models_dir / f"cnn_{args.arch}.safetensors"
    dev = pick_device(args.device)

    if not args.skip_train:
        cfg = CnnTrainConfig(arch=args.arch, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                             workers=args.workers, amp=args.amp, seed=args.seed, max_steps=args.max_steps,
                             pool_alpha=args.pool_alpha, aug=AugConfig())
        res = train_cnn(cfg, df[df.split == "train"], df[df.split == "val"], weights, str(dev), log)
        res["weights"] = weights.name
        write_json(reports / "cnn_training.json", res)
        plot_history(res["history"], reports / "figures" / "cnn_training_curves.png")
        log(f"saved {weights} ({weights.stat().st_size / 2**20:.1f} MB)")

    model = load_cnn(weights, args.arch, dev)
    sub = df[df.split.isin(args.predict_splits.split(","))]
    pred = predict_cnn(model, sub["abspath"].tolist(), dev, workers=min(args.workers, 6), log=log)
    np.savez(cache / "preds" / "cnn.npz", paths=np.array(sub["path"].tolist()), **pred)
    log(f"cached CNN predictions for {len(sub)} rows -> {cache / 'preds' / 'cnn.npz'}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
