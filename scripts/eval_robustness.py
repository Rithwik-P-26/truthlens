"""Robustness of each member and the deployed ensemble to common image degradations.

For a seeded, class- and source-stratified subsample of TEST and TEST-UNSEEN, every image is
re-processed end to end (decode -> perturbation -> deployed normalisation -> each member ->
calibrated ensemble) under: clean, JPEG q90/q75/q50, 0.5x bicubic downscale (which the
normalisation then upsamples back to 224), Gaussian blur sigma=1. Calibration and thresholds are
the deployed ones (fitted on clean VAL), so this measures real-world degradation.

Writes ``reports/robustness.json``, ``reports/figures/robustness.png`` and, in
``reports/metrics.json``, ``robustness`` (Contract 3: ``{perturbation: {accuracy, roc_auc, ece,
detection_rate, false_positive_rate, split, by_split}}``, headline numbers on test_unseen) and
``robustness_meta``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.calibrate import Ensemble  # noqa: E402
from truthlens.features import PERTURBATIONS, ClipEmbedder, pick_device  # noqa: E402
from truthlens.metrics import binary_metrics  # noqa: E402
from truthlens.models import LinearHead, load_cnn, stack_features  # noqa: E402
from truthlens.train import FeatureRows, get_logger, predict_cnn, read_manifest, update_json, write_json  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` rows: half real, half AI, each half spread evenly over its sources."""
    rng = np.random.default_rng(seed)
    parts = []
    for lab in (0, 1):
        g = df[df.label == lab]
        srcs = sorted(g.source.unique())
        per = max(1, (n // 2) // max(len(srcs), 1))
        for s in srcs:
            gs = g[g.source == s]
            parts.append(gs.iloc[rng.permutation(len(gs))[:per]])
    return pd.concat(parts).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--n-per-split", type=int, default=1000)
    ap.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=str(ROOT / "logs" / "eval_robustness.log"))
    args = ap.parse_args()

    log = get_logger(args.log)
    models_dir, reports = Path(args.models_dir), Path(args.reports_dir)
    config = json.loads((models_dir / "config.json").read_text())
    ens = Ensemble.from_dict(config["ensemble"])
    mc = config["members"]
    dev = pick_device(args.device)
    df = read_manifest(args.manifest, args.root)
    sub = pd.concat([stratified_sample(df[df.split == s], args.n_per_split, args.seed).assign(split=s)
                     for s in ("test", "test_unseen") if (df.split == s).any()]).reset_index(drop=True)
    log(f"robustness subsample: {sub.groupby(['split', 'label']).size().to_dict()}")

    clip = ClipEmbedder(mc["clip"]["backbone"], device=dev) if "clip" in mc else None
    clip_head = LinearHead.load(models_dir / mc["clip"]["head"]) if clip else None
    freq_head = LinearHead.load(models_dir / mc["freq"]["head"]) if "freq" in ens.members else None
    cnn = load_cnn(models_dir / mc["cnn"]["weights"], mc["cnn"]["arch"], dev) if "cnn" in mc else None

    paths, y = sub.abspath.tolist(), sub.label.to_numpy()
    results: dict[str, dict] = {s: {} for s in sub.split.unique()}
    for pert in args.perturbations.split(","):
        logits: dict[str, np.ndarray] = {}
        dl = DataLoader(FeatureRows(paths, clip.preprocess if clip else None, perturbation=pert, with_meta=False),
                        batch_size=args.batch, num_workers=args.workers)
        cl, fr = [], []
        for b in dl:
            if clip:
                cl.append(clip_head.logits(stack_features(clip.embed(b["clip"], amp=True), clip_head.feature_keys)))
            if freq_head:
                fr.append(freq_head.logits(b["spec"].numpy()))
        if clip:
            logits["clip"] = np.concatenate(cl)
        if freq_head:
            logits["freq"] = np.concatenate(fr)
        if cnn is not None:
            logits["cnn"] = predict_cnn(cnn, paths, dev, perturbation=pert, workers=args.workers,
                                        log=lambda m: None)["logits"]
        probs = {"ensemble": ens.prob({m: logits[m] for m in ens.members})}
        for m in ("clip", "cnn", "freq"):
            if m in logits and m in ens.member_calibrators:
                probs[m] = ens.member_calibrators[m](logits[m])
        for split in results:
            msk = (sub.split == split).to_numpy()
            for sid, p in probs.items():
                met = binary_metrics(y[msk], p[msk], config["decision"]["threshold"])
                results[split].setdefault(sid, {})[pert] = {k: met[k] for k in ("roc_auc", "accuracy", "ece",
                                                                                  "recall_tpr", "specificity_tnr")}
        log(f"{pert}: " + ", ".join(f"{s}/{sid} AUC {results[s][sid][pert]['roc_auc']:.3f}"
                                    for s in results for sid in probs))

    out = {"n_per_split": args.n_per_split, "perturbations": args.perturbations.split(","),
           "definitions": {k: list(v) for k, v in PERTURBATIONS.items()}, "results": results}
    write_json(reports / "robustness.json", out)
    head = "test_unseen" if "test_unseen" in results else next(iter(results))
    contract = {}
    for pert in out["perturbations"]:
        e = results[head]["ensemble"][pert]
        contract[pert] = {"split": head, "accuracy": e["accuracy"], "roc_auc": e["roc_auc"], "ece": e["ece"],
                          "detection_rate": e["recall_tpr"], "false_positive_rate": 1 - e["specificity_tnr"],
                          "by_split": {s: {sid: r[pert] for sid, r in results[s].items()} for s in results}}
    update_json(reports / "metrics.json", "robustness", contract)
    update_json(reports / "metrics.json", "robustness_meta",
                {k: out[k] for k in ("n_per_split", "perturbations", "definitions")})
    plot(results, reports / "figures" / "robustness.png")
    log("done")


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(results)
    fig, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 6.4), squeeze=False, sharex=True)
    for j, split in enumerate(splits):
        for row, metric in enumerate(("roc_auc", "accuracy")):
            ax = axes[row, j]
            for i, (sid, d) in enumerate(results[split].items()):
                perts = list(d)
                ax.plot(range(len(perts)), [d[p][metric] for p in perts], "-o", markersize=5,
                        color=PALETTE[i % len(PALETTE)], label=sid)
                ax.set_xticks(range(len(perts)), perts, rotation=30)
            ax.set(title=f"{split}: {metric}", ylim=(0.4, 1.01))
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
