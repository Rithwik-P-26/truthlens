"""Train linear heads on cached features and select the deployed CLIP probe.

Heads (all: StandardScaler + L2 logistic regression, C by 5-fold CV on TRAIN):
* ``clip_<backbone>_final``: UnivFD-style probe on the final CLIP embedding (published baseline),
* ``clip_<backbone>_multi``: probe on final + intermediate CLS tokens (RINE-style),
* ``freq``: spectral features (cheap, explainable baseline/member),
* ``meta``: file-metadata shortcut audit (format, size, JPEG quality...). Never deployed:
  if it scores high, the dataset contains shortcuts and robustness results matter more.

Samples are weighted by source pool (``pool_weights``, alpha=0.5) so the small modern-generator
pool is not drowned by GenImage. Selection of the deployed CLIP probe uses leave-one-FAMILY-out
(LOFO) AUC on TRAIN/VAL only (mirrors how test_unseen was held out), so ``test_unseen`` stays untouched. Outputs: ``models/clip_head.npz``,
``models/freq_head.npz``, logits for every row in ``<cache>/preds/<name>.npz`` and
``reports/head_selection.json``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.features import META_FEATURES  # noqa: E402
from truthlens.models import stack_features  # noqa: E402
from truthlens.train import (_auc, fit_linear_head, generator_family, get_logger, load_cached, lofo_eval,  # noqa: E402
                             pool_weights, read_manifest, write_json)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--cache-dir", default=str(ROOT / "models" / "cache"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--backbones", default="clip_vitl14,clip_vitb16")
    ap.add_argument("--Cs", default="0.001,0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--pool-alpha", type=float, default=0.5)
    ap.add_argument("--deploy-clip", default="auto", help="'auto' (best LOGO AUC) or a head name")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=str(ROOT / "logs" / "train_heads.log"))
    args = ap.parse_args()

    log = get_logger(args.log)
    df = read_manifest(args.manifest, args.root)
    cache, models_dir = Path(args.cache_dir), Path(args.models_dir)
    (cache / "preds").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    Cs = tuple(float(c) for c in args.Cs.split(","))
    y = df["label"].to_numpy()
    fam = np.array([generator_family(s) for s in df["source"].astype(str)])
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    w_tr = pool_weights(y[tr], df["dataset"].to_numpy()[tr], args.pool_alpha)
    log(f"train families: {sorted(map(str, set(fam[tr & (y == 1)])))}")
    paths = np.array(df["path"].tolist())

    def run(name: str, X: np.ndarray, keys: list[str], meta: dict) -> dict:
        t0 = time.time()
        head, cv = fit_linear_head(X[tr], y[tr], keys, Cs=Cs, seed=args.seed, n_jobs=args.n_jobs, meta=meta,
                                   sample_weight=w_tr)
        logits = head.logits(X)
        lofo = lofo_eval(X[tr], y[tr], fam[tr], X[va], y[va], fam[va], cv["best_C"], args.seed, args.n_jobs, w_tr)
        rec = {"cv": cv, "val_auc": _auc(y[va], logits[va]), "lofo": lofo, "dim": int(X.shape[1]),
               "seconds": round(time.time() - t0, 1)}
        head.save(cache / "preds" / f"{name}_head.npz")
        np.savez(cache / "preds" / f"{name}.npz", paths=paths, logits=logits.astype(np.float32))
        log(f"{name}: dim {X.shape[1]} best C {cv['best_C']} cv AUC {cv['cv_auc_by_C'][str(cv['best_C'])]:.4f} "
            f"val AUC {rec['val_auc']:.4f} LOFO mean AUC {lofo['mean_auc']:.4f} (min {lofo['min_auc']:.4f}) "
            f"[{rec['seconds']}s]")
        return rec

    results: dict[str, dict] = {}
    for bb in [b for b in args.backbones.split(",") if b]:
        feats = load_cached(cache / "features" / f"{bb}.npz", df)
        blocks = sorted(k for k in feats if k.startswith("b"))
        for variant, keys in (("final", ["final"]), ("multi", ["final"] + blocks)):
            name = f"clip_{bb.removeprefix('clip_')}_{variant}"
            results[name] = run(name, stack_features(feats, keys), keys, {"backbone": bb, "variant": variant})
            results[name]["backbone"] = bb

    spec = load_cached(cache / "features" / "spectral.npz", df)
    results["freq"] = run("freq", spec["spec"], ["spec"], {"kind": "spectral"})
    meta = load_cached(cache / "features" / "metadata.npz", df)
    results["meta"] = run("meta", meta["meta"], META_FEATURES, {"kind": "metadata_shortcut_audit"})

    clip_names = [n for n in results if n.startswith("clip_")]
    if args.deploy_clip == "auto":
        chosen = max(clip_names, key=lambda n: (round(np.nan_to_num(results[n]["lofo"]["mean_auc"], nan=-1), 3),
                                               np.nan_to_num(results[n]["val_auc"], nan=-1)))
        rule = "highest leave-one-family-out mean AUC on TRAIN->VAL (rounded to 3 dp; tie-break: VAL AUC)"
    else:
        chosen = args.deploy_clip
        rule = "manual override (--deploy-clip)"
    shutil.copy(cache / "preds" / f"{chosen}_head.npz", models_dir / "clip_head.npz")
    shutil.copy(cache / "preds" / f"{chosen}.npz", cache / "preds" / "clip.npz")
    shutil.copy(cache / "preds" / "freq_head.npz", models_dir / "freq_head.npz")
    selection = {"deployed_clip_head": chosen, "deployed_backbone": results[chosen]["backbone"],
                 "selection_rule": rule, "heads": results}
    write_json(Path(args.reports_dir) / "head_selection.json", selection)
    log(f"deployed CLIP head: {chosen} ({rule})")


if __name__ == "__main__":
    main()
