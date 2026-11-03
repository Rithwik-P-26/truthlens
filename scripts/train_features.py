"""Extract and cache features for every manifest row (one image decode feeds everything).

Writes to ``<cache-dir>/features/``:
* ``<backbone>.npz``: CLIP ``final`` embedding + intermediate CLS tokens (float16),
* ``spectral.npz``: spectral feature vector + (hf_ratio, peak_score, noise_level),
* ``metadata.npz``: file-container features for the shortcut-audit baseline.
Each file stores the manifest ``paths`` for alignment.

Usage::

    .venv/bin/python scripts/train_features.py --manifest data/manifest.csv \
        --backbones clip_vitl14,clip_vitb16 2>&1 | tee logs/features.log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truthlens.features import ClipEmbedder, pick_device  # noqa: E402
from truthlens.train import FeatureRows, get_logger, manifest_summary, read_manifest, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--root", default=str(ROOT), help="directory manifest paths are relative to")
    ap.add_argument("--cache-dir", default=str(ROOT / "models" / "cache"))
    ap.add_argument("--backbones", default="clip_vitl14,clip_vitb16")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true", help="fp32 CLIP on GPU (default: fp16 autocast)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N rows")
    ap.add_argument("--incremental", action="store_true",
                    help="reuse cached rows for paths still in the manifest, embed only missing paths, "
                         "drop rows whose paths left the manifest")
    ap.add_argument("--log", default=str(ROOT / "logs" / "features.log"))
    args = ap.parse_args()

    log = get_logger(args.log)
    df = read_manifest(args.manifest, args.root)
    if args.limit:
        df = df.iloc[: args.limit]
    log(f"manifest rows: {len(df)} | {manifest_summary(df)}")
    out_dir = Path(args.cache_dir) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(args.device)

    backbones = [k for k in args.backbones.split(",") if k]
    full = df
    old_cache: dict[str, dict[str, np.ndarray]] = {}
    if args.incremental:
        names = backbones + ["spectral", "metadata"]
        if all((out_dir / f"{n}.npz").exists() for n in names):
            for n in names:
                z = np.load(out_dir / f"{n}.npz", allow_pickle=False)
                old_cache[n] = {k: z[k] for k in z.files}
            cached = set(old_cache["spectral"]["paths"].tolist())
            for n in names:  # only paths present in every cache count as cached
                cached &= set(old_cache[n]["paths"].tolist())
            df = full[~full["path"].isin(cached)]
            log(f"incremental: {len(full) - len(df)} rows reused, {len(df)} to embed, "
                f"{len(cached - set(full['path']))} cached rows dropped (no longer in manifest)")
        else:
            log("incremental: no complete cache found, extracting everything")

    embedders = [ClipEmbedder(k, device=dev) for k in backbones]
    pre = embedders[0].preprocess if embedders else None
    for e in embedders[1:]:
        assert str(e.preprocess) == str(pre), "backbones must share preprocessing to share one decode"
    ds = FeatureRows(df["abspath"].tolist(), pre, with_meta=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    persistent_workers=False, prefetch_factor=4 if args.workers else None)

    clip_out: dict[str, dict[str, list[np.ndarray]]] = {e.key: {} for e in embedders}
    spec, stats, meta, ok = [], [], [], []
    t0, done = time.time(), 0
    for n_batches, batch in enumerate(dl, 1):
        for e in embedders:
            feats = e.embed(batch["clip"], amp=not args.no_amp)
            for k, v in feats.items():
                clip_out[e.key].setdefault(k, []).append(v.astype(np.float16))
        spec.append(batch["spec"].numpy())
        stats.append(batch["stats"].numpy())
        meta.append(batch["meta"].numpy())
        ok.append(batch["ok"].numpy())
        done += len(batch["ok"])
        if n_batches % 10 == 0 or done == len(df):
            rate = done / (time.time() - t0)
            log(f"{done}/{len(df)} images, {rate:.1f} img/s, ETA {(len(df) - done) / max(rate, 1e-9) / 60:.1f} min")

    new_paths = np.array(df["path"].tolist())
    fresh: dict[str, dict[str, np.ndarray]] = {e: {k: np.concatenate(v) for k, v in f.items()} for e, f in clip_out.items()}
    ok_arr = np.concatenate(ok).astype(bool) if ok else np.zeros(0, dtype=bool)
    if spec:
        fresh["spectral"] = {"spec": np.concatenate(spec), "stats": np.concatenate(stats), "ok": ok_arr}
        fresh["metadata"] = {"meta": np.concatenate(meta)}
    out_paths = np.array(full["path"].tolist())
    for name in backbones + ["spectral", "metadata"]:
        if old_cache:  # merge reused + fresh rows in manifest order
            pos_old = {q: i for i, q in enumerate(old_cache[name]["paths"].tolist())}
            pos_new = {q: i for i, q in enumerate(new_paths.tolist())}
            keys = [k for k in old_cache[name] if k != "paths"]
            arrays = {}
            for k in keys:
                ref = old_cache[name][k]
                arr = np.zeros((len(out_paths),) + ref.shape[1:], dtype=ref.dtype)
                for i, q in enumerate(out_paths.tolist()):
                    arr[i] = fresh[name][k][pos_new[q]] if q in pos_new else ref[pos_old[q]]
                arrays[k] = arr
        else:
            arrays = fresh[name]
        np.savez(out_dir / f"{name}.npz", paths=out_paths, **arrays)
    bad = df.loc[~ok_arr, "path"].tolist() if len(ok_arr) else []
    write_json(out_dir / "extraction_info.json", {
        "n": len(full), "n_embedded_this_run": len(df), "n_reused": len(full) - len(df),
        "backbones": backbones, "amp_fp16": not args.no_amp,
        "device": str(dev), "seconds": round(time.time() - t0, 1), "unreadable": bad})
    log(f"done in {(time.time() - t0) / 60:.1f} min; unreadable images: {len(bad)}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
