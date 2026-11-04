#!/usr/bin/env python
"""TruthLens data preparation: download -> normalize -> dedup -> split -> manifest.

Reproducible end-to-end pipeline (deterministic given the same upstream files, seed=1337):

    .venv/bin/python scripts/prepare_data.py --stage all      # everything
    .venv/bin/python scripts/prepare_data.py --stage download # only fetch raw parquet
    .venv/bin/python scripts/prepare_data.py --stage normalize
    .venv/bin/python scripts/prepare_data.py --stage split    # dedup + split + manifest + figures

Sources (public, ungated Hugging Face datasets, no login needed):
  * TheKernel01/Tiny-GenImage  (CC BY-NC-SA 4.0) - ImageNet reals + GenImage generators
  * ComplexDataLab/OpenFake    (CC BY-NC 4.0)    - LAION/Pexels/DOCCI/ImageNet reals + 100s of modern generators
    (we read individual parquet row groups over HTTP so we never download a full 5 GB shard)

Normalization policy (applied identically to BOTH classes):
  RGB -> center square crop (side = shorter side) -> Lanczos downscale to 224x224 -> JPEG q95 (4:4:4), no EXIF/ICC.
  Images whose shorter side is < 256 are dropped, so every image is downscaled by >= 1.14x and nothing is upscaled. See docs/DATA_CARD.md for why.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
IMG = DATA / "images"
META = DATA / "meta"
FIG = ROOT / "reports" / "figures"
SEED = 1337
OUT_SIZE = 224   # every image ends up 224x224
MIN_SIDE = 256   # drop anything smaller: every kept image is DOWN-scaled by >= 1.14x (no upscaling, no un-resampled class)
JPEG_Q = 95

TINY_GENIMAGE = "TheKernel01/Tiny-GenImage"
OPENFAKE = "ComplexDataLab/OpenFake"
LICENSES = {
    TINY_GENIMAGE: "CC-BY-NC-SA-4.0 (GenImage); real images: ImageNet terms (non-commercial research)",
    OPENFAKE: "CC-BY-NC-4.0 (OpenFake); real images keep upstream terms (LAION/Pexels/DOCCI/ImageNet)",
}
# finer-grained license per real source (used in manifest + examples README)
REAL_SOURCE_LICENSE = {
    "imagenet": "ImageNet terms of access (non-commercial research); dataset card: CC-BY-NC-SA-4.0",
    "laion": "LAION-400M URLs (images keep original web copyright); OpenFake card: CC-BY-NC-4.0",
    "pexels": "Pexels License (free to use); OpenFake card: CC-BY-NC-4.0",
    "docci": "DOCCI (Google) CC-BY-4.0; OpenFake card: CC-BY-NC-4.0",
}

GENIMAGE_GEN = {  # Tiny-GenImage class index -> canonical source name
    0: "imagenet", 1: "adm", 2: "biggan", 3: "glide", 4: "midjourney_v5",
    5: "sd14", 6: "sd15", 7: "vqdm", 8: "wukong",
}

# Downloads (kept modest: ~13 GB raw, deleted after normalization unless --keep-raw)
TINY_FILES = [f"data/train-{i:05d}-of-00014.parquet" for i in range(8)] + \
             [f"data/validation-{i:05d}-of-00004.parquet" for i in range(4)]
OPENFAKE_PLAN = {  # split -> (shards, row groups per shard)
    "validation": ([f"core/validation-{i:05d}-of-00015.parquet" for i in range(15)], 4),
    "test": ([f"core/test-{i:05d}-of-00013.parquet" for i in range(5)], 4),
}
OPENFAKE_COLS = ["image", "label", "model", "type", "release_date", "prompt"]

# Generator families held out for test_unseen. Any OpenFake model whose name starts with one of
# these prefixes is NEVER used in train/val/test, even if it appears in the OpenFake validation split.
UNSEEN_FAMILY_PREFIXES = (
    "midjourney", "gpt-image", "z-image", "flux.2", "illustrious", "veo", "sora", "wan",
    "seedream", "nano-banana", "ernie", "recraft", "ideogram", "aurora", "frames", "halfmoon",
    "lumina", "dall", "imagen-4",
)
UNSEEN_CAPS = {"genimage": 800, "openfake_val": 450}  # max unseen fakes drawn from these pools
DROP_GENERATORS = {"biggan"}  # 128x128 native -> would need upscaling (see data card)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def canon(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9.]+", "_", s).strip("_")
    return s.replace(".", "_")


# --------------------------------------------------------------------------- download
def _download_tiny():
    from huggingface_hub import hf_hub_download
    out = RAW / "tiny_genimage"
    out.mkdir(parents=True, exist_ok=True)
    for f in TINY_FILES:
        p = hf_hub_download(TINY_GENIMAGE, f, repo_type="dataset", local_dir=out)
        log("tiny-genimage ok", p)


def _download_openfake_rg(shard: str, rg: int, out: Path):
    from huggingface_hub import HfFileSystem
    dst = out / f"{Path(shard).stem}-rg{rg:02d}.parquet"
    if dst.exists():
        return dst
    fs = HfFileSystem()
    for attempt in range(3):
        try:
            with fs.open(f"datasets/{OPENFAKE}/{shard}", block_size=16 << 20) as fh:
                t = pq.ParquetFile(fh).read_row_group(rg, columns=OPENFAKE_COLS)
            tmp = dst.with_suffix(".tmp")
            pq.write_table(t, tmp)
            tmp.rename(dst)
            return dst
        except Exception as e:  # noqa: BLE001
            log("retry", shard, rg, e)
            time.sleep(5)
    raise RuntimeError(f"failed {shard} rg{rg}")


def _download_openfake():
    out = RAW / "openfake"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, rg) for _, (shards, n) in OPENFAKE_PLAN.items() for s in shards for rg in range(n)]
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(_download_openfake_rg, s, rg, out): (s, rg) for s, rg in jobs}
        for i, f in enumerate(as_completed(futs)):
            try:
                log(f"openfake {i + 1}/{len(jobs)}", f.result().name)
            except Exception as e:  # noqa: BLE001
                log("openfake FAILED", futs[f], e)


def stage_download():
    with ThreadPoolExecutor(2) as ex:
        a = ex.submit(_download_tiny)
        b = ex.submit(_download_openfake)
        a.result(), b.result()


# --------------------------------------------------------------------------- normalize
def dhash64(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return format(int(sum(1 << i for i, b in enumerate(bits) if b)), "016x")  # hex string (uint64 > int64)


def normalize_one(raw_bytes: bytes):
    """Return (jpeg_bytes, info) or (None, info) if dropped."""
    im = Image.open(io.BytesIO(raw_bytes))
    info = {
        "orig_format": (im.format or "unknown").upper(),
        "orig_w": im.width, "orig_h": im.height, "orig_mode": im.mode,
        "orig_has_exif": bool(im.info.get("exif")) or bool(getattr(im, "getexif", lambda: {})()),
        "orig_bytes": len(raw_bytes),
        "raw_md5": hashlib.md5(raw_bytes).hexdigest(),
    }
    if im.format == "JPEG":
        try:
            q = im.quantization
            info["orig_jpeg_q0"] = int(np.mean(q[0])) if q else None
        except Exception:  # noqa: BLE001
            info["orig_jpeg_q0"] = None
    im = ImageOps.exif_transpose(im) if info["orig_has_exif"] else im
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    side = min(im.size)
    if side < MIN_SIDE:
        info["drop"] = "too_small"
        return None, info
    left, top = (im.width - side) // 2, (im.height - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.LANCZOS)  # always a downscale: side >= MIN_SIDE > OUT_SIZE
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_Q, subsampling=0, optimize=True)  # no exif/icc passed
    info["dhash"] = dhash64(im)
    return buf.getvalue(), info


def _iter_raw(tiny_dir: str = "tiny_genimage", of_dir: str = "openfake"):
    for p in sorted((RAW / tiny_dir).rglob("*.parquet")):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=256):
            d = b.to_pydict()
            for img, lab, gen in zip(d["image"], d["label"], d["generator"]):
                src = GENIMAGE_GEN[int(gen)]
                yield {"dataset": TINY_GENIMAGE, "orig_split": p.stem.split("-")[0], "label": int(lab),
                       "source": src, "model_raw": src, "prompt": "", "release_date": "",
                       "bytes": img["bytes"], "shard": p.name}
    for p in sorted((RAW / of_dir).glob("*.parquet")):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=128):
            d = b.to_pydict()
            for img, lab, model, rd, pr in zip(d["image"], d["label"], d["model"], d["release_date"], d["prompt"]):
                yield {"dataset": OPENFAKE, "orig_split": p.stem.split("-")[0], "label": int(lab == "fake"),
                       "source": canon(model), "model_raw": model, "prompt": (pr or "")[:300],
                       "release_date": rd or "", "bytes": img["bytes"], "shard": p.name}


def _safe_normalize(raw):
    try:
        return normalize_one(raw)
    except Exception as e:  # noqa: BLE001
        return None, {"drop": f"decode_error:{type(e).__name__}"}


def _flush(batch, rows, ex):
    results = ex.map(_safe_normalize, [b.pop("bytes") for b in batch], chunksize=8)
    for rec, (jpg, info) in zip(batch, results):
        rec.update(info)
        if jpg is not None:
            ds = "genimage" if rec["dataset"] == TINY_GENIMAGE else "openfake"
            uid = hashlib.sha1(f"{rec['dataset']}|{info['raw_md5']}|{rec['source']}".encode()).hexdigest()[:16]
            rel = f"data/images/{ds}/{'ai' if rec['label'] else 'real'}_{rec['source']}_{uid}.jpg"
            (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
            (ROOT / rel).write_bytes(jpg)
            rec["path"] = rel
            rec["out_md5"] = hashlib.md5(jpg).hexdigest()
        rows.append(rec)


def stage_normalize(tiny_dir: str = "tiny_genimage", of_dir: str = "openfake", out_csv: str = "normalized.csv"):
    from concurrent.futures import ProcessPoolExecutor
    META.mkdir(parents=True, exist_ok=True)
    rows, batch, n = [], [], 0
    with ProcessPoolExecutor(max(2, (os.cpu_count() or 4) - 2)) as ex:
        for rec in _iter_raw(tiny_dir, of_dir):
            if rec["label"] == 1 and rec["source"] in DROP_GENERATORS:
                rows.append({k: v for k, v in rec.items() if k != "bytes"} | {"drop": "generator_excluded"})
                continue
            batch.append(rec)
            n += 1
            if len(batch) >= 512:
                _flush(batch, rows, ex)
                batch = []
                log("normalized", n)
        if batch:
            _flush(batch, rows, ex)
    df = pd.DataFrame(rows)
    df.to_csv(META / out_csv, index=False)
    log("normalize done", len(df), "kept", df["path"].notna().sum())


# --------------------------------------------------------------------------- dedup + split
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def near_dup_clusters(hashes: np.ndarray, md5s: list[str], thr: int = 4) -> np.ndarray:
    """Cluster ids: exact raw md5 matches + dHash hamming <= thr (8x8-bit chunk LSH, exact for thr<=7)."""
    n = len(hashes)
    uf = UF(n)
    first = {}
    for i, m in enumerate(md5s):
        if m in first:
            uf.union(first[m], i)
        else:
            first[m] = i
    h = hashes.astype(np.uint64)
    for c in range(8):
        chunk = ((h >> np.uint64(8 * c)) & np.uint64(0xFF)).astype(np.int64)
        order = np.argsort(chunk, kind="stable")
        sc = chunk[order]
        bounds = np.flatnonzero(np.diff(sc)) + 1
        for grp in np.split(order, bounds):
            if len(grp) < 2:
                continue
            if len(grp) > 4000:  # degenerate bucket - compare in slices to bound memory
                pass
            hg = h[grp]
            for k in range(0, len(grp), 2000):
                a = hg[k:k + 2000]
                x = a[:, None] ^ hg[None, :]
                # popcount on uint64
                x = x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
                d = np.unpackbits(x, axis=2).sum(axis=2)
                ii, jj = np.nonzero(d <= thr)
                for a_i, b_j in zip(ii, jj):
                    if k + a_i < b_j:
                        uf.union(int(grp[k + a_i]), int(grp[b_j]))
    return np.array([uf.find(i) for i in range(n)])


def stage_split():
    rng = random.Random(SEED)
    df = pd.read_csv(META / "normalized.csv", dtype={"dhash": str})
    stats = {"normalized_rows": int(len(df))}
    kept = df[df["path"].notna()].copy().reset_index(drop=True)
    stats["dropped"] = df[df["path"].isna()]["drop"].fillna("other").value_counts().to_dict()
    kept["dhash_i"] = kept["dhash"].map(lambda s: int(s, 16))
    cl = near_dup_clusters(np.array(kept["dhash_i"].tolist(), dtype=np.uint64), kept["raw_md5"].tolist())
    kept["cluster"] = cl
    sizes = kept.groupby("cluster")["path"].transform("size")
    mixed = kept.groupby("cluster")["label"].transform("nunique") > 1
    stats["exact_dup_raw_md5"] = int(kept["raw_md5"].duplicated().sum())
    stats["near_dup_clusters_size_gt1"] = int((kept.groupby("cluster").size() > 1).sum())
    stats["images_in_dup_clusters"] = int((sizes > 1).sum())
    stats["mixed_label_clusters_dropped_images"] = int(mixed.sum())
    kept = kept[~mixed]
    # keep one representative per cluster (deterministic: lowest path)
    kept = kept.sort_values("path").drop_duplicates("cluster", keep="first").reset_index(drop=True)
    stats["after_dedup"] = int(len(kept))

    # ---------------- assign generator roles
    is_unseen_family = kept["source"].str.startswith(tuple(canon(p) for p in UNSEEN_FAMILY_PREFIXES))
    is_unseen_family |= kept["source"].str.contains("wan2")  # e.g. studio_ghibli_wan2_1 (Wan fine-tune)
    fake = kept["label"] == 1
    gi = kept["dataset"] == TINY_GENIMAGE
    of_test = (kept["dataset"] == OPENFAKE) & (kept["orig_split"] == "test")
    of_val = (kept["dataset"] == OPENFAKE) & (kept["orig_split"] == "validation")
    kept["role"] = "drop"
    junk = kept["source"].str.contains("tiny_random")  # random-weight test checkpoint (noise images), not a real generator
    stats["junk_generator_dropped"] = int(junk.sum())
    # Unseen fakes: GenImage Midjourney v5 + every OpenFake-test generator + unseen families anywhere
    kept.loc[fake & gi & (kept["source"] == "midjourney_v5"), "role"] = "unseen"
    kept.loc[fake & of_test, "role"] = "unseen"
    kept.loc[fake & of_val & is_unseen_family, "role"] = "unseen"
    kept.loc[fake & gi & (kept["source"] != "midjourney_v5"), "role"] = "seen"
    kept.loc[fake & of_val & ~is_unseen_family, "role"] = "seen"
    seen_names = set(kept.loc[kept["role"] == "seen", "source"])
    # safety: an unseen generator name must never coincide with a seen one
    clash = kept["role"].eq("unseen") & kept["source"].isin(seen_names)
    kept.loc[clash, "role"] = "drop"
    stats["unseen_dropped_name_clash"] = int(clash.sum())
    kept.loc[~fake, "role"] = "real"
    kept.loc[junk, "role"] = "drop"

    # group key so real/fake are balanced per upstream dataset (dataset style can't predict the label)
    kept["pool"] = np.where(gi, "genimage", np.where(of_test, "openfake_test", "openfake_val"))
    kept["split"] = ""

    def shuffled(idx):
        idx = list(idx)
        rng.shuffle(idx)
        return idx

    # ---- seen fakes: stratified per source; rare sources (<10 imgs) go to train only
    frac = {"train": 0.78, "val": 0.10, "test": 0.12}
    for src, g in kept[kept["role"] == "seen"].groupby("source"):
        idx = shuffled(g.index)
        if len(idx) < 10:
            kept.loc[idx, "split"] = "train"
            continue
        n_va, n_te = max(1, round(len(idx) * frac["val"])), max(1, round(len(idx) * frac["test"]))
        kept.loc[idx[:n_va], "split"] = "val"
        kept.loc[idx[n_va:n_va + n_te], "split"] = "test"
        kept.loc[idx[n_va + n_te:], "split"] = "train"
    # ---- unseen fakes: all OpenFake-test generators; capped GenImage-MJv5 and OpenFake-val unseen families
    # (caps keep test_unseen ~5k and leave enough real images for training)
    for pool, g in kept[kept["role"] == "unseen"].groupby("pool"):
        idx = shuffled(g.index)[: UNSEEN_CAPS.get(pool, len(g))]
        kept.loc[idx, "split"] = "test_unseen"

    # ---- reals: per pool, match the fake count of each split within the pool
    fake_counts = kept[fake & (kept["split"] != "")].groupby(["pool", "split"]).size()
    for pool, g in kept[kept["role"] == "real"].groupby("pool"):
        idx = shuffled(g.index)
        need = {s: int(fake_counts.get((pool, s), 0)) for s in ["test_unseen", "test", "val", "train"]}
        pos = 0
        for s in ["test_unseen", "test", "val", "train"]:
            take = idx[pos:pos + need[s]]
            kept.loc[take, "split"] = s
            pos += len(take)
        stats[f"real_pool_{pool}_unused"] = len(idx) - pos
    # if a pool lacks reals for its fakes, downsample those fakes (keep balance per pool/split)
    for (pool, s), g in kept[kept["split"] != ""].groupby(["pool", "split"]):
        nr, nf = (g["label"] == 0).sum(), (g["label"] == 1).sum()
        if nf > nr:
            # drop surplus fakes, removing from the most frequent sources first (keeps rare gens)
            fk = g[g["label"] == 1]
            order = fk.assign(r=fk.groupby("source").cumcount(ascending=False)).sort_values("r", ascending=False)
            drop_idx = list(order.index[: nf - nr])
            kept.loc[drop_idx, "split"] = ""
            stats[f"fakes_downsampled_{pool}_{s}"] = int(nf - nr)

    final = kept[kept["split"] != ""].copy()
    final["license"] = np.where(
        final["label"] == 0,
        final["source"].map(REAL_SOURCE_LICENSE).fillna(final["dataset"].map(LICENSES)),
        final["dataset"].map(LICENSES),
    )
    # verify invariants
    tr_gens = set(final.loc[(final.label == 1) & final.split.isin(["train", "val"]), "source"])
    un_gens = set(final.loc[(final.label == 1) & (final.split == "test_unseen"), "source"])
    assert not (tr_gens & un_gens), tr_gens & un_gens
    te_gens = set(final.loc[(final.label == 1) & (final.split == "test"), "source"])
    tr_only = set(final.loc[(final.label == 1) & (final.split == "train"), "source"])
    assert te_gens <= tr_only, te_gens - tr_only
    assert final["cluster"].is_unique
    assert final["path"].map(lambda p: (ROOT / p).exists()).all()

    order = {"train": 0, "val": 1, "test": 2, "test_unseen": 3}
    final = final.sort_values(["split", "label", "source", "path"], key=lambda c: c.map(order) if c.name == "split" else c)
    final[["path", "label", "source", "split", "dataset", "license"]].to_csv(DATA / "manifest.csv", index=False)
    extra_cols = ["path", "label", "source", "split", "dataset", "pool", "model_raw", "release_date", "prompt",
                  "orig_format", "orig_w", "orig_h", "orig_mode", "orig_has_exif", "orig_bytes", "orig_jpeg_q0",
                  "raw_md5", "out_md5", "dhash"]
    final[extra_cols].to_csv(META / "manifest_full.csv", index=False)
    stats["final"] = int(len(final))
    stats["counts"] = final.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
    stats["seen_generators"] = sorted(tr_gens)
    stats["unseen_generators"] = sorted(un_gens)
    (META / "split_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    log(json.dumps({k: v for k, v in stats.items() if k not in ("seen_generators",)}, indent=1, default=str))
    # images not in the manifest (dups, surplus, excluded) stay on disk but are simply not referenced


# --------------------------------------------------------------------------- audit + figures
def stage_audit():
    """Bias checks before/after normalization + figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = pd.read_csv(META / "normalized.csv", dtype={"dhash": str})
    if (META / "normalized_topup.csv").exists():  # v2 content-filter top-up (scripts/prepare_data_filter.py)
        norm = pd.concat([norm, pd.read_csv(META / "normalized_topup.csv", dtype={"dhash": str})], ignore_index=True)
    # "before" stats describe only images that ended up in the final manifest
    final_paths = set(pd.read_csv(DATA / "manifest.csv")["path"])
    norm = norm[norm["path"].isin(final_paths)]
    full = pd.read_csv(META / "manifest_full.csv", dtype={"dhash": str})
    rep = {}
    # --- before normalization (all decoded rows)
    n = norm[norm["orig_w"].notna()].copy()
    n["cls"] = np.where(n["label"] == 1, "AI", "real")
    n["aspect"] = n["orig_w"] / n["orig_h"]
    n["minside"] = n[["orig_w", "orig_h"]].min(axis=1)
    by = n.groupby(["dataset", "cls"])
    rep["before_format"] = n.groupby(["dataset", "cls", "orig_format"]).size().to_dict()
    rep["before_minside_median"] = by["minside"].median().to_dict()
    rep["before_square_frac"] = by["aspect"].apply(lambda a: float((a.round(3) == 1).mean())).to_dict()
    rep["before_bytes_median"] = by["orig_bytes"].median().to_dict()
    rep["before_exif_frac"] = by["orig_has_exif"].mean().to_dict()
    rep["before_mode"] = n.groupby(["cls", "orig_mode"]).size().to_dict()
    per_src = n.groupby(["cls", "source"]).agg(n=("source", "size"), fmt=("orig_format", lambda s: s.mode().iat[0]),
                                                   minside_med=("minside", "median"),
                                                   square=("aspect", lambda a: float((a.round(3) == 1).mean())))
    per_src.to_csv(META / "per_source_before.csv")
    # --- after normalization: file size / sharpness / color stats per class on final images
    sample = full.sample(frac=1.0, random_state=SEED).groupby(["split", "label"]).head(400)
    feats = []
    for _, r in sample.iterrows():
        p = ROOT / r["path"]
        im = Image.open(p)
        a = np.asarray(im, dtype=np.float32)
        g = a.mean(axis=2)
        lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean()
        feats.append({"split": r["split"], "label": r["label"], "w": im.width, "h": im.height, "mode": im.mode,
                      "fmt": im.format, "kb": p.stat().st_size / 1024, "lap": lap, "sat": a.std(axis=2).mean(),
                      "bright": g.mean(), "exif": bool(im.info.get("exif"))})
    f = pd.DataFrame(feats)
    rep["after_sizes"] = f.groupby("label")[["w", "h"]].agg(["min", "max"]).to_dict()
    rep["after_format_mode"] = f.groupby(["label", "fmt", "mode"]).size().to_dict()
    rep["after_exif_any"] = bool(f["exif"].any())
    rep["after_stats_by_label"] = f.groupby("label")[["kb", "lap", "sat", "bright"]].median().round(2).to_dict()
    # single-feature AUC: how well does a trivial statistic separate the classes on the final data?
    def auc(x, y):
        from scipy.stats import rankdata  # type: ignore
        r = rankdata(x)
        n1 = (y == 1).sum(); n0 = (y == 0).sum()
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    try:
        rep["trivial_feature_auc"] = {c: round(max(a := auc(f[c].values, f["label"].values), 1 - a), 3)
                                      for c in ["kb", "lap", "sat", "bright"]}
    except ImportError:
        def auc_np(x, y):
            o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
            n1 = (y == 1).sum(); n0 = (y == 0).sum()
            return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        rep["trivial_feature_auc"] = {c: round(max(a := auc_np(f[c].values, f["label"].values), 1 - a), 3)
                                      for c in ["kb", "lap", "sat", "bright"]}
    (META / "audit.json").write_text(json.dumps({k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                                                 for k, v in rep.items()}, indent=2, default=str))
    log("audit", json.dumps(rep["trivial_feature_auc"]), rep["after_stats_by_label"])

    # --- figures
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # fig 1: counts per source x split (stacked horizontal bars)
    ct = full.groupby(["source", "split"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=[c for c in ["train", "val", "test", "test_unseen"] if c in ct.columns])
    lab = full.groupby("source")["label"].first()
    tot = ct.sum(axis=1)
    big = tot[tot >= 40].index
    other_f = ct.loc[[s for s in ct.index if s not in big and lab[s] == 1]].sum()
    ct2 = ct.loc[big].copy()
    if other_f.sum():
        ct2.loc["other AI (<40 imgs each)"] = other_f
    ct2 = ct2.loc[ct2.sum(axis=1).sort_values().index]
    colors = {"train": "#4C72B0", "val": "#8DA0CB", "test": "#55A868", "test_unseen": "#C44E52"}
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(ct2))))
    left = np.zeros(len(ct2))
    for c in ct2.columns:
        ax.barh(ct2.index, ct2[c], left=left, color=colors[c], label=c, height=0.75)
        left += ct2[c].values
    names = [("real: " if (s in lab and lab[s] == 0) else "AI: ") + s if not s.startswith("other") else s for s in ct2.index]
    ax.set_yticks(range(len(ct2)), names)
    ax.set_xlabel("images")
    ax.set_title("TruthLens dataset: images per source and split")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "data_counts_per_source_split.png", dpi=150)
    plt.close(fig)
    # fig 2: original resolution (shorter side) + format per class, before normalization
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.logspace(np.log10(64), np.log10(4096), 40)
    for cls, col in [("real", "#4C72B0"), ("AI", "#C44E52")]:
        axes[0].hist(n.loc[n.cls == cls, "minside"], bins=bins, alpha=0.6, label=cls, color=col)
    axes[0].axvline(MIN_SIDE, color="k", ls="--", lw=1)
    axes[0].text(MIN_SIDE * 1.05, axes[0].get_ylim()[1] * 0.9, f"min kept {MIN_SIDE}px", fontsize=8)
    axes[0].set_xscale("log"); axes[0].set_xlabel("original shorter side (px, log)"); axes[0].set_ylabel("images")
    axes[0].set_title("Before normalization: resolution per class"); axes[0].legend(frameon=False)
    fmt = n.groupby(["cls", "orig_format"]).size().unstack(fill_value=0)
    fmt = fmt.div(fmt.sum(axis=1), axis=0)
    bottom = np.zeros(len(fmt))
    pal = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]
    for i, c in enumerate(fmt.columns):
        axes[1].bar(fmt.index, fmt[c], bottom=bottom, label=c, color=pal[i % len(pal)])
        bottom += fmt[c].values
    axes[1].set_ylabel("fraction"); axes[1].set_title("Before normalization: file format per class")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"After normalization every image is {OUT_SIZE}x{OUT_SIZE} RGB JPEG q95 without metadata", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "data_resolution_format_before_norm.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "download", "normalize", "split", "audit"])
    ap.add_argument("--keep-raw", action="store_true", help="keep raw parquet after normalization")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.stage in ("all", "download"):
        stage_download()
    if a.stage in ("all", "normalize"):
        stage_normalize()
        if a.stage == "all" and not a.keep_raw:
            import shutil
            shutil.rmtree(RAW, ignore_errors=True)
    if a.stage in ("all", "split"):
        (DATA / "READY").unlink(missing_ok=True)
        stage_split()
    if a.stage in ("all", "split", "audit"):
        stage_audit()
    log("done")


if __name__ == "__main__":
    sys.exit(main())
