#!/usr/bin/env python
"""TruthLens v2 data: remove EXPLICIT content and top the splits back up with clean images.

Runs after scripts/prepare_data.py (v1 manifest). Steps:
  1. scan every v1 manifest image + examples with 3 signals (scripts/prepare_data_nsfw.py)   [--stage scan_v1]
  2. download extra OpenFake row groups not used in v1 and normalize them identically          [--stage topup_norm]
  3. scan the top-up candidates (+ unused v1 GenImage images) with the SAME filters            [--stage scan_cand]
  4. thresholds + generator-level rule, delete flagged files, dedup candidates against
     everything kept, refill each (pool, split, label) to its v1 count, write manifest + scan csv [--stage build]

    .venv/bin/python scripts/prepare_data_filter.py --stage all
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_data as P  # noqa: E402
import prepare_data_nsfw as N  # noqa: E402

ROOT, META = P.ROOT, P.META
SCAN = META / "scan"
GEN_DROP_FRAC = 0.25      # drop a whole generator / real source if >= 25% of its images are flagged
GEN_DROP_MIN_N = 8        # ... and it has at least this many scanned images (avoid 1-of-3 noise)
SEED = 2026

TOPUP_GENIMAGE = ["data/train-00008-of-00014.parquet", "data/train-00009-of-00014.parquet"]
TOPUP_JOBS = [(f"core/validation-{i:05d}-of-00015.parquet", rg) for rg in range(4, 10) for i in range(15)] + \
             [(f"core/test-{i:05d}-of-00013.parquet", rg) for i in range(5, 13) for rg in range(4)]


def stage_scan_v1():
    SCAN.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(META / "manifest_v1_full.csv")
    d = f[["path", "label", "source", "split", "dataset", "pool", "prompt"]].copy()
    d["prompt"] = d["prompt"].fillna("")
    e = pd.read_csv(ROOT / "examples/labels.csv")
    ex = pd.DataFrame({"path": "examples/" + e["file"], "label": (e["label"] == "AI").astype(int),
                       "source": e["source"], "split": "examples", "dataset": "examples", "pool": "examples",
                       "prompt": ""})
    N.scan(pd.concat([d, ex], ignore_index=True)).to_csv(SCAN / "scan_v1_raw.csv", index=False)


def stage_topup_norm():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = P.RAW / "openfake_topup"
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(P._download_openfake_rg, s, rg, out) for s, rg in TOPUP_JOBS]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                P.log("topup download failed", e)
    P.stage_normalize(tiny_dir="__none__", of_dir="openfake_topup", out_csv="normalized_topup.csv")
    from huggingface_hub import hf_hub_download
    for f in TOPUP_GENIMAGE:
        hf_hub_download(P.TINY_GENIMAGE, f, repo_type="dataset", local_dir=P.RAW / "tiny_genimage_topup")
    P.stage_normalize(tiny_dir="tiny_genimage_topup", of_dir="__none__", out_csv="normalized_topup_gi.csv")


def _candidates() -> pd.DataFrame:
    v1 = set(pd.read_csv(META / "manifest_v1.csv")["path"])
    old = pd.read_csv(META / "normalized.csv", dtype={"dhash": str})
    old = old[old["path"].notna() & ~old["path"].isin(v1)]
    old = old[old["path"].map(lambda p: (ROOT / p).exists())]
    new = pd.concat([pd.read_csv(META / f, dtype={"dhash": str}) for f in
                     ("normalized_topup.csv", "normalized_topup_gi.csv") if (META / f).exists()], ignore_index=True)
    new = new[new["path"].notna() & ~new["path"].isin(v1)]
    c = pd.concat([old.assign(origin="v1_unused"), new.assign(origin="topup")], ignore_index=True)
    c = c.drop_duplicates("path")
    gi = c["dataset"] == P.TINY_GENIMAGE
    c["pool"] = np.where(gi, "genimage", np.where(c["orig_split"] == "test", "openfake_test", "openfake_val"))
    c["prompt"] = c["prompt"].fillna("")
    return c


# Max candidates scanned per (pool, role). The GPU was shared with training (~6 img/s for ViT-L/14), so we scan a
# random subset sized at ~1.5-2x the expected need instead of all ~14k candidates.
SCAN_CAPS = {("genimage", "real"): 650, ("genimage", "seen"): 1400, ("genimage", "unseen"): 120,
             ("openfake_test", "real"): 160, ("openfake_test", "unseen"): 560,
             ("openfake_val", "real"): 220, ("openfake_val", "seen"): 600, ("openfake_val", "unseen"): 160}


def stage_scan_cand():
    c = _candidates()
    c["role"] = _role(c)
    c = c[c["role"] != "drop"]
    c = pd.concat([g.sample(min(len(g), SCAN_CAPS.get(k, 0)), random_state=SEED)
                   for k, g in c.groupby(["pool", "role"])]).drop(columns=["role"])
    P.log("candidates to scan:", len(c), c.groupby(["pool", "label"]).size().to_dict())
    N.scan(c).to_csv(SCAN / "scan_candidates_raw.csv", index=False)


def _role(df: pd.DataFrame) -> pd.Series:
    fam = df["source"].str.startswith(tuple(P.canon(p) for p in P.UNSEEN_FAMILY_PREFIXES)) | \
        df["source"].str.contains("wan2")
    role = pd.Series("seen", index=df.index)
    role[df["label"] == 0] = "real"
    unseen = (df["label"] == 1) & (fam | (df["source"] == "midjourney_v5") | (df["pool"] == "openfake_test"))
    role[unseen] = "unseen"
    role[(df["label"] == 1) & df["source"].str.contains("tiny_random")] = "drop"
    role[(df["label"] == 1) & df["source"].isin(P.DROP_GENERATORS)] = "drop"
    return role


def stage_build(falcon_thr: float, sd_thr: float):
    rng = random.Random(SEED)
    v1 = pd.read_csv(META / "manifest_v1_full.csv", dtype={"dhash": str})
    s1 = pd.read_csv(SCAN / "scan_v1_raw.csv")
    sc = pd.read_csv(SCAN / "scan_candidates_raw.csv", dtype={"dhash": str})
    for d in (s1, sc):
        d["flag_falconsai"] = d["p_falconsai"] >= falcon_thr
        d["flag_sdsafety"] = d["sd_safety_score"] > sd_thr
        d["flag_prompt"] = N.keyword_flag(d["prompt"]) & (d["label"] == 1)
        d["flagged"] = d["flag_falconsai"] | d["flag_sdsafety"] | d["flag_prompt"]
    stats = {"falcon_thr": falcon_thr, "sd_thr": sd_thr}

    # ---- generator / real-source level rule over ALL scanned images (v1 manifest + candidates)
    allsc = pd.concat([s1[s1["split"] != "examples"][["source", "label", "flagged"]],
                       sc[["source", "label", "flagged"]]])
    g = allsc.groupby(["label", "source"])["flagged"].agg(["mean", "size"])
    bad = g[(g["mean"] >= GEN_DROP_FRAC) & (g["size"] >= GEN_DROP_MIN_N)]
    dropped_sources = {(int(l), s): {"flag_rate": round(float(r["mean"]), 3), "n_scanned": int(r["size"])}
                       for (l, s), r in bad.iterrows()}
    stats["dropped_sources"] = {f"{'AI' if l else 'real'}:{s}": v for (l, s), v in dropped_sources.items()}
    drop_src = {(l, s) for l, s in dropped_sources}
    P.log("dropped sources:", json.dumps(stats["dropped_sources"], indent=1))

    def src_dropped(df):
        return pd.Series([(int(l), s) in drop_src for l, s in zip(df["label"], df["source"])], index=df.index)

    s1m = s1[s1["split"] != "examples"].copy()
    s1m["removed_reason"] = np.where(s1m["flagged"], "flagged",
                                     np.where(src_dropped(s1m), "source_dropped", ""))
    s1m["removed"] = s1m["removed_reason"] != ""
    stats["removed_v1"] = s1m[s1m["removed"]].groupby(["split", "label", "dataset", "removed_reason"]).size() \
        .rename("n").reset_index().to_dict("records")

    keep = v1[v1["path"].isin(s1m.loc[~s1m["removed"], "path"])].copy()
    keep["origin"] = "v1"

    # ---- candidates: clean, not from dropped sources, role-consistent
    sc["removed_reason"] = np.where(sc["flagged"], "flagged", np.where(src_dropped(sc), "source_dropped", ""))
    cand = sc[sc["removed_reason"] == ""].copy()
    cand["role"] = _role(cand)
    cand = cand[cand["role"] != "drop"]

    # ---- dedup candidates against kept (md5 + dHash) and among themselves
    both = pd.concat([keep.assign(_k=1), cand.assign(_k=0)], ignore_index=True)
    cl = P.near_dup_clusters(np.array([int(h, 16) for h in both["dhash"]], dtype=np.uint64),
                             both["raw_md5"].tolist())
    both["cluster"] = cl
    kept_clusters = set(both.loc[both["_k"] == 1, "cluster"])
    mixed = both.groupby("cluster")["label"].transform("nunique") > 1
    c2 = both[(both["_k"] == 0) & ~both["cluster"].isin(kept_clusters) & ~mixed]
    c2 = c2.sort_values("path").drop_duplicates("cluster")
    stats["candidates_clean"] = int(len(cand))
    stats["candidates_after_dedup"] = int(len(c2))
    cand = c2.drop(columns=["_k"]).copy()
    cand_idx = {k: list(v.index) for k, v in cand.groupby(["pool", "role"])}
    for k in cand_idx:
        rng.shuffle(cand_idx[k])

    # ---- refill per (pool, split, label) to v1 counts
    target = v1.groupby(["pool", "split", "label"]).size()
    train_src = set(keep.loc[(keep["split"] == "train") & (keep["label"] == 1), "source"])
    added = []
    take_log = {}
    for (pool, split, label), tgt in target.items():
        have = ((keep["pool"] == pool) & (keep["split"] == split) & (keep["label"] == label)).sum()
        need = int(tgt - have)
        if need <= 0:
            continue
        role = "real" if label == 0 else ("unseen" if split == "test_unseen" else "seen")
        lst = cand_idx.get((pool, role), [])
        picked = []
        rest = []
        for i in lst:
            if len(picked) >= need:
                rest.append(i)
                continue
            if role == "seen" and split in ("val", "test") and cand.at[i, "source"] not in train_src:
                rest.append(i)  # test/val may only contain generators present in train
                continue
            picked.append(i)
        cand_idx[(pool, role)] = rest
        if role == "seen" and split == "train":
            train_src |= set(cand.loc[picked, "source"])
        a = cand.loc[picked].copy()
        a["split"] = split
        added.append(a)
        take_log[f"{pool}|{split}|{label}"] = {"needed": need, "added": len(picked)}
    stats["topup"] = take_log
    add = pd.concat(added) if added else cand.iloc[:0]
    add["origin"] = np.where(add["origin"] == "v1_unused", "v1_unused", "topup")
    final = pd.concat([keep, add], ignore_index=True)

    # ---- val/test may only contain generators that still have train images
    trn0 = set(final.loc[(final.label == 1) & (final.split == "train"), "source"])
    orphan = (final.label == 1) & final.split.isin(["val", "test"]) & ~final.source.isin(trn0)
    stats["valtest_orphan_generator_rows_dropped"] = int(orphan.sum())
    final = final[~orphan]
    # ---- re-balance: equal real/AI per (pool, split) (downsample the larger class, most frequent sources first)
    for (pool, split), grp in final.groupby(["pool", "split"]):
        nr, nf = (grp["label"] == 0).sum(), (grp["label"] == 1).sum()
        if nr == nf:
            continue
        big = 1 if nf > nr else 0
        gb = grp[grp["label"] == big]
        gb = gb.assign(r=gb.groupby("source").cumcount(ascending=False)).sort_values(
            ["r", "origin"], ascending=[False, False])
        final = final.drop(index=gb.index[: abs(nf - nr)])
        stats.setdefault("rebalance_dropped", {})[f"{pool}|{split}|{big}"] = int(abs(nf - nr))

    # ---- invariants
    tr = set(final.loc[(final.label == 1) & final.split.isin(["train", "val"]), "source"])
    un = set(final.loc[(final.label == 1) & (final.split == "test_unseen"), "source"])
    assert not (tr & un), tr & un
    te = set(final.loc[(final.label == 1) & (final.split == "test"), "source"])
    trn = set(final.loc[(final.label == 1) & (final.split == "train"), "source"])
    assert te <= trn, te - trn
    va = set(final.loc[(final.label == 1) & (final.split == "val"), "source"])
    assert va <= trn, va - trn
    assert final["path"].is_unique
    assert final["path"].map(lambda p: (ROOT / p).exists()).all()
    bal = final.groupby(["pool", "split", "label"]).size().unstack()
    assert (bal[0] == bal[1]).all(), bal
    fl = pd.concat([s1m.loc[s1m["flagged"], "path"], sc.loc[sc["flagged"], "path"]])
    assert not final["path"].isin(fl).any()
    assert not any((int(l), s) in drop_src for l, s in zip(final["label"], final["source"]))

    final["license"] = np.where(
        final["label"] == 0,
        final["source"].map(P.REAL_SOURCE_LICENSE).fillna(final["dataset"].map(P.LICENSES)),
        final["dataset"].map(P.LICENSES))
    order = {"train": 0, "val": 1, "test": 2, "test_unseen": 3}
    final = final.sort_values(["split", "label", "source", "path"],
                              key=lambda c: c.map(order) if c.name == "split" else c)
    final[["path", "label", "source", "split", "dataset", "license"]].to_csv(P.DATA / "manifest.csv", index=False)
    cols = ["path", "label", "source", "split", "dataset", "pool", "origin", "model_raw", "release_date", "prompt",
            "orig_format", "orig_w", "orig_h", "orig_mode", "orig_has_exif", "orig_bytes", "orig_jpeg_q0",
            "raw_md5", "out_md5", "dhash"]
    final[cols].to_csv(META / "manifest_full.csv", index=False)
    new_paths = final.loc[final["origin"] != "v1", "path"]
    new_paths.to_csv(META / "new_paths_v2.csv", index=False)
    stats["new_paths"] = int(len(new_paths))
    stats["final_counts"] = final.groupby(["split", "label"]).size().unstack().to_dict()
    stats["final_by_pool"] = {f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in
                              final.groupby(["pool", "split", "label"]).size().items()}
    stats["unseen_generators"] = sorted(un)
    stats["seen_generators"] = sorted(tr)

    # ---- scan report (scores only, no image content)
    s1m["removed"] = s1m["removed"] | ~s1m["path"].isin(final["path"])  # v1 images not kept for any reason
    sc["removed"] = ~sc["path"].isin(final["path"])
    sc["split"] = sc["path"].map(final.set_index("path")["split"]).fillna("candidate_unused")
    sc["removed_reason"] = np.where(sc["removed_reason"] != "", sc["removed_reason"],
                                    np.where(sc["removed"], "not_needed_or_dup", ""))
    s1m["removed_reason"] = np.where(s1m["removed_reason"] != "", s1m["removed_reason"],
                                     np.where(s1m["removed"], "rebalance", ""))
    rep_cols = ["path", "split", "label", "source", "dataset", "p_falconsai", "sd_safety_score", "p_clip_zeroshot",
                "flag_falconsai", "flag_sdsafety", "flag_prompt", "flagged", "removed", "removed_reason"]
    ex = s1[s1["split"] == "examples"].assign(removed=False, removed_reason="", dataset="examples")
    pd.concat([s1m.assign(scan_set="v1_manifest"), sc.assign(scan_set="topup_candidate"),
               ex.assign(scan_set="examples")])[rep_cols + ["scan_set"]] \
        .assign(removed=lambda d: np.where(d["removed"], "yes", "no")) \
        .to_csv(META / "nsfw_scan.csv", index=False)
    stats["examples"] = ex[["path", "p_falconsai", "sd_safety_score", "flag_prompt", "flagged"]].to_dict("records")
    (META / "filter_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    # ---- delete flagged / unused image files and raw caches
    keep_set = set(final["path"])
    n_rm = 0
    for p in (P.IMG).rglob("*.jpg"):
        if str(p.relative_to(ROOT)) not in keep_set:
            p.unlink()
            n_rm += 1
    stats["files_deleted"] = n_rm
    (META / "filter_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    P.log("deleted image files not in final manifest:", n_rm)
    P.log(json.dumps({k: stats[k] for k in ("final_counts", "topup", "candidates_after_dedup")}, indent=1,
                     default=str))


# --------------------------------------------------------------------------- v2.1: drop anime-style generators
ANIME_DROP = ["animagine_xl_3_0", "animagine_xl_3_1", "animagine_xl_4_0", "one_obsession_17_red_sdxl"]
TOPUP2_JOBS = [(f"core/validation-{i:05d}-of-00015.parquet", rg) for rg in (10, 11) for i in range(8)]


def _assert_invariants(final: pd.DataFrame, flagged_paths: set, dropped_sources: set):
    tr = set(final.loc[(final.label == 1) & final.split.isin(["train", "val"]), "source"])
    un = set(final.loc[(final.label == 1) & (final.split == "test_unseen"), "source"])
    assert not (tr & un), tr & un
    trn = set(final.loc[(final.label == 1) & (final.split == "train"), "source"])
    for s in ("val", "test"):
        assert set(final.loc[(final.label == 1) & (final.split == s), "source"]) <= trn
    assert final["path"].is_unique
    assert final["path"].map(lambda p: (ROOT / p).exists()).all()
    bal = final.groupby(["pool", "split", "label"]).size().unstack()
    assert (bal[0] == bal[1]).all(), bal
    assert not final["path"].isin(flagged_paths).any()
    assert not (final["label"].eq(1) & final["source"].isin(dropped_sources)).any()
    return tr, un


def stage_drop_anime():
    """Human decision (v2.1): drop the 4 remaining anime-style generators as a precaution (NSFW classifiers are
    least reliable on anime), refill the freed AI slots with clean SEEN OpenFake images from new row groups."""
    from concurrent.futures import ThreadPoolExecutor
    rng = random.Random(SEED + 1)
    stats = json.loads((META / "filter_stats.json").read_text())
    full = pd.read_csv(META / "manifest_full.csv", dtype={"dhash": str})
    scan_rep = pd.read_csv(META / "nsfw_scan.csv")
    prev_dropped = {k.split(":", 1)[1] for k in stats["dropped_sources"]} | {"tiny_random_sana"}
    dropped = prev_dropped | set(ANIME_DROP)

    rm = full[full["source"].isin(ANIME_DROP)]
    need = rm.groupby("split").size().to_dict()
    P.log("anime rows removed:", need)
    keep = full[~full["source"].isin(ANIME_DROP)].copy()

    # new OpenFake validation row groups -> normalize -> scan seen fakes with the same 3 filters
    out = P.RAW / "openfake_topup2"
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: P._download_openfake_rg(j[0], j[1], out), TOPUP2_JOBS))
    P.stage_normalize(tiny_dir="__none__", of_dir="openfake_topup2", out_csv="normalized_topup2.csv")
    c = pd.read_csv(META / "normalized_topup2.csv", dtype={"dhash": str})
    c = c[c["path"].notna() & (c["label"] == 1)].copy()
    c["pool"] = "openfake_val"
    c["prompt"] = c["prompt"].fillna("")
    c["role"] = _role(c)
    c = c[(c["role"] == "seen") & ~c["source"].isin(dropped) & ~c["path"].isin(set(keep["path"]))]
    sc = N.scan(c.drop(columns=["role"]))
    sc.to_csv(SCAN / "scan_candidates2_raw.csv", index=False)
    # generator-level rule over every scan so far (+ this one)
    allsc = pd.concat([scan_rep[scan_rep["scan_set"] != "examples"][["source", "label", "flagged"]],
                       sc[["source", "label", "flagged"]]])
    g = allsc[allsc["label"] == 1].groupby("source")["flagged"].agg(["mean", "size"])
    newbad = set(g[(g["mean"] >= GEN_DROP_FRAC) & (g["size"] >= GEN_DROP_MIN_N)].index) - dropped
    assert not keep["source"].isin(newbad).any(), newbad
    clean = sc[~sc["flagged"] & ~sc["source"].isin(newbad)].copy()
    # dedup vs everything kept
    both = pd.concat([keep.assign(_k=1), clean.assign(_k=0)], ignore_index=True)
    both["cluster"] = P.near_dup_clusters(np.array([int(h, 16) for h in both["dhash"]], dtype=np.uint64),
                                          both["raw_md5"].tolist())
    kc = set(both.loc[both["_k"] == 1, "cluster"])
    cand = both[(both["_k"] == 0) & ~both["cluster"].isin(kc)].sort_values("path").drop_duplicates("cluster")
    idx = list(cand.index)
    rng.shuffle(idx)
    train_src = set(keep.loc[(keep.label == 1) & (keep.split == "train"), "source"])
    added, log = [], {}
    for split in ["test", "val", "train"]:
        n = int(need.get(split, 0))
        pick = [i for i in idx if split == "train" or cand.at[i, "source"] in train_src][:n]
        idx = [i for i in idx if i not in set(pick)]
        a = cand.loc[pick].copy()
        a["split"], a["origin"] = split, "topup2"
        added.append(a)
        log[split] = {"removed_anime": n, "added": len(pick)}
    final = pd.concat([keep] + added, ignore_index=True)
    # keep balance: if AI refill fell short, drop matching reals in that openfake_val cell
    for split, grp in final[final["pool"] == "openfake_val"].groupby("split"):
        nr, nf = (grp.label == 0).sum(), (grp.label == 1).sum()
        if nr > nf:
            drop = rng.sample(list(grp[grp.label == 0].index), nr - nf)
            final = final.drop(index=drop)
            log[split]["reals_dropped_for_balance"] = int(nr - nf)
    flagged = set(scan_rep.loc[scan_rep["flagged"], "path"]) | set(sc.loc[sc["flagged"], "path"])
    tr, un = _assert_invariants(final, flagged, dropped | newbad)

    order = {"train": 0, "val": 1, "test": 2, "test_unseen": 3}
    final = final.sort_values(["split", "label", "source", "path"],
                              key=lambda col: col.map(order) if col.name == "split" else col)
    final["license"] = np.where(final["label"] == 0,
                                final["source"].map(P.REAL_SOURCE_LICENSE).fillna(final["dataset"].map(P.LICENSES)),
                                final["dataset"].map(P.LICENSES))
    final[["path", "label", "source", "split", "dataset", "license"]].to_csv(P.DATA / "manifest.csv", index=False)
    cols = [c_ for c_ in full.columns if c_ in final.columns]
    final[cols].to_csv(META / "manifest_full.csv", index=False)
    v1 = set(pd.read_csv(META / "manifest_v1.csv")["path"])
    final.loc[~final["path"].isin(v1), ["path"]].to_csv(META / "new_paths_v2.csv", index=False)

    # scan report: append new candidates; mark removed anime rows
    sc["split"] = sc["path"].map(final.set_index("path")["split"]).fillna("candidate_unused")
    sc["removed"] = np.where(sc["path"].isin(final["path"]), "no", "yes")
    sc["removed_reason"] = np.where(sc["flagged"], "flagged", np.where(sc["removed"] == "yes", "not_needed_or_dup", ""))
    sc["scan_set"] = "topup2_candidate"
    scan_rep.loc[scan_rep["path"].isin(rm["path"]), ["removed", "removed_reason"]] = ["yes", "anime_generator_dropped"]
    scan_rep.loc[~scan_rep["path"].isin(final["path"]) & (scan_rep["scan_set"] != "examples"), "removed"] = "yes"
    pd.concat([scan_rep, sc[scan_rep.columns.intersection(sc.columns)]], ignore_index=True) \
        .to_csv(META / "nsfw_scan.csv", index=False)

    stats["anime_drop_v2_1"] = {"generators": ANIME_DROP, "per_split": log,
                                "candidates_scanned": int(len(sc)), "candidates_flagged": int(sc["flagged"].sum()),
                                "new_generator_level_drops": sorted(newbad)}
    stats["final_counts"] = final.groupby(["split", "label"]).size().unstack().to_dict()
    stats["seen_generators"], stats["unseen_generators"] = sorted(tr), sorted(un)
    stats["new_paths"] = int((~final["path"].isin(v1)).sum())
    keep_set = set(final["path"])
    n_rm = 0
    for p in P.IMG.rglob("*.jpg"):
        if str(p.relative_to(ROOT)) not in keep_set:
            p.unlink()
            n_rm += 1
    import shutil
    shutil.rmtree(P.RAW, ignore_errors=True)
    stats["anime_drop_v2_1"]["files_deleted"] = n_rm
    (META / "filter_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    P.log(json.dumps(stats["anime_drop_v2_1"], indent=1), json.dumps(stats["final_counts"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "scan_v1", "topup_norm", "scan_cand", "build", "drop_anime"])
    ap.add_argument("--falcon-thr", type=float, default=N.FALCON_THR)
    ap.add_argument("--sd-thr", type=float, default=N.SD_THR)
    a = ap.parse_args()
    if a.stage in ("all", "scan_v1"):
        stage_scan_v1()
    if a.stage in ("all", "topup_norm"):
        stage_topup_norm()
    if a.stage in ("all", "scan_cand"):
        stage_scan_cand()
    if a.stage in ("all", "build"):
        stage_build(a.falcon_thr, a.sd_thr)
    if a.stage in ("all", "drop_anime"):
        stage_drop_anime()
