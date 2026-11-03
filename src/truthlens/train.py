"""Training utilities: manifest handling, datasets, augmentations, CNN fine-tuning and
linear-head fitting (with cross-validation and leave-one-generator-out evaluation).

Data-hygiene rules enforced here:
* the manifest is validated (columns, labels, splits, duplicate paths) and
  ``test_unseen`` AI sources must never appear in train/val (raises otherwise);
* hyper-parameters are chosen by cross-validation on TRAIN or on VAL, never on test;
* generalisation to new generators is estimated *without* touching test_unseen via
  leave-one-generator-out (LOGO): train without generator g, score g's VAL images vs VAL reals.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .features import (apply_perturbation, jpeg_roundtrip, load_image, metadata_features, normalize_image,
                       pick_device, resolve_path, set_seed, spectral_features)
from .models import CnnDetector, LinearHead, save_cnn

MANIFEST_COLUMNS = ["path", "label", "source", "split", "dataset", "license"]
SPLITS = ("train", "val", "test", "test_unseen")

# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def read_manifest(path: str | Path, root: str | Path | None = None) -> pd.DataFrame:
    """Load and validate a Contract-1 manifest; adds an ``abspath`` column."""
    path = Path(path)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS[:4] if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {path} lacks columns {missing}")
    df["label"] = df["label"].astype(int)
    if not set(df["label"]).issubset({0, 1}):
        raise ValueError("labels must be 0 (real) or 1 (AI)")
    bad = set(df["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"unknown splits {bad}")
    if df["path"].duplicated().any():
        raise ValueError(f"{int(df['path'].duplicated().sum())} duplicate paths in manifest")
    df["source"] = df["source"].astype(str)
    seen = set(df[df["split"].isin(["train", "val"]) & (df["label"] == 1)]["source"])
    unseen = set(df[(df["split"] == "test_unseen") & (df["label"] == 1)]["source"])
    if seen & unseen:
        raise ValueError(f"test_unseen generators leak into train/val: {sorted(seen & unseen)}")
    df["abspath"] = [str(resolve_path(p, root)) for p in df["path"]]
    return df.reset_index(drop=True)


def manifest_summary(df: pd.DataFrame) -> dict:
    """Counts per split/label/source for logs and reports."""
    out = {}
    for split, g in df.groupby("split"):
        out[split] = {"n": int(len(g)), "n_real": int((g.label == 0).sum()), "n_ai": int((g.label == 1).sum()),
                      "sources": {s: int(n) for s, n in g["source"].value_counts().items()}}
    return out


def pool_weights(labels: np.ndarray, pools: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Per-sample weights: within each class, pool p gets total weight proportional to n_p**alpha.

    alpha=1 keeps the natural mix (85% GenImage), alpha=0 gives every pool an equal share;
    alpha=0.5 (pre-registered, not tuned) lifts the small modern-generator pool from ~15% to ~30%
    without letting ~1.2k images dominate. Both classes get the same total weight. Mean weight is 1.
    """
    w = np.zeros(len(labels), dtype=np.float64)
    for lab in np.unique(labels):
        m = labels == lab
        ps, counts = np.unique(pools[m], return_counts=True)
        share = counts.astype(np.float64) ** alpha
        share /= share.sum()
        for pool, n, sh in zip(ps, counts, share):
            w[m & (pools == pool)] = sh / n
    return w / w.mean()


#: generator family rules (first match wins) used for leave-one-family-out evaluation
FAMILY_RULES: list[tuple[str, str]] = [
    (r"(sd_?xl|stable_diffusion_xl|_xl_|xl_v|pony|animagine|novaanime|one_obsession)", "sdxl"),
    (r"^(sd15|sd_1_5|stable_diffusion_v1|realistic_vision)", "sd1"),
    (r"^(sd_2|sd_turbo)", "sd2"),
    (r"^sd_3", "sd3"),
    (r"^(flux|chroma|mystic)", "flux1"),
    (r"^anima", "anima"),
    (r"^imagen", "imagen"),
    (r"^playground", "playground"),
]


def generator_family(source: str) -> str:
    """Map a generator name to its family (e.g. all SDXL fine-tunes -> 'sdxl')."""
    s = source.lower()
    for pat, fam in FAMILY_RULES:
        if re.search(pat, s):
            return fam
    return s.split("_")[0]


# --------------------------------------------------------------------------------------
# Augmentations / datasets
# --------------------------------------------------------------------------------------


@dataclass
class AugConfig:
    """Robustness augmentations at the native 224 px training resolution (both classes)."""

    size: int = 224
    p_rescale: float = 0.3  # downscale then back to `size` (resolution loss)
    scale_range: tuple[float, float] = (0.5, 0.95)
    p_blur: float = 0.2
    blur_sigma: tuple[float, float] = (0.1, 1.5)
    p_jpeg: float = 0.5
    jpeg_quality: tuple[int, int] = (60, 100)
    p_flip: float = 0.5


_RESAMPLE = (Image.BILINEAR, Image.BICUBIC, Image.LANCZOS)


class TrainAugment:
    """PIL (already 224x224, normalised) -> ``[3, 224, 224]`` tensor in [0, 1]."""

    def __init__(self, cfg: AugConfig) -> None:
        self.cfg = cfg

    def __call__(self, img: Image.Image) -> torch.Tensor:
        c = self.cfg
        if img.size != (c.size, c.size):  # defensive: normalise like deployment
            img = normalize_image(img)[0]
        if random.random() < c.p_rescale:
            s = random.uniform(*c.scale_range)
            small = img.resize((max(8, round(c.size * s)),) * 2, random.choice(_RESAMPLE))
            img = small.resize((c.size, c.size), random.choice(_RESAMPLE))
        if random.random() < c.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*c.blur_sigma)))
        if random.random() < c.p_jpeg:
            img = jpeg_roundtrip(img, random.randint(*c.jpeg_quality))
        if random.random() < c.p_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return _to_tensor(img)


class EvalTransform:
    """Deployment view: the normalised 224x224 image as a tensor (normalises if it is not)."""

    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != (self.size, self.size):
            img = normalize_image(img)[0]
        return _to_tensor(img)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


class ImageRows(Dataset):
    """``(transform(image), label, index, ok)`` for a list of image paths."""

    def __init__(self, paths: list[str], labels: np.ndarray, transform: Callable,
                 perturbation: str | None = None) -> None:
        self.paths, self.labels, self.transform, self.perturbation = list(paths), np.asarray(labels), transform, perturbation

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        ok = True
        try:
            img = load_image(self.paths[i])
        except Exception:  # noqa: BLE001 - unreadable file: keep the batch alive, flag it
            img, ok = Image.new("RGB", (256, 256), (128, 128, 128)), False
        if self.perturbation:
            img = apply_perturbation(img, self.perturbation)
        return self.transform(img), int(self.labels[i]), i, ok


class FeatureRows(Dataset):
    """Everything the cached-feature pipeline needs from one decode of an image.

    Returns a dict: ``clip`` (preprocessed tensor), ``spec`` (spectral vector),
    ``stats`` (hf_ratio, peak_score, noise_level), ``meta`` (file metadata), ``ok``.
    """

    def __init__(self, paths: list[str], clip_preprocess: Callable | None, perturbation: str | None = None,
                 with_meta: bool = True) -> None:
        self.paths, self.pre, self.perturbation, self.with_meta = list(paths), clip_preprocess, perturbation, with_meta

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        ok = True
        try:
            img = load_image(self.paths[i])
        except Exception:  # noqa: BLE001
            img, ok = Image.new("RGB", (256, 256), (128, 128, 128)), False
        if self.perturbation:  # degrade the dataset image, then re-apply the deployed normalisation
            img = normalize_image(apply_perturbation(img, self.perturbation))[0]
        # clean dataset files already ARE normalize(original upload): no second JPEG pass
        vec, st = spectral_features(img)
        out = {"spec": torch.from_numpy(vec),
               "stats": torch.tensor([st.hf_ratio, st.peak_score, st.noise_level], dtype=torch.float32),
               "ok": ok, "index": i}
        if self.pre is not None:
            out["clip"] = self.pre(img)
        if self.with_meta:
            try:
                out["meta"] = torch.from_numpy(metadata_features(self.paths[i]))
            except Exception:  # noqa: BLE001
                out["meta"] = torch.zeros(12)
        return out


# --------------------------------------------------------------------------------------
# CNN training
# --------------------------------------------------------------------------------------


@dataclass
class CnnTrainConfig:
    arch: str = "efficientnet_b0"
    epochs: int = 6
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    workers: int = 8
    amp: str = "bf16"  # none | fp16 | bf16
    seed: int = 0
    max_steps: int = 0  # >0 caps optimisation steps (smoke tests)
    pool_alpha: float = 0.5  # see pool_weights
    pretrained: bool = True
    aug: AugConfig = field(default_factory=AugConfig)


def _amp_ctx(device: torch.device, amp: str):
    if amp == "none" or device.type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


def train_cnn(cfg: CnnTrainConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, out_path: Path,
              device: str | None = None, log: Callable[[str], None] = print) -> dict:
    """Fine-tune :class:`CnnDetector` at 224 px; keeps the checkpoint with the best VAL AUC."""
    from sklearn.metrics import roc_auc_score

    set_seed(cfg.seed)
    dev = pick_device(device)
    model = CnnDetector(cfg.arch, pretrained=cfg.pretrained).to(dev)
    labels = train_df["label"].to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(pool_weights(labels, train_df["dataset"].to_numpy(), cfg.pool_alpha)),
                                    num_samples=len(train_df), replacement=True,
                                    generator=torch.Generator().manual_seed(cfg.seed))
    dl = DataLoader(ImageRows(train_df["abspath"].tolist(), labels, TrainAugment(cfg.aug)),
                    batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.workers, drop_last=True,
                    persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers else None)
    vdl = DataLoader(ImageRows(val_df["abspath"].tolist(), val_df["label"].to_numpy(), EvalTransform()),
                     batch_size=64, shuffle=False, num_workers=cfg.workers)
    steps_per_epoch = len(dl)
    total = cfg.max_steps or cfg.epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    lossf = torch.nn.BCEWithLogitsLoss()
    history, best, step, t0 = [], -1.0, 0, time.time()
    epochs = math.ceil(total / steps_per_epoch)
    for epoch in range(epochs):
        model.train()
        run_loss, n_seen, t_epoch = 0.0, 0, time.time()
        for x, y, _, ok in dl:
            x, y = x.to(dev, non_blocking=True), y.float().to(dev)
            with _amp_ctx(dev, cfg.amp):
                logits = model(x)
            loss = lossf(logits.float()[ok.to(dev)], y[ok.to(dev)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += float(loss.detach()) * len(y)
            n_seen += len(y)
            if step % 50 == 0:
                log(f"epoch {epoch} step {step}/{total} loss {run_loss / n_seen:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {n_seen / (time.time() - t_epoch + 1e-9):.0f} img/s")
            if step >= total:
                break
        model.eval()
        vp, vy = [], []
        with torch.inference_mode():
            for x, y, _, ok in vdl:
                with _amp_ctx(dev, cfg.amp):
                    vp.append(model(x.to(dev)).float().cpu())
                vy.append(y)
        vp_t, vy_t = torch.cat(vp), torch.cat(vy).float()
        vloss = float(torch.nn.functional.binary_cross_entropy_with_logits(vp_t, vy_t))
        try:
            vauc = float(roc_auc_score(vy_t.numpy(), vp_t.numpy()))
        except ValueError:
            vauc = float("nan")
        vacc = float(((vp_t > 0).float() == vy_t).float().mean())
        rec = {"epoch": epoch, "step": step, "train_loss": run_loss / max(n_seen, 1), "val_loss": vloss,
               "val_auc": vauc, "val_acc": vacc, "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        log(json.dumps(rec))
        score = vauc if not math.isnan(vauc) else -vloss
        if score > best:
            best = score
            save_cnn(model, out_path)
        if step >= total:
            break
    return {"config": _jsonable(asdict(cfg)), "history": history, "steps_per_epoch": steps_per_epoch}


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))


# --------------------------------------------------------------------------------------
# CNN prediction with the deployment preprocessing (native resolution, pixel budget)
# --------------------------------------------------------------------------------------


class _CnnEvalRows(Dataset):
    """Deployment view of each image: the dataset file as is (it already is the normalised
    upload), or perturbation followed by the deployed normalisation."""

    def __init__(self, paths: list[str], perturbation: str | None) -> None:
        self.paths, self.perturbation = paths, perturbation

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            img = load_image(self.paths[i])
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.perturbation:
            img = apply_perturbation(img, self.perturbation)
        if self.perturbation or img.size != (224, 224):
            img = normalize_image(img)[0]
        return _to_tensor(img), i


def predict_cnn(model: CnnDetector, paths: list[str], device: str | torch.device,
                perturbation: str | None = None, batch_size: int = 64, workers: int = 6,
                log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """CNN logits + CAM statistics (per-image 99th percentile and max of the CAM) at 224 px."""
    dev = torch.device(device)
    model = model.eval().to(dev)
    dl = DataLoader(_CnnEvalRows(paths, perturbation), batch_size=batch_size, num_workers=workers)
    logits = np.zeros(len(paths), dtype=np.float32)
    cam_p99 = np.zeros(len(paths), dtype=np.float32)
    cam_max = np.zeros(len(paths), dtype=np.float32)
    done, t0 = 0, time.time()
    with torch.inference_mode():
        for x, idx in dl:
            lg, cam = model.forward_with_cam(x.to(dev))
            cam = cam.float().flatten(1).cpu()
            ii = idx.numpy()
            logits[ii] = lg.float().cpu().numpy()
            cam_p99[ii] = torch.quantile(cam.clamp_min(0), 0.99, dim=1).numpy()
            cam_max[ii] = cam.max(1).values.numpy()
            done += len(ii)
            if done % (batch_size * 20) < batch_size or done == len(paths):
                log(f"cnn predict {done}/{len(paths)} ({done / (time.time() - t0):.1f} img/s)")
    return {"logits": logits, "cam_p99": cam_p99, "cam_max": cam_max}


# --------------------------------------------------------------------------------------
# Linear heads on cached features
# --------------------------------------------------------------------------------------


def _fit_lr(X: np.ndarray, y: np.ndarray, C: float, seed: int = 0, w: np.ndarray | None = None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    lr.fit(scaler.transform(X), y, sample_weight=w)
    return scaler, lr


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def fit_linear_head(X: np.ndarray, y: np.ndarray, feature_keys: list[str], Cs: tuple[float, ...] = (0.01, 0.1, 1.0),
                    folds: int = 5, seed: int = 0, n_jobs: int = 4, meta: dict | None = None,
                    sample_weight: np.ndarray | None = None) -> tuple[LinearHead, dict]:
    """Standardise + L2 logistic regression; ``C`` picked by stratified K-fold CV AUC on TRAIN."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold

    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    folds = max(2, min(folds, int(np.bincount(y).min())))
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y))

    def one(C: float, tr: np.ndarray, te: np.ndarray) -> float:
        sc, lr = _fit_lr(X[tr], y[tr], C, seed, w[tr])
        return _auc(y[te], lr.decision_function(sc.transform(X[te])))

    scores = Parallel(n_jobs=n_jobs)(delayed(one)(C, tr, te) for C in Cs for tr, te in splits)
    cv = {str(C): float(np.nanmean(scores[i * folds:(i + 1) * folds])) for i, C in enumerate(Cs)}
    best_C = max(Cs, key=lambda C: cv[str(C)])
    scaler, lr = _fit_lr(X, y, best_C, seed, w)
    head = LinearHead.from_sklearn(scaler, lr, feature_keys, {**(meta or {}), "C": best_C, "cv_auc": cv})
    return head, {"cv_auc_by_C": cv, "best_C": best_C, "folds": folds}


def lofo_eval(X_tr: np.ndarray, y_tr: np.ndarray, fam_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
              fam_va: np.ndarray, C: float, seed: int = 0, n_jobs: int = 4, w_tr: np.ndarray | None = None,
              min_val: int = 5) -> dict:
    """Leave-one-family-out: for each AI generator family F in TRAIN, fit without F's images and
    score VAL images of F against all VAL reals. Uses TRAIN/VAL only (test_unseen untouched).
    ``mean_auc`` averages families with at least ``min_val`` VAL images."""
    from joblib import Parallel, delayed

    w_tr = np.ones(len(y_tr)) if w_tr is None else w_tr
    fams = sorted(set(fam_tr[y_tr == 1]))

    def one(f: str) -> tuple[str, dict]:
        n_val = int(((y_va == 1) & (fam_va == f)).sum())
        if n_val == 0:
            return f, {"n_val": 0}
        keep = ~((y_tr == 1) & (fam_tr == f))
        sc, lr = _fit_lr(X_tr[keep], y_tr[keep], C, seed, w_tr[keep])
        sel = (y_va == 0) | (fam_va == f)
        s = lr.decision_function(sc.transform(X_va[sel]))
        return f, {"auc": _auc(y_va[sel], s), "detection_rate": float((s[y_va[sel] == 1] > 0).mean()),
                   "n_val": n_val, "n_train_removed": int((~keep).sum())}

    res = dict(Parallel(n_jobs=n_jobs)(delayed(one)(f) for f in fams))
    aucs = [r["auc"] for r in res.values() if r.get("n_val", 0) >= min_val and not math.isnan(r.get("auc", math.nan))]
    return {"per_family": res, "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "min_auc": float(np.min(aucs)) if aucs else float("nan"), "n_families_scored": len(aucs)}


# --------------------------------------------------------------------------------------
# Script helpers
# --------------------------------------------------------------------------------------


def get_logger(log_path: str | Path | None = None) -> Callable[[str], None]:
    """Return a ``log(msg)`` function printing to stdout and appending to ``log_path``."""
    fh = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "a", buffering=1)  # noqa: SIM115 - lives for the process

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if fh:
            fh.write(line + "\n")

    return log


def write_json(path: str | Path, obj: object) -> None:
    """Write JSON with NaN -> null and numpy scalars converted."""
    def default(o: object):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        return str(o)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_nan_to_none(obj), indent=2, default=default, allow_nan=False)
    Path(path).write_text(text + "\n")


def _nan_to_none(o: object) -> object:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (np.floating,)) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    return o


def update_json(path: str | Path, key: str, value: object) -> None:
    """Read-modify-write one top-level key of a JSON file (creates it if missing)."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    write_json(path, data)


def load_cached(path: str | Path, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load an ``.npz`` cache written with a ``paths`` array and align its rows to ``df``."""
    z = np.load(path, allow_pickle=False)
    pos = {p: i for i, p in enumerate(z["paths"].tolist())}
    missing = [p for p in df["path"] if p not in pos]
    if missing:
        raise KeyError(f"{path}: {len(missing)} manifest rows missing from cache (e.g. {missing[0]}); re-run extraction")
    idx = np.array([pos[p] for p in df["path"]])
    return {k: z[k][idx] for k in z.files if k != "paths"}
