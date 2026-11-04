"""Evaluate every system on val/test/test_unseen and write the reports.

Systems: the deployed ensemble, each member, and baselines/ablations (UnivFD-style
final-layer probe, multi-layer probes per backbone, spectral LR, metadata shortcut audit,
stacked-LR ensemble). Every non-deployed system is Platt-calibrated on VAL so that
calibration metrics are comparable. Nothing is fitted or selected on test.

Outputs
-------
``reports/metrics.json`` (schema_version 2, Contract 3; ``robustness`` and ``latency`` are
written by ``eval_robustness.py`` / ``eval_latency.py`` and preserved here)::

    accuracy, f1, roc_auc, ece, detection_rate, false_positive_rate : deployed ensemble on headline_split
    headline_split : "test_unseen" (generators never seen in training)
    latency_ms_cpu : median Detector.predict, 1024x1024 upload, 2 CPU threads
    splits         : {test|test_unseen|val: {n, accuracy, f1, roc_auc, average_precision, ece, brier,
                      detection_rate, false_positive_rate, balanced_accuracy, coverage,
                      selective_accuracy, ci95{roc_auc, accuracy}}}
    per_generator  : [{split, source, label, accuracy (detection rate or specificity), n, roc_auc_vs_real}]
    robustness     : {perturbation: {accuracy, roc_auc, ece, detection_rate, false_positive_rate, by_split}}
    models         : {system id: {label, kind, test{...}, test_unseen{...}, val{...}}}
    decision, confusion_with_uncertain, data, head_selection, latency, per_source_by_system

All accuracy-type values are fractions in [0, 1]; CIs are stratified bootstrap (1000 resamples).

``reports/RESULTS.md`` (human-readable) and figures in ``reports/figures/``.
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

from truthlens.calibrate import (Ensemble, fit_calibrator, fit_ensemble, reliability_curve,  # noqa: E402
                                 selective_stats, sigmoid)
from truthlens.metrics import binary_metrics, bootstrap_ci, per_source_metrics  # noqa: E402
from truthlens.train import get_logger, manifest_summary, read_manifest, write_json  # noqa: E402

EVAL_SPLITS = ("val", "test", "test_unseen")
# dataviz reference palette (fixed categorical order), light surface
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------------------
# Assemble systems
# --------------------------------------------------------------------------------------


def build_systems(df, cache: Path, config: dict, log, models_dir: Path | None = None) -> dict[str, dict]:
    """Return {id: {label, kind, prob: np.ndarray aligned to df rows (nan where missing)}}."""
    val = (df.split == "val").to_numpy()
    yv = df.label.to_numpy()[val]
    systems: dict[str, dict] = {}

    def load_logits(name: str) -> np.ndarray | None:
        p = cache / "preds" / f"{name}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        pos = {q: i for i, q in enumerate(z["paths"].tolist())}
        out = np.full(len(df), np.nan, dtype=np.float64)
        idx = [(i, pos[q]) for i, q in enumerate(df.path) if q in pos]
        if idx:
            rows, src = zip(*idx)
            out[list(rows)] = z["logits"][list(src)]
        return out

    ens = Ensemble.from_dict(config["ensemble"])
    logits = {m: load_logits(m) for m in ("clip", "cnn", "freq", "meta")}
    have = ~np.any([np.isnan(logits[m]) for m in ens.members], axis=0)
    prob = np.full(len(df), np.nan)
    prob[have] = ens.prob({m: logits[m][have] for m in ens.members})
    names = " + ".join(config["members"][m]["display_name"] for m in ens.members)
    systems["ensemble"] = {"label": f"TruthLens ensemble ({names})", "kind": "deployed", "prob": prob}
    raw = np.full(len(df), np.nan)
    raw[have] = sigmoid(np.mean([logits[m][have] for m in ens.members], axis=0))
    systems["ensemble_uncalibrated"] = {"label": "Ensemble before calibration (mean raw logit)",
                                        "kind": "ablation", "prob": raw}

    fast_path = (models_dir or Path("models")) / "config_fast.json"
    if fast_path.exists():  # deployable ViT-B/16 fallback for slow CPUs
        fcfg = json.loads(fast_path.read_text())
        fens = Ensemble.from_dict(fcfg["ensemble"])
        zf = {"clip": load_logits(fcfg["members"]["clip"]["head_name"]), "cnn": logits["cnn"]}
        okf = ~np.any([np.isnan(zf[m]) for m in fens.members], axis=0)
        pf = np.full(len(df), np.nan)
        pf[okf] = fens.prob({m: zf[m][okf] for m in fens.members})
        systems["ensemble_fast"] = {"label": "Fast fallback ensemble (CLIP ViT-B/16 probe + CNN)",
                                    "kind": "ablation", "prob": pf}

    for m in ens.members:
        p = np.full(len(df), np.nan)
        ok = ~np.isnan(logits[m])
        p[ok] = ens.member_calibrators[m](logits[m][ok])
        systems[m] = {"label": f"{config['members'][m]['display_name']} (member)", "kind": "member", "prob": p}

    # alternative combinations (fitted on VAL only)
    if have[val].all() and len(ens.members) > 1:
        zv = {m: logits[m][val] for m in ens.members}
        alt = fit_ensemble(zv, yv, ens.members, method="stacked_lr")
        p = np.full(len(df), np.nan)
        p[have] = alt.prob({m: logits[m][have] for m in ens.members})
        systems["ensemble_stacked"] = {"label": "Stacked-LR ensemble (weights fit on VAL)", "kind": "ablation",
                                       "prob": p, "weights": alt.weights}
        extra = [m for m in ("freq",) if m not in ens.members and logits[m] is not None]
        if extra:
            mem = ens.members + extra
            ok3 = have & ~np.isnan(logits["freq"])
            alt3 = fit_ensemble({m: logits[m][val] for m in mem}, yv, mem)
            p = np.full(len(df), np.nan)
            p[ok3] = alt3.prob({m: logits[m][ok3] for m in mem})
            systems["ensemble_plus_freq"] = {"label": "Ensemble + spectral member", "kind": "ablation", "prob": p}

    # baselines / ablations from cached heads, each Platt-calibrated on VAL
    for f in sorted((cache / "preds").glob("*.npz")):
        name = f.stem
        if name in ("clip", "cnn") or name.endswith("_head"):
            continue
        if name in ens.members or name == config["members"]["clip"].get("head_name"):
            continue  # the deployed probe is already the "clip" member row
        z = load_logits(name)
        ok = ~np.isnan(z)
        cal = fit_calibrator(z[val & ok], df.label.to_numpy()[val & ok], "platt")
        p = np.full(len(df), np.nan)
        p[ok] = cal(z[ok])
        label, kind = _baseline_label(name)
        systems[name] = {"label": label, "kind": kind, "prob": p}
    log(f"systems: {list(systems)}")
    return systems


def _baseline_label(name: str) -> tuple[str, str]:
    if name == "meta":
        return "File-metadata shortcut audit (format/size/JPEG quality LR)", "baseline"
    if name == "freq":
        return "Spectral LR (azimuthal power spectrum + residual peaks)", "baseline"
    if name.startswith("clip_"):
        _, bb, variant = name.split("_")
        arch = {"vitl14": "ViT-L/14", "vitb16": "ViT-B/16", "vitb32": "ViT-B/32"}.get(bb, bb)
        if variant == "final":
            return f"UnivFD-style probe, CLIP {arch} final embedding", "baseline"
        return f"Multi-layer CLIP {arch} probe (final + intermediate CLS)", "ablation"
    return name, "baseline"


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def evaluate(df, systems: dict, threshold: float, n_boot: int, log) -> dict:
    out = {}
    y_all, src_all = df.label.to_numpy(), df.source.to_numpy().astype(str)
    for sid, s in systems.items():
        rec = {"label": s["label"], "kind": s["kind"], "splits": {}, "per_source": {}}
        if "weights" in s:
            rec["weights"] = s["weights"]
        for split in EVAL_SPLITS:
            m = (df.split == split).to_numpy() & ~np.isnan(s["prob"])
            if m.sum() == 0:
                continue
            y, p = y_all[m], s["prob"][m]
            met = binary_metrics(y, p, threshold)
            if split != "val":
                met["ci95"] = {k: bootstrap_ci(y, p, k, threshold, n_boot=n_boot) for k in ("roc_auc", "accuracy")}
            rec["splits"][split] = met
            if split != "val":
                rec["per_source"][split] = per_source_metrics(y, p, src_all[m], threshold)
        out[sid] = rec
        t = rec["splits"].get("test", {})
        u = rec["splits"].get("test_unseen", {})
        log(f"{sid:28s} test AUC {t.get('roc_auc', float('nan')):.4f} acc {t.get('accuracy', float('nan')):.4f} | "
            f"unseen AUC {u.get('roc_auc', float('nan')):.4f} acc {u.get('accuracy', float('nan')):.4f} "
            f"ECE {u.get('ece', float('nan')):.4f}")
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
        "lines.linewidth": 1.8,
    })
    return plt


def fig_roc(df, systems: dict, ids: list[str], fig_dir: Path) -> None:
    from sklearn.metrics import roc_curve

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    for ax, split in zip(axes, ("test", "test_unseen")):
        for i, sid in enumerate(ids):
            m = (df.split == split).to_numpy() & ~np.isnan(systems[sid]["prob"])
            if m.sum() == 0 or len(set(df.label[m])) < 2:
                continue
            fpr, tpr, _ = roc_curve(df.label[m], systems[sid]["prob"][m])
            from sklearn.metrics import auc as _auc

            ls = "-" if systems[sid]["kind"] in ("deployed", "member") else "--"
            ax.plot(fpr, tpr, ls, color=PALETTE[i % len(PALETTE)],
                    label=f"{systems[sid]['label']} (AUC {_auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], ":", color=INK2, linewidth=1)
        ax.set(title={"test": "Test (seen generators)", "test_unseen": "Test-unseen (new generators)"}[split],
               xlabel="False positive rate (real flagged as AI)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    axes[0].set_ylabel("True positive rate (AI detected)")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(df, systems: dict, fig_dir: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, split in zip(axes, ("test", "test_unseen")):
        for i, sid in enumerate(("ensemble", "ensemble_uncalibrated")):
            if sid not in systems:
                continue
            m = (df.split == split).to_numpy() & ~np.isnan(systems[sid]["prob"])
            if m.sum() == 0:
                continue
            rc = reliability_curve(systems[sid]["prob"][m], df.label.to_numpy()[m], n_bins=10)
            cnt = np.array(rc["count"])
            keep = cnt > 0
            ax.plot(np.array(rc["mean_pred"])[keep], np.array(rc["frac_ai"])[keep], "-o", markersize=5,
                    color=PALETTE[i], label="calibrated" if sid == "ensemble" else "before calibration")
        ax.plot([0, 1], [0, 1], ":", color=INK2, linewidth=1, label="perfect calibration")
        ax.set(title={"test": "Test", "test_unseen": "Test-unseen"}[split], xlabel="Predicted P(AI)")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Observed fraction AI")
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability_diagram.png", dpi=150)
    plt.close(fig)


def fig_confusion(df, p_all: np.ndarray, decision: dict, fig_dir: Path) -> dict:
    """2x3 matrices: true Real/AI x predicted Real/Uncertain/AI, with the deployed band."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, split in zip(axes, ("test", "test_unseen")):
        m = (df.split == split).to_numpy() & ~np.isnan(p_all)
        y, p = df.label.to_numpy()[m], p_all[m]
        conf = np.where(p >= decision["threshold"], p, 1 - p)
        pred = np.where(conf < decision["min_confidence"], 1, np.where(p >= decision["threshold"], 2, 0))
        mat = np.array([[int(((y == t) & (pred == k)).sum()) for k in range(3)] for t in (0, 1)])
        out[split] = {"rows_true": ["Real", "AI"], "cols_pred": ["Real", "Uncertain", "AI"], "counts": mat.tolist()}
        frac = mat / np.maximum(mat.sum(1, keepdims=True), 1)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]}\n{100 * frac[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                        color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), ["Real", "Uncertain", "AI"])
        ax.set_yticks(range(2), ["Real", "AI"])
        ax.set(xlabel="Verdict", ylabel="Truth", title={"test": "Test", "test_unseen": "Test-unseen"}[split])
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    return out


def fig_per_generator(results: dict, ids: list[str], fig_dir: Path) -> None:
    """Dot plot, one row per unseen generator (sorted): AUC vs reals per system + ensemble detection rate."""
    plt = _plt()
    per = {sid: results[sid]["per_source"].get("test_unseen", {}) for sid in ids}
    srcs = [s for s, r in per[ids[0]].items() if r["label"] == 1]
    if not srcs:
        return
    srcs.sort(key=lambda s: per[ids[0]][s].get("roc_auc_vs_real") or 0)
    y = np.arange(len(srcs))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 0.27 * len(srcs) + 1.6), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 2]})
    for i, sid in enumerate(ids):
        vals = [per[sid].get(s, {}).get("roc_auc_vs_real", np.nan) for s in srcs]
        ax.plot(vals, y, "o", color=PALETTE[i % len(PALETTE)], markersize=6 if i == 0 else 4.5,
                label=results[sid]["label"], zorder=3 if i == 0 else 2)
    ax.axvline(0.5, color=INK2, linestyle=":", linewidth=1)
    ax.set_yticks(y, [f"{s} (n={per[ids[0]][s]['n']})" for s in srcs], fontsize=7.5)
    ax.set(xlabel="ROC-AUC vs real images of the split", xlim=(0.4, 1.01),
           title="Generators never seen in training")
    rate = [per[ids[0]][s].get("detection_rate", np.nan) for s in srcs]
    ax2.barh(y, rate, height=0.6, color=PALETTE[0])
    ax2.set(xlabel="Ensemble detection rate (p >= 0.5)", xlim=(0, 1))
    for a in (ax, ax2):
        a.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(fig_dir / "per_generator_unseen_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_risk_coverage(df, p_all: np.ndarray, decision: dict, fig_dir: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, split in enumerate(EVAL_SPLITS):
        m = (df.split == split).to_numpy() & ~np.isnan(p_all)
        if m.sum() == 0:
            continue
        y, p = df.label.to_numpy()[m], p_all[m]
        pts = [selective_stats(p, y, decision["threshold"], c) for c in np.linspace(0.5, 0.99, 50)]
        ax.plot([q["coverage"] for q in pts], [q["selective_accuracy"] for q in pts], color=PALETTE[i], label=split)
        op = selective_stats(p, y, decision["threshold"], decision["min_confidence"])
        ax.plot(op["coverage"], op["selective_accuracy"], "o", color=PALETTE[i], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set(xlabel="Coverage (fraction given a verdict)", ylabel="Accuracy on covered images",
           title=f"Selective prediction (dot = deployed band, conf >= {decision['min_confidence']:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# RESULTS.md
# --------------------------------------------------------------------------------------


def _f(x, nd: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def card(m: dict) -> dict:
    """Contract-3 style summary of one split's metrics (fractions in [0, 1])."""
    out = {k: m.get(k) for k in ("n", "n_real", "n_ai", "accuracy", "balanced_accuracy", "f1", "precision",
                                 "roc_auc", "average_precision", "ece", "brier")}
    out["detection_rate"] = m.get("recall_tpr")
    out["false_positive_rate"] = None if m.get("specificity_tnr") is None else 1 - m["specificity_tnr"]
    if "ci95" in m:
        out["ci95"] = m["ci95"]
    return out


def _version_lines(metrics: dict) -> list[str]:
    """Two-line v1 vs current headline comparison (accuracy and ROC-AUC, seen and unseen)."""
    prev = metrics.get("previous_version")
    if not prev:
        return []
    out = []
    for split, name in (("test", "Seen generators (test)"), ("test_unseen", "Unseen generators (test_unseen)")):
        a, b = prev.get(split) or {}, metrics["splits"].get(split) or {}
        out.append(f"* {name}: AUC {_f(a.get('roc_auc'))} (v1) -> {_f(b.get('roc_auc'))} "
                   f"({metrics['data_info'].get('data_version')}), accuracy {_f(a.get('accuracy'))} -> "
                   f"{_f(b.get('accuracy'))}.")
    return out


def write_results_md(metrics: dict, path: Path) -> None:
    S = metrics["splits"]
    info = metrics.get("data_info", {})
    n_unseen = sum(1 for r in metrics["per_generator"] if r["split"] == "test_unseen" and r["label"] == "AI")
    new_reals = sorted({r["source"] for r in metrics["per_generator"] if r["split"] == "test_unseen"
                        and r["label"] == "real"} - {r["source"] for r in metrics["per_generator"]
                                                     if r["split"] == "test" and r["label"] == "real"})
    L = ["# TruthLens results", "",
         f"_Generated {metrics['generated']} by `scripts/eval_models.py` from `reports/metrics.json`._", "",
         "## Protocol", "",
         "* Members are trained on TRAIN. Calibration, the ensemble and the uncertainty band are fitted on VAL. "
         "The CLIP probe variant is chosen by leave-one-family-out AUC on TRAIN/VAL. Nothing is tuned on a test split.",
         f"* TEST shares its {info.get('n_seen_generators', '?')} generators with training. TEST-UNSEEN holds "
         f"{n_unseen} generators from families never seen in training or selection, plus new real-photo domain(s) "
         f"({', '.join(new_reals) or 'none'}), so we report the detection rate on fakes and the false-positive "
         "rate on reals separately.",
         "* One preprocessing policy end to end: every image (train, val, test and every upload) is centre-cropped "
         "to a square, Lanczos-resized to 224x224 and re-encoded as JPEG q95, exactly as in `normalize_image`. This "
         "discards some high-frequency evidence in large photos, but it makes these test numbers representative "
         "of the deployed app.",
         "* Family closeness remains between some seen and unseen generators (Flux.1 seen vs Flux.2 unseen; "
         "Imagen 3 seen vs Imagen 4 unseen), so a few 'unseen' families are near neighbours of training data.",
         "", "## Data version", "",
         f"Dataset {info.get('data_version', '?')}: content-filtered (NSFW image classifier, the Stable Diffusion "
         "safety checker on CLIP embeddings and a prompt keyword filter, applied identically to real and AI images; "
         "13 generators dropped and replaced with clean, filtered images; split sizes and class balance unchanged). "
         "Version 1 results are superseded and archived in `reports/archive_v1/` and `models/archive_v1/`. The "
         "two versions differ in content, so the deltas below are for transparency, not a controlled comparison.",
         ""] + _version_lines(metrics) + [
         "", "## Headline (deployed ensemble)", "",
         "| split | n | accuracy | F1 | ROC-AUC (95% CI) | ECE | detection rate (fakes) | FPR (reals) | "
         "coverage | selective acc. |", "|---|---|---|---|---|---|---|---|---|---|"]
    for split in ("test", "test_unseen"):
        s = S.get(split)
        if not s:
            continue
        ci = (s.get("ci95") or {}).get("roc_auc", [None, None])
        L.append(f"| {split} | {s['n']} | {_f(s['accuracy'])} | {_f(s['f1'])} | {_f(s['roc_auc'])} "
                 f"({_f(ci[0])}-{_f(ci[1])}) | {_f(s['ece'])} | {_f(s['detection_rate'])} | "
                 f"{_f(s['false_positive_rate'])} | {_f(s.get('coverage'))} | {_f(s.get('selective_accuracy'))} |")
    L += ["", "Coverage and selective accuracy refer to images that get a verdict; the rest are labelled "
          f"Uncertain (calibrated confidence < {metrics['decision']['min_confidence']:.2f}).",
          "", "## Members, ensemble and baselines", "",
          "| system | kind | test acc | test AUC | test F1 | test ECE | unseen acc | unseen AUC | unseen F1 | "
          "unseen ECE | unseen det. rate | unseen FPR |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"deployed": 0, "member": 1, "baseline": 2, "ablation": 3}
    for sid, r in sorted(metrics["models"].items(), key=lambda kv: order.get(kv[1]["kind"], 9)):
        t, u = r.get("test") or {}, r.get("test_unseen") or {}
        L.append(f"| {r['label']} | {r['kind']} | {_f(t.get('accuracy'))} | {_f(t.get('roc_auc'))} | "
                 f"{_f(t.get('f1'))} | {_f(t.get('ece'))} | {_f(u.get('accuracy'))} | {_f(u.get('roc_auc'))} | "
                 f"{_f(u.get('f1'))} | {_f(u.get('ece'))} | {_f(u.get('detection_rate'))} | "
                 f"{_f(u.get('false_positive_rate'))} |")
    meta = metrics["models"].get("meta")
    if meta:
        L += ["", f"**Shortcut audit.** A logistic regression on file metadata alone (format, size, JPEG quality, "
              f"bits per pixel, EXIF) reaches ROC-AUC {_f((meta.get('test') or {}).get('roc_auc'))} on test and "
              f"{_f((meta.get('test_unseen') or {}).get('roc_auc'))} on test-unseen after normalisation (raw data: "
              "format alone separated the classes). The remaining signal is mostly JPEG bits per pixel, which "
              "reflects image texture."]
    for split in ("test_unseen", "test"):
        rows = [r for r in metrics["per_generator"] if r["split"] == split]
        if not rows:
            continue
        L += ["", f"## Per source, {split} (deployed ensemble)", "",
              "Accuracy is the detection rate for AI sources and the specificity (1 - FPR) for real sources.", "",
              "| source | label | n | accuracy | ROC-AUC vs reals | mean P(AI) |", "|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r["label"] != "real", -r["n"])):
            L.append(f"| {r['source']} | {r['label']} | {r['n']} | {_f(r['accuracy'])} | "
                     f"{_f(r.get('roc_auc_vs_real'))} | {_f(r['mean_prob_ai'])} |")
    rob = metrics.get("robustness")
    if rob:
        L += ["", "## Robustness (deployed ensemble, perturbation then the deployed normalisation)", "",
              f"Subsample: {metrics.get('robustness_meta', {}).get('n_per_split', '?')} images per split, "
              "stratified by class and source.", "",
              "| perturbation | unseen acc | unseen AUC | unseen det. rate | unseen FPR | test acc | test AUC |",
              "|---|---|---|---|---|---|---|"]
        for pert, r in rob.items():
            t = r.get("by_split", {}).get("test", {}).get("ensemble", {})
            L.append(f"| {pert} | {_f(r.get('accuracy'))} | {_f(r.get('roc_auc'))} | {_f(r.get('detection_rate'))} | "
                     f"{_f(r.get('false_positive_rate'))} | {_f(t.get('accuracy'))} | {_f(t.get('roc_auc'))} |")
    lat = metrics.get("latency")
    if lat:
        L += ["", "## CPU latency", "", lat.get("note", ""), "", "| component | median ms | p90 ms |", "|---|---|---|"]
        for k, v in lat.get("components", {}).items():
            L.append(f"| {k} | {_f(v.get('median_ms'), 0)} | {_f(v.get('p90_ms'), 0)} |")
    hs = metrics.get("head_selection")
    if hs:
        L += ["", "## CLIP probe selection (leave-one-family-out on TRAIN/VAL)", "",
              f"Deployed: `{hs['deployed_clip_head']}`; rule: {hs['selection_rule']}.", "",
              "| head | dim | CV AUC (train) | VAL AUC | LOFO mean AUC | LOFO min AUC |", "|---|---|---|---|---|---|"]
        for name, r in hs["heads"].items():
            L.append(f"| {name} | {r['dim']} | {_f(r['cv']['cv_auc_by_C'][str(r['cv']['best_C'])], 4)} | "
                     f"{_f(r['val_auc'], 4)} | {_f(r['lofo']['mean_auc'], 4)} | {_f(r['lofo']['min_auc'], 4)} |")
    L += ["", "## Caveats", "",
          f"* All sources are non-commercial research datasets. About {info.get('genimage_train_share', 0.85):.0%} of "
          "TRAIN AI images are GenImage (ImageNet content, 2022-era generators); modern generators contribute "
          f"{info.get('n_train_ai_modern', '?')} training images, so they are up-weighted by source pool "
          "(alpha = 0.5, fixed in advance).",
          "* Many unseen generators have fewer than 50 images, so their per-source rates have wide uncertainty.",
          "* Known failure modes from the robustness table: blur and upsampling make real photos look generated "
          "(ranking AUC stays high but the 0.5 threshold produces many false positives), and JPEG recompression "
          "hides generator traces (more missed fakes). Inputs smaller than 224 px are therefore answered with "
          "'Uncertain' in the app; blur cannot be detected reliably and remains a limitation.",
          "* The pre-registered ensemble (equal-weight calibrated logits) is slightly below the CLIP probe alone on "
          "clean test sets, but more robust under JPEG compression. The stacked-LR row is shown for comparison only "
          "and was not selected after seeing test results.",
          "* Robustness numbers use a stratified subsample and perturb the 224 px normalised image. An upload "
          "degraded at full resolution is partly 'cleaned' by the downscale, so these numbers are conservative.",
          "* CPU latency was measured on Apple silicon (AMX-accelerated matrix multiplies); a small x86 CPU "
          "is slower. The live Space runs on ZeroGPU (measured numbers in the README). `predict()` reports "
          "`latency_ms` live.",
          "* Calibration is fitted on VAL (seen generators). On unseen generators the probabilities are less "
          "reliable; see the test-unseen ECE and the reliability diagram."]
    L += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted((path.parent / "figures").glob("*.png"))
                                   if not p.name.startswith("data_")]
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--cache-dir", default=str(ROOT / "models" / "cache"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--data-version", default="v2.1", help="dataset version label for the reports")
    ap.add_argument("--previous-metrics", default=str(ROOT / "reports" / "archive_v1" / "metrics.json"),
                    help="metrics.json of the superseded version, for the v1 vs current comparison")
    ap.add_argument("--log", default=str(ROOT / "logs" / "eval_models.log"))
    args = ap.parse_args()

    log = get_logger(args.log)
    df = read_manifest(args.manifest, args.root)
    cache, reports = Path(args.cache_dir), Path(args.reports_dir)
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.models_dir) / "config.json").read_text())
    decision = config["decision"]

    systems = build_systems(df, cache, config, log, Path(args.models_dir))
    results = evaluate(df, systems, decision["threshold"], args.n_boot, log)
    p_ens = systems["ensemble"]["prob"]
    splits = {}
    for split in ("test", "test_unseen", "val"):
        m = (df.split == split).to_numpy() & ~np.isnan(p_ens)
        if m.any() and split in results["ensemble"]["splits"]:
            splits[split] = {**card(results["ensemble"]["splits"][split]),
                             **selective_stats(p_ens[m], df.label.to_numpy()[m], decision["threshold"],
                                               decision["min_confidence"])}

    plot_ids = [s for s in ("ensemble", "clip", "cnn", "freq", "clip_vitl14_final", "meta") if s in systems]
    fig_roc(df, systems, plot_ids, fig_dir)
    fig_reliability(df, systems, fig_dir)
    confusion = fig_confusion(df, p_ens, decision, fig_dir)
    fig_per_generator(results, [s for s in ("ensemble", "clip", "cnn", "freq") if s in results], fig_dir)
    fig_risk_coverage(df, p_ens, decision, fig_dir)

    per_generator = []
    for split in ("test_unseen", "test"):
        for src, r in results["ensemble"]["per_source"].get(split, {}).items():
            per_generator.append({"split": split, "source": src, "label": "AI" if r["label"] else "real",
                                  "accuracy": r.get("detection_rate", r.get("specificity")), "n": r["n"],
                                  "roc_auc_vs_real": r.get("roc_auc_vs_real"), "mean_prob_ai": r["mean_prob_ai"]})
    metrics_path = reports / "metrics.json"
    old = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    tr_ai = df[(df.split == "train") & (df.label == 1)]
    gi = int((tr_ai.dataset == "TheKernel01/Tiny-GenImage").sum())
    data_info = {"data_version": args.data_version,
                 "n_seen_generators": int(tr_ai.source.nunique()),
                 "n_unseen_generators": int(df[(df.split == "test_unseen") & (df.label == 1)].source.nunique()),
                 "n_train_ai_modern": int(len(tr_ai) - gi), "genimage_train_share": gi / max(len(tr_ai), 1)}
    prev_path = Path(args.previous_metrics)
    previous = None
    if prev_path.exists():
        pm = json.loads(prev_path.read_text())
        previous = {"source": str(prev_path.relative_to(ROOT)) if prev_path.is_relative_to(ROOT) else str(prev_path),
                    **{sp: {k: pm["splits"][sp].get(k) for k in ("accuracy", "roc_auc", "f1", "ece")}
                       for sp in ("test", "test_unseen") if sp in pm.get("splits", {})}}
    head = splits.get("test_unseen", splits.get("test", {}))
    metrics = {
        "schema_version": 2,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": results["ensemble"]["label"],
        "headline_split": "test_unseen" if "test_unseen" in splits else "test",
        "accuracy": head.get("accuracy"), "f1": head.get("f1"), "roc_auc": head.get("roc_auc"),
        "ece": head.get("ece"), "detection_rate": head.get("detection_rate"),
        "false_positive_rate": head.get("false_positive_rate"),
        "latency_ms_cpu": (old.get("latency") or {}).get("latency_ms_cpu"),
        "splits": splits,
        "per_generator": per_generator,
        "robustness": old.get("robustness", {}),
        "models": {sid: {"label": r["label"], "kind": r["kind"],
                         **{sp: card(m) for sp, m in r["splits"].items()},
                         **({"weights": r["weights"]} if "weights" in r else {})} for sid, r in results.items()},
        "decision": dict(decision),
        "confusion_with_uncertain": confusion,
        "data": manifest_summary(df),
        "data_info": data_info,
        "previous_version": previous,
        "config_version": config.get("version"),
        "per_source_by_system": {sid: r["per_source"] for sid, r in results.items()},
    }
    head_sel = reports / "head_selection.json"
    if head_sel.exists():
        hs = json.loads(head_sel.read_text())
        metrics["head_selection"] = {"deployed_clip_head": hs["deployed_clip_head"], "selection_rule": hs["selection_rule"],
                                     "heads": {k: {kk: v[kk] for kk in ("dim", "cv", "val_auc", "lofo")}
                                               for k, v in hs["heads"].items()}}
    for key in ("robustness_meta", "latency"):
        if key in old:
            metrics[key] = old[key]
    write_json(metrics_path, metrics)
    write_results_md(json.loads(metrics_path.read_text()), reports / "RESULTS.md")
    log(f"wrote {metrics_path} and {reports / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
