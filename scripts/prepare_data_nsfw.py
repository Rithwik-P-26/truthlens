#!/usr/bin/env python
"""TruthLens content filter: score images for EXPLICIT content (nudity / sexual content) with three
independent signals and flag an image if ANY signal fires. Applied identically to real and AI images.

Signals
  a. Falconsai/nsfw_image_detection (ViT image classifier)          -> p_falconsai  (flag >= FALCON_THR)
  b. CompVis Stable-Diffusion safety checker concepts on CLIP ViT-L/14 embeddings -> sd_safety_score (flag > SD_THR)
     (a zero-shot text-prompt CLIP score is also stored as p_clip_zeroshot for diagnostics; it was too
      uncalibrated to use as a flag: median 0.26 even on ImageNet)
  c. Prompt keyword filter (AI images only; reals have no prompt)     -> kw_flag

Used as a library by scripts/prepare_data.py (top-up) and as a CLI:
    .venv/bin/python scripts/prepare_data_nsfw.py --paths-csv X.csv --out scores.csv
Never writes, copies or displays image content; outputs scores only.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FALCON_THR = 0.30
SD_THR = 0.0     # official SD safety-checker decision boundary

KEYWORD_RE = re.compile(
    r"\b(?:nude|nudes|nudity|naked|topless|bottomless|nsfw|porn|porno|pornographic|pornography|hentai|"
    r"explicit|sex|sexual|sexually|sexy|erotic|erotica|eroticism|genital|genitals|genitalia|penis|vagina|vulva|"
    r"nipple|nipples|areola|areolae|breasts|boobs|tits|orgasm|masturbat\w*|intercourse|fellatio|blowjob|"
    r"cum|cumshot|undressed|unclothed|pussy|anus|anal|fetish|bdsm|bondage|lewd|ahegao|futanari|"
    r"spread legs|without clothes|no clothes)\b",
    re.I,
)

EXPLICIT_PROMPTS = [
    "an explicit pornographic photo",
    "a photo of a naked woman with exposed breasts",
    "a photo of a naked man with exposed genitals",
    "a nude person, full frontal nudity",
    "a photo of people having sex",
    "an erotic nude photograph",
    "explicit hentai anime porn",
    "a topless woman",
    "an explicit sexual image",
]
NEUTRAL_PROMPTS = [
    "a photo", "a photo of a woman wearing clothes", "a photo of a man wearing clothes",
    "a photo of a woman in a swimsuit at the beach", "a fashion photo of a model in a dress",
    "a portrait photo of a person", "a group of people", "an anime illustration of a girl",
    "a fantasy illustration", "a landscape", "an animal", "a building or city", "food",
    "a painting", "text, a poster or a screenshot", "an everyday object", "a car or vehicle",
    "a shirtless man at the gym", "a statue or sculpture",
]


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _load_rgb(p):
    with Image.open(ROOT / p) as im:
        return im.convert("RGB")


@torch.no_grad()
def score_falconsai(paths: list[str], bs: int = 64) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(device()).eval()
    nsfw_idx = [i for i, n in model.config.id2label.items() if n.lower() == "nsfw"][0]
    out = []
    for i in range(0, len(paths), bs):
        ims = [_load_rgb(p) for p in paths[i:i + bs]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(device())
        out.append(model(pixel_values=x).logits.float().softmax(-1)[:, nsfw_idx].cpu().numpy())
        if (i // bs) % 50 == 0:
            print(f"falconsai {i}/{len(paths)}", flush=True)
    return np.concatenate(out) if out else np.zeros(0)


class ClipZeroShot:
    def __init__(self):
        import open_clip
        self.dev = device()
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", force_quick_gelu=True)
        self.model = self.model.to(self.dev).eval()
        tok = open_clip.get_tokenizer("ViT-L-14")
        with torch.no_grad():
            t = self.model.encode_text(tok(EXPLICIT_PROMPTS + NEUTRAL_PROMPTS).to(self.dev)).float()
        self.text = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
        self.n_exp = len(EXPLICIT_PROMPTS)

    def from_embeddings(self, emb: np.ndarray) -> np.ndarray:
        e = emb.astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        logits = 100.0 * e @ self.text.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return p[:, : self.n_exp].sum(axis=1)

    @torch.no_grad()
    def embed(self, paths: list[str], bs: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(paths), bs):
            x = torch.stack([self.pre(_load_rgb(p)) for p in paths[i:i + bs]]).to(self.dev)
            out.append(self.model.encode_image(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def clip_embeddings(paths: list[str], cache_npz: Path | None = ROOT / "models/cache/features/clip_vitl14.npz"):
    """open_clip ViT-L/14 (openai, quick_gelu) projected image embeddings; reuses the model agent's cache."""
    zs = ClipZeroShot()
    emb = np.zeros((len(paths), 768), np.float32)
    have = np.zeros(len(paths), bool)
    if cache_npz is not None and cache_npz.exists():
        d = np.load(cache_npz)
        idx = {p: i for i, p in enumerate(d["paths"].tolist())}
        rows = [idx.get(p, -1) for p in paths]
        have = np.array([r >= 0 for r in rows])
        if have.any():
            emb[have] = d["final"][[r for r in rows if r >= 0]].astype(np.float32)
    miss = [p for p, h in zip(paths, have) if not h]
    print(f"clip: {have.sum()} cached embeddings, computing {len(miss)}", flush=True)
    if miss:
        emb[~have] = zs.embed(miss)
    return emb, zs


def sd_safety_score(emb: np.ndarray) -> np.ndarray:
    """CompVis/stable-diffusion-safety-checker logic on CLIP ViT-L/14 image embeddings.

    Its vision tower + projection are bit-identical to open_clip ViT-L/14 'openai' (verified), so we only need
    its 17 concept embeddings + thresholds (and 3 'special care' ones). Returns max concept score; > 0 means the
    official checker flags the image as NSFW (it is tuned for nudity / sexual content).
    """
    c = SD_CONCEPTS
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    ce = c["concept_embeds"] / np.linalg.norm(c["concept_embeds"], axis=1, keepdims=True)
    se = c["special_care_embeds"] / np.linalg.norm(c["special_care_embeds"], axis=1, keepdims=True)
    special = (e @ se.T - c["special_care_embeds_weights"][None]) > 0
    adj = special.any(axis=1, keepdims=True) * 0.01
    return ((e @ ce.T - c["concept_embeds_weights"][None]) + adj).max(axis=1)


def _load_sd_concepts():
    f = ROOT / "data/meta/scan/sd_safety_concepts.pt"
    if not f.exists():
        from huggingface_hub import hf_hub_download
        sd = torch.load(hf_hub_download("CompVis/stable-diffusion-safety-checker", "pytorch_model.bin"),
                        map_location="cpu", weights_only=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: sd[k].clone() for k in ["concept_embeds", "special_care_embeds", "concept_embeds_weights",
                                               "special_care_embeds_weights"]}, f)
    return {k: v.float().numpy() for k, v in torch.load(f, weights_only=True).items()}


SD_CONCEPTS = None


def keyword_flag(prompts: pd.Series) -> pd.Series:
    return prompts.fillna("").astype(str).str.contains(KEYWORD_RE, regex=True)


def scan(df: pd.DataFrame) -> pd.DataFrame:
    """df needs columns path, label, prompt. Returns df with score/flag columns added."""
    global SD_CONCEPTS
    SD_CONCEPTS = SD_CONCEPTS or _load_sd_concepts()
    df = df.copy()
    paths = df["path"].tolist()
    df["p_falconsai"] = score_falconsai(paths)
    emb, zs = clip_embeddings(paths)
    df["sd_safety_score"] = sd_safety_score(emb)
    df["p_clip_zeroshot"] = zs.from_embeddings(emb)  # diagnostic only (uncalibrated, not used to flag)
    df["flag_falconsai"] = df["p_falconsai"] >= FALCON_THR
    df["flag_sdsafety"] = df["sd_safety_score"] > SD_THR
    df["flag_prompt"] = keyword_flag(df["prompt"]) & (df["label"] == 1)
    df["flagged"] = df["flag_falconsai"] | df["flag_sdsafety"] | df["flag_prompt"]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.paths_csv)
    if "prompt" not in d:
        d["prompt"] = ""
    scan(d).to_csv(a.out, index=False)
