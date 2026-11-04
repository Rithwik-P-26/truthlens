# TruthLens data card

**Task:** binary classification of a single image as a real photo (label 0) or AI-generated (label 1).
**Key design goal:** an honest test of **generalization to unseen generators**. The `test_unseen` split holds
generator *families* that never appear in `train` or `val`.

* Manifest: `data/manifest.csv` (`path,label,source,split,dataset,license`). It is final when `data/READY` exists.
* Extra per-image metadata (original format, resolution, EXIF, hashes, prompt): `data/meta/manifest_full.csv`.
* Pipeline: `scripts/prepare_data.py` (`--stage all|download|normalize|split|audit`, seed 1337). It takes about
  10 minutes on an idle machine, downloads about 13 GB of raw parquet, keeps about 0.9 GB of normalized JPEGs,
  and deletes the raw files in `--stage all`.
* Stats and audit outputs: `data/meta/split_stats.json`, `data/meta/audit.json`, `data/meta/per_source_before.csv`.
* **v2.1 (current): explicit-content filter + top-up** by `scripts/prepare_data_filter.py` (uses
  `scripts/prepare_data_nsfw.py`). The pre-filter manifest is kept as `data/meta/manifest_v1.csv`. See
  [Content filtering](#content-filtering-v2).

## Sources

| Dataset | What we used | License |
|---|---|---|
| [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), a subset of [GenImage](https://github.com/GenImage-Dataset/GenImage) (Zhu et al., NeurIPS 2023) 14 of 18 parquet shards (12 in v1, plus 2 for the v2 top-up): real **ImageNet** photos and fakes from ADM, GLIDE, VQDM, Wukong, SD 1.5 and Midjourney v5, generated from ImageNet class names (content-matched to the reals) | CC BY-NC-SA 4.0. The real images fall under the ImageNet terms of access (non-commercial research). |
| [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (`core` config) | 60 row groups (v1) plus 90 more (v2 top-up) from the `validation` shards (real **LAION** and **Pexels** photos plus about 80 modern generators in our sample, around 30–55 images each, of which 57 remain after dedup and balancing) and 20 (v1) plus 32 (v2 top-up) row groups from the `test` shards (real **ImageNet** and **DOCCI** photos plus the newest generators). The fakes are generated from captions of the real images, so content is roughly matched. | CC BY-NC 4.0. The real images keep their upstream terms: LAION (web copyright, URLs only), Pexels License, DOCCI CC BY 4.0, ImageNet terms. |

We searched the hub for other ungated candidates and rejected them:

* CommunityForensics (CC BY-NC-SA, 200 GB+, mostly huge PNGs)
* ELSA D3 (real images are only URLs)
* AIGC-Detection-Benchmark (overlaps GenImage)
* DRAGON and Fake2M (tar-only, TB-scale)
* small Kaggle mirrors (unclear provenance or license)

**Non-commercial only.** Every source is NC-licensed, so the dataset and anything trained on it are for research
and demo use only.

## Splits and counts (final, v2.1 after content filtering)

| split | real | AI | total | AI generators |
|---|---:|---:|---:|---|
| train | 7,569 | 7,569 | 15,138 | 5 GenImage + 33 OpenFake "seen" generators (38 seen in total) |
| val | 996 | 996 | 1,992 | same pool as train |
| test | 1,192 | 1,192 | 2,384 | same pool as train (in-distribution) |
| test_unseen | 2,663 | 2,663 | 5,326 | 32 held-out generators, see below |

Breakdown by source pool. Within each pool and split, real and AI counts are equal, so the "which upstream
dataset does this look like" signal cannot predict the label.

| pool | split | real (source: n) | AI (source: n) |
|---|---|---|---|
| GenImage | train | imagenet 6,389 | adm 1,299, glide 1,225, sd15 1,363, vqdm 1,202, wukong 1,300 |
| GenImage | val | imagenet 820 | 150–173 per generator |
| GenImage | test | imagenet 982 | 180–205 per generator |
| GenImage | test_unseen | imagenet 800 | **midjourney_v5 800** |
| OpenFake-val | train / val / test | laion + pexels 1,180 / 176 / 210 | 33 seen generators (SD 1.x/2.1/3.5, SDXL and fine-tunes, SD-Turbo, Flux.1 dev/schnell/pro and fine-tunes, Imagen 3, Kolors, Qwen-Image, HiDream, Playground v2.5, Grok-2-image, Chroma, Mystic, ...) |
| OpenFake-val | test_unseen | laion + pexels 450 | 450 images from held-out families (DALL-E 3, Ideogram 3, Seedream 4/4.5, Flux.2-dev/klein, Nano-Banana 2, Midjourney 6, Imagen 4, GPT-image-1, Wan 2.2, ...) |
| OpenFake-test | test_unseen | imagenet 953 + **docci 460** | 1,413 images from the newest generators: z-image-turbo 427, flux.2-klein-9b 337, gpt-image-1.5 193, midjourney-7 134, veo-3 77, recraft v3 42, wan-2.5 40, gpt-image-2, lumina, sora-2, recraft v2, ideogram 2, seedream 5, nano-banana-pro, ernie-image, ... |

**Unseen generators (32):** midjourney_v5, midjourney_6, midjourney_7, dalle_3, gpt_image_1, gpt_image_1_5,
gpt_image_2, z_image_turbo, flux_2_dev, flux_2_klein_4b, flux_2_klein_base_4b, flux_2_klein_9b, imagen_4_0,
ideogram_2_0, ideogram_3_0, seedream_v4_0, seedream_v4_5, seedream_v5_0, nano_banana_2, nano_banana_pro,
recraft_v2, recraft_v3, veo_3, sora_2, wan_video_2_2, wan_video_2_5, studio_ghibli_wan2_1_t2v_14b_v1_0,
lumina_17_2_25, ernie_image, aurora_20_1_25, frames_23_1_25, halfmoon_4_4_25.
In v2, `illustrious` and `ernie_image_turbo` were removed entirely by the content filter (see below).

**How the hold-out is enforced:**

* Hold-out is by **family prefix**, not just by exact name. Any OpenFake model whose name starts with
  midjourney, gpt-image, z-image, flux.2, illustrious, veo, sora, wan, seedream, nano-banana, ernie, recraft,
  ideogram, aurora, frames, halfmoon, lumina, dall or imagen-4, or that contains "wan2", is excluded from
  train/val/test.
* The script asserts that no generator name appears in both train/val and test_unseen.
* The script asserts that every `test` generator also appears in `train`. Generators with fewer than 10 images go
  to train only.

Note: `source` in `test_unseen` also contains **one new real source, DOCCI**, so the split measures false
positives on an unfamiliar real-photo domain as well as recall on unfamiliar generators.

## Preprocessing policy (identical for both classes)

1. Decode, apply the EXIF orientation, then convert to RGB. RGBA and palette images are composited onto white,
   and grayscale is converted to RGB.
2. Center-crop a square whose side equals the shorter side.
3. **Drop** the image if the shorter side is under 256 px, so nothing is ever upscaled.
4. Lanczos-resize to **224x224**. Every kept image is therefore *downscaled* by at least 1.14x.
5. Save as JPEG quality 95, 4:4:4 chroma, with no EXIF, ICC or other metadata.

**Why this policy:** the raw data had strong label-correlated shortcuts.

| Property (raw) | Real | AI |
|---|---|---|
| Format, GenImage | 100% JPEG | 100% PNG |
| Format, OpenFake | 99.3% JPEG | 37% PNG, 62% JPEG, 1% WEBP |
| Median shorter side, GenImage | 375 px | 256 px (ADM/GLIDE/VQDM), 512 (SD/Wukong), 1024 (MJ) |
| Median shorter side, OpenFake | 534 px (Pexels up to 6000 px) | 1024 px |
| Exactly square | 3% (GenImage) / 5% (OpenFake) | 100% (GenImage) / 44% (OpenFake) |
| Median file size, OpenFake | 140 KB | 428 KB |
| Has EXIF, OpenFake | 7% | 20% |
| Mode | RGB, some L (grayscale) | RGB, some RGBA |

Without normalization, format, size, aspect ratio or metadata alone would separate the classes. After
normalization, all 24,840 images are 224x224 RGB JPEG q95 with no EXIF, so none of these properties are
informative. See `reports/figures/data_resolution_format_before_norm.png`.

**Generators dropped:**

* BigGAN: native 128 px, which would require upscaling.
* `tiny-random-sana`: a random-weight test checkpoint that produces noise.
* SD 1.4 is a GenImage class but was absent from the 12 Tiny-GenImage shards we downloaded.

1,268 images (mostly small ImageNet reals) were dropped by the 256 px rule.

## Deduplication and leakage checks

* **Exact duplicates:** 11 by md5 of the raw bytes.
* **Near duplicates:** 64-bit dHash on the normalized image, Hamming distance at most 4, using 8-way chunk LSH
  (exact for distances up to 7) and union-find clustering. This found 26 clusters covering 90 images.
* Clusters containing both labels were dropped entirely (49 images). This catches cases like a real image and a
  fake made from it.
* Exactly one image per cluster is kept, so **no near-identical pair can cross splits**. The script asserts that
  cluster ids are unique.
* The check runs across datasets as well, so ImageNet reals that appear in both GenImage and OpenFake are
  deduplicated.

## Bias checks after normalization

These use 400 random images per split and label (3,200 total). The table shows single-feature AUC: how well a
trivial statistic alone separates real from AI (0.5 means no signal).

| Feature | Median real | Median AI | AUC |
|---|---:|---:|---:|
| JPEG file size (KB) | 30.9 | 27.2 | 0.60 |
| Laplacian sharpness | 23.4 | 18.7 | 0.61 |
| Colorfulness (per-pixel channel std) | 14.1 | 15.9 | 0.54 |
| Brightness | 112.9 | 111.5 | 0.50 |

Width, height, mode and format are identical for every image, and no image has EXIF. The remaining weak
signals in sharpness and file size are partly genuine (generators produce smoother images) and partly a
resampling effect, because different sources were downscaled by different factors. They give an AUC of only
about 0.6, far from a shortcut.

## Known limitations and advice for modelling

* **Residual compression history.** Real photos were JPEG-compressed once before our re-encode, while most
  fakes were PNG. Downscaling by at least 1.14x breaks the 8x8 JPEG grid alignment but may not erase every trace.
  **Use random JPEG (quality 60–100) and resize/blur augmentation on both classes in training.** This is also
  needed for the robustness claim.
* **Different downscale factors per source.** GenImage ADM/GLIDE/VQDM were scaled 256→224 (1.14x), ImageNet
  reals about 1.5x, and modern 1024 px fakes about 4.6x. Report per-generator metrics (use the `source` column).
* **Imbalance of training content.** About 85% of training images are from GenImage (ImageNet-style content
  and older generators). Modern seen generators number 33 but have only about 1.2k training images in total.
  Consider sample weights by pool or source.
* **test_unseen mixes two kinds of shift:** new generators, plus DOCCI, a new real domain. Report TPR on fakes
  and FPR on reals separately, ideally per source.
* **Some family overlap is inevitable.** For example, seen SDXL fine-tunes versus unseen Illustrious (an SDXL
  derivative), and seen Flux.1 versus unseen Flux.2. Hold-out is at the level of model release or family, not
  architecture.
* **Images are 224 px.** Fine detail from high-resolution generators is lost by design.
* The 224x224 crop discards image borders. At inference, apply the same center-crop and resize policy
  (`scripts/prepare_data.py::normalize_one`) for consistency.
* OpenFake "real" LAION images are web images and may contain watermarks, compression or edits. Labels are
  as provided by the datasets and were not manually verified.
* All data is licensed **non-commercial**.

## Content filtering (v2)

The v1 dataset contained explicit images (mostly from the OpenFake anime/"Pony" fine-tunes and LAION). Policy
decided by the team: remove **explicit** content (nudity, exposed genitals or breasts, sexual acts,
pornographic or erotic imagery). Merely suggestive content is out of scope. Every image was scanned: all 24,840
v1 manifest images, all top-up candidates and the 12 gallery images. Real and AI images went through the
**same** filter. **An image is removed if ANY of the three signals fires.**

| Signal | Model / rule | Threshold | Why this threshold |
|---|---|---|---|
| a. Image classifier | [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection) (ViT-B/16), p(nsfw) | **≥ 0.30** | The score is strongly bimodal. 24,514 of 24,840 v1 images score < 0.05 and 62 score > 0.95, with a thin, flat valley in between. 0.30 sits in that valley, below the classifier's 0.5 default, so it is moderately conservative. Lowering it to 0.2 adds only 28 images. |
| b. CLIP concept check | [`CompVis/stable-diffusion-safety-checker`](https://huggingface.co/CompVis/stable-diffusion-safety-checker): its 17 NSFW concept embeddings and per-concept thresholds (plus 3 "special care" concepts), applied to CLIP ViT-L/14 image embeddings | max concept score **> 0** (the checker's own decision rule) | We verified that the checker's vision tower and projection are bit-identical to open_clip `ViT-L-14`/`openai` (`force_quick_gelu`). That let us reuse the cached training embeddings (`models/cache/features/clip_vitl14.npz`) for v1 images and compute fresh ones for candidates. The checker is known to over-flag (skin, swimwear), which is acceptable for removal. |
| c. Prompt keywords | word-boundary regex of explicit terms (`KEYWORD_RE` in `scripts/prepare_data_nsfw.py`), applied to AI images only, since reals have no prompt | any match | Prompts are euphemistic, so this is only a backstop. |

A zero-shot CLIP text-prompt score (explicit vs neutral prompts) was also computed. It is stored as
`p_clip_zeroshot` but is **not used**, because it was uncalibrated: its median was 0.26 on ImageNet.

**Generator / source rule:** if at least 25% of a generator's scanned images (v1 plus candidates, n ≥ 8) are
flagged, the generator is dropped from all splits. These are explicit-content fine-tunes. The same rule applies
to real sources; none reached it. Flag rates were ImageNet 4.9%, LAION 3.5%, DOCCI 4.1% and Pexels 6.3%.

| Dropped generator | Flag rate | Images scanned | Where it was |
|---|---:|---:|---|
| cyberrealistic_pony_v16_0 | 63% | 46 | seen (OpenFake) |
| cyberrealistic_pony_v14_0 | 48% | 52 | seen |
| anima_preview | 48% | 50 | seen |
| cyberrealistic_pony_v15_0 | 47% | 51 | seen |
| illustrious | 43% | 307 | **unseen** (OpenFake test) |
| cyberrealistic_pony_v14_1 | 41% | 51 | seen |
| anima_preview2 | 39% | 52 | seen |
| ernie_image_turbo | 30% | 23 | **unseen** |
| novaanimexl_ilv140 | 26% | 57 | seen |
| animagine_xl_3_0, animagine_xl_3_1, animagine_xl_4_0, one_obsession_17_red_sdxl | below 25% | 177 in manifest | seen (dropped in v2.1, see below) |

**Signal overlap on the v1 manifest:**

| Fired | Images |
|---|---:|
| SD-checker only | 1,641 |
| Falconsai only | 101 |
| Prompt only | 31 |
| Two or more signals | 68 |
| Any signal | 1,841 of 24,840 (7.4%) |

The flag rate was 4.8% on real images and 10.0% on AI images.

**Removed from v1 by the filter (v2):** 2,111 images (1,841 flagged, 270 from dropped generators). By split, label and dataset:

| split | label | OpenFake flagged | OpenFake dropped-generator | GenImage flagged | total |
|---|---|---:|---:|---:|---:|
| train | real | 56 | 0 | 319 | 375 |
| train | AI | 126 | 108 | 626 | 860 |
| val | real | 7 | 0 | 40 | 47 |
| val | AI | 20 | 12 | 70 | 102 |
| test | real | 7 | 0 | 50 | 57 |
| test | AI | 26 | 19 | 97 | 142 |
| test_unseen | real | 90 | 0 | 31 | 121 |
| test_unseen | AI | 224 | 131 | 52 | 407 |

**Top-up, also 2,111 images, so every (pool, split, label) cell is back to its v1 count.** Candidates were:

* 90 new OpenFake `validation` row groups and 32 new `test` row groups
* 2 new Tiny-GenImage shards (train-00008, train-00009)
* v1 images that were never used (spare ImageNet reals, Midjourney v5)

All candidates went through the identical normalization. Because the GPU was shared with training, a random
subset of 3,870 candidates (about 1.5–2x the need per pool and role) was scanned with the **same three filters**.
359 were flagged and removed. The rest were deduplicated (md5 plus dHash, Hamming distance ≤ 4) against every
kept image in all splits and against each other, leaving 3,369.

Hold-out rules for candidates:

* Any fake from an OpenFake `test` shard, from an unseen family prefix, or from Midjourney v5 can only go to
  `test_unseen`.
* Any other fake is "seen" and can go to train, val or test. It can only enter val or test if its generator has
  train images.

| pool | split | real added | AI added |
|---|---|---:|---:|
| GenImage | train / val / test / test_unseen | 319 / 40 / 50 / 31 | 626 / 70 / 97 / 52 |
| OpenFake-val | train / val / test / test_unseen | 56 / 7 / 7 / 21 | 234 / 32 / 45 / 36 |
| OpenFake-test | test_unseen | 69 | 319 |

1,816 of the added images are from the new downloads and 295 are previously unused v1 images.

**v2.1: anime-style generators dropped as a precaution.** NSFW classifiers are least reliable on anime and
stylized content, so the team also dropped the four remaining anime-style seen generators: `animagine_xl_3_0`,
`animagine_xl_3_1`, `animagine_xl_4_0` and `one_obsession_17_red_sdxl`. They were dropped even though their flag
rates were under 25%. This removed 177 AI images (train 134, val 20, test 23, all from the OpenFake-val pool), and
their files were deleted. The freed slots were refilled one-for-one with **seen** OpenFake generators from 16 new
`validation` row groups (row groups 10–11 of shards 0–7). Of 331 candidates scanned with the same three filters,
22 were flagged. The refill used the same dedup and the same hold-out rules: test and val only take generators
that are present in train. No real images had to be dropped, and no new generator hit the 25% rule.

**`data/meta/new_paths_v2.csv` lists all 2,248 manifest paths that were not in the v1 manifest.** Feature
caches only need to embed those. All invariants still hold:

* no generator appears in both train/val and test_unseen
* every val and test generator is in train
* real and AI counts are equal per pool and split
* no flagged image and no dropped generator remains
* every manifest file exists

Every image file not in the final manifest was deleted from disk (17,539 files: flagged, dropped, duplicate and
unused), along with the raw download cache. Per-image scores and flags are in `data/meta/nsfw_scan.csv`, which
has path, split, label, source, every score and flag, and removed yes/no, with no image content. Summary numbers
are in `data/meta/filter_stats.json`.

**Gallery (`examples/`):** all 12 images are clean (max p_falconsai 0.0004, max SD-checker score −0.001, no prompt
flags), and all of them are in the current manifest.

**Limitations of the filter:**

* Classifiers are imperfect. Some explicit images may remain (false negatives), especially stylized or anime
  content and small or partial nudity at 224 px. Merely suggestive images, such as swimwear, lingerie or pin-up
  poses, are intentionally kept.
* The SD checker over-flags some benign images (skin, swimwear, people). Those were removed too, which slightly
  reduces images of people in both classes.
* The filter removed about twice as many AI as real images (10% vs 4.8%). Each cell was refilled to balance
  with clean images, but the remaining AI images are slightly less "skin-heavy" than before.
* Thresholds were chosen from score distributions, not by viewing images. The author then spot-checked the filtered
  image folders by hand.

## Figures

* `reports/figures/data_counts_per_source_split.png`: images per source and split.
* `reports/figures/data_resolution_format_before_norm.png`: raw resolution and file format per class before
  normalization.
