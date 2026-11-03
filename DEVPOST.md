# TruthLens: Devpost submission text

Paste each section into the matching Devpost field. All numbers come from `reports/metrics.json` / `reports/RESULTS.md`.

**Project name:** TruthLens

**Tagline (≤ 200 characters):**
Is this photo real or AI-generated? A calibrated answer with visual evidence, tested on 32 image generators it never saw in training.

**Elevator pitch:**
TruthLens is an AI-generated-image detector that gives a calibrated probability, says "Uncertain" instead of guessing,
and shows its evidence: a heatmap, the frequency spectrum and plain-language reasons. Most detectors only work on the
generators they were trained on. TruthLens is tested on 32 generators it never saw, including Midjourney 7, GPT-Image
and FLUX.2, and reaches ROC-AUC 0.927 on them while flagging only 3.2% of real photos. The dataset was cleaned of the
format and resolution shortcuts that inflate many published results.

**Links:**

- Demo: https://huggingface.co/spaces/Rith-P/truthlens
- Code: https://github.com/Rithwik-P-26/truthlens
- A demo video may be added later; the README has a demo GIF.

---

## Inspiration

A fake "photo" is now one prompt away, and fakes show up in news feeds, insurance claims, marketplace listings and
scams. When we looked at existing detectors, two things stood out. First, they break as soon as a new generator is
released. Second, many of them score near-perfectly in papers for the wrong reason: in popular datasets the real
photos are JPEGs and the fakes are PNGs of fixed sizes, so a model can "detect AI" by reading the file format. We
wanted a detector whose numbers we could trust, and that admits when it does not know.

## What it does

Upload, paste or drop an image and TruthLens returns:

- **P(AI-generated)**, calibrated on held-out data;
- a verdict, **AI-generated, Real or Uncertain**: it abstains between 35% and 65%, and on images too small to judge;
- **evidence**:
  - the probability from each of its two detectors;
  - a heatmap of the regions that pushed the decision towards "AI";
  - the frequency spectrum of the analysed image;
  - 2–4 short reasons computed from measured quantities.

The example gallery deliberately includes two hard cases: a Midjourney 6 image it confidently gets wrong, and a real
DOCCI photo where it abstains. Batch mode scores up to 50 images and exports CSV. A "How it works" tab shows the evaluation
results, and a "Limitations & responsible use" tab explains what the tool cannot do.

## How we built it

**Data.** We used 24,840 images (dataset v2.1) from GenImage (via Tiny-GenImage) and OpenFake. The real photos come from ImageNet,
LAION, Pexels and DOCCI, and the fakes from more than 80 generators. We measured the shortcuts in the raw data: 100% of
GenImage reals were JPEG and 100% of its fakes PNG, and OpenFake fakes had about twice the resolution of the reals. We
removed these shortcuts with three steps:

- normalizing both classes identically: square crop, 224 px, JPEG q95, metadata stripped;
- removing near-duplicates with perceptual hashing, including any cluster that held both a real photo and a fake made
  from it;
- balancing real and AI images within every source pool.

A classifier on file metadata alone now scores AUC 0.577, close to chance.

**Content filtering.** We found explicit images in the web-scale source data and removed them with three automatic
signals: an NSFW image classifier, the Stable Diffusion safety checker on CLIP embeddings, and a prompt keyword
filter. A human then spot-checked the result. Real and AI images were filtered identically. 13 generators were
dropped (9 with at least 25% flagged images, plus 4 anime-style ones as a precaution), and the gaps were refilled
with clean images so the split sizes stayed the same.

**An honest split.** We held out 32 generators from unseen families (Midjourney 5/6/7, DALL-E 3, GPT-Image 1/1.5/2,
FLUX.2, Seedream, Nano-Banana, Imagen 4, Sora 2, Veo 3, ...), plus a real-photo source the model never saw (DOCCI). That
lets us measure both missed fakes and false alarms on unfamiliar data.

**Models.** Two complementary detectors:

- a **CLIP probe**: a linear classifier on frozen CLIP ViT-L/14 features from the final layer and 5 intermediate
  layers. It follows UnivFD (CVPR 2023) and RINE (ECCV 2024). We chose this variant by leave-one-generator-family-out
  validation, where it scored AUC 0.985 against 0.931 for the classic final-layer probe;
- an **artifact CNN**: an EfficientNet-B0 fine-tuned with JPEG, blur and rescale augmentation. Its class-activation
  map is an exact decomposition of its decision over image regions, which gives the heatmap.

Each detector is calibrated on validation data. The two are averaged with fixed equal weights, then calibrated again.
Nothing was tuned on the test sets. The app uses exactly the same preprocessing as the training data, so the reported
numbers describe what users get.

**Stack.** Python, PyTorch (Apple MPS for training; ZeroGPU on Hugging Face and CPU locally for serving), open_clip, timm, scikit-learn, Hugging Face
datasets, and Gradio on Hugging Face Spaces, with 14 pytest tests. We built TruthLens with Claude Code (Anthropic,
Claude Opus 5.5) as an AI coding assistant: a lead agent coordinated sub-agents for data, modelling, the app and the
docs. The human author chose the problem, directed the work and made the decisions.

## Challenges we ran into

- **Shortcut learning.** Our first data audit showed that file format, size or aspect ratio alone would separate the
  classes almost perfectly. Fixing the data mattered more than any model choice.
- **Generalization vs. in-distribution accuracy.** On seen generators everything looks great (test AUC 0.998). On unseen
  ones, the classic UnivFD-style probe drops to 0.854. Selecting by leave-one-family-out validation instead of ordinary
  validation pointed us to the multi-layer probe (0.932 unseen).
- **Calibration does not transfer.** Calibration error is 0.006 on seen generators but 0.168 on unseen ones. We report
  this rather than hide it, and use an "Uncertain" band.
- **Robustness trade-offs.** JPEG recompression hides traces, while blur and upscaling make real photos look fake. That
  is why the app refuses to judge images under 224 px.
- **Getting data within a weekend.** We streamed individual parquet row groups over HTTP instead of downloading
  multi-GB shards, and rejected datasets with unclear licenses.

## Accomplishments that we're proud of

- ROC-AUC 0.927 (95% CI 0.921–0.934) on 32 generators from families never seen in training, flagging 3.2% of real
  photos. In-distribution: 98.2% accuracy, AUC 0.998.
- Abstention that answers 94.1% of unseen-generator images, at 81.6% accuracy on those it answers.
- A dataset audit and data card that document every shortcut we found and how we removed it.
- Evidence that is computed, not decorative: the heatmap is an exact decomposition of the CNN's logit, and every
  explanation sentence comes from a measured quantity.
- Honesty in the product itself. The gallery shows a hard case it gets wrong, and the app tells users it catches only
  about 63% of fakes from brand-new generators.

## What we learned

- For this problem, evaluation design is the hard part. A detector is only as good as its results on the generators it
  has not seen.
- Measure your dataset before training on it: the cheapest features, like file format and size, can be the most
  predictive.
- Semantic features generalize better than pixel artifacts. Our CLIP probe reached 0.932 AUC on unseen generators and
  the artifact CNN 0.780. The CNN still helps under JPEG compression.
- Web-scale datasets need a content audit, not just a label audit. Public deepfake datasets contained explicit
  images, concentrated in a few fine-tuned generators. Filtering real and AI images with the same rule kept the
  labels unbiased. Retraining on the filtered data changed little: unseen AUC went from 0.935 to 0.927 and seen
  AUC stayed at 0.998.
- Calibration and abstention make a classifier usable by non-experts. A confident wrong answer is worse than
  "Uncertain".

## What's next for TruthLens

- Analyse full-resolution crops alongside the 224 px view, and add newer generators and phone photos to training.
- Detect blur and upscaling and abstain on them; recalibrate on a broader validation set.
- Combine with provenance signals (C2PA content credentials, watermarks) when they are present, and localize partial
  edits.
- A browser extension and an API for newsrooms and trust-and-safety teams.

## Built with

python, pytorch, open-clip, clip, timm, efficientnet, scikit-learn, numpy, pandas, pillow, hugging-face,
hugging-face-datasets, hugging-face-spaces, gradio, safetensors, matplotlib, pytest, playwright, apple-silicon-mps,
claude-code
