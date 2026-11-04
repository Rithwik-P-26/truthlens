# TruthLens submission checklist

Deadline: **2026-11-04 18:30 GMT+4 (8:00 PM IST)**. Aim to submit by 17:30 GMT+4.
Remaining placeholder (on purpose): `{{DEVPOST_URL}}` in README.md.
Find them with `grep -rn "{{" README.md DEVPOST.md LICENSE docs/`.

## Official requirements (checked against the files)

### README

- [x] Project title and **problem statement number (FT-02)** plus track: README header table
- [x] **Team name and member names**: TruthLens (solo), Rithwik Ponnana (@Rithwik-P-26), in the README header table
- [x] Problem description: README "The problem"
- [x] Proposed solution / approach: README "What TruthLens does", "How it works", "Data pipeline"
- [x] Tech stack and libraries: README "Tech stack"
- [x] Setup and installation (macOS/Linux, Python 3.11, uv or pip, venv): README "Quickstart"
- [x] Instructions to run locally: README "Run the app", "Tests", "Reproduce the results"
- [x] **Screenshots and demo GIF** (re-captured for model v2.1.0): `docs/demo.gif` (15 s, 6.5 MB) and 9 PNGs in
      `docs/screenshots/`. Regenerate with `scripts/capture_media.py`. All README image paths exist.
- [x] **Model accuracy/performance metrics**: README "Results at a glance" and "Results". Covers accuracy, F1, ROC-AUC
      with CI, ECE, detection rate/FPR, seen vs unseen, members vs baselines vs ablations, per generator, abstention,
      robustness and CPU latency, all from `reports/metrics.json` and `reports/RESULTS.md`.
- [x] **AI development tools disclosed**: README "Use of AI development tools" (Claude Code, Claude Opus 5.5, lead agent
      + sub-agents; the human's role)
- [x] Live demo link: https://huggingface.co/spaces/Rith-P/truthlens in README and DEVPOST (**not live until step 5 below**)
- [x] Responsible-AI disclaimers, limitations, privacy, licenses and the training-data content filter: README
      "Responsible AI" and "Content filtering (dataset v2.1)"; app tab "Limitations & responsible use"; disclaimer
      under every result

### Project

- [x] Python for all AI/ML parts (`src/truthlens/`, `scripts/`)
- [ ] Working demo: the Space runs the **real model**, not demo mode (no "Demo mode" banner). Locally verified: the
      screenshots and GIF use the real model (v2.1.0 on CPU).
- [x] Original work built during 2–4 November 2026; third-party datasets and pretrained models credited (README
      "Acknowledgements and citations", citations verified)
- [x] Code license: `LICENSE` (MIT, Rithwik Ponnana); non-commercial data, examples and weights noted in README and LICENSE
- [x] Tests: `.venv/bin/python -m pytest -q tests` (15 tests)

## Manual steps for the human (agents do not run git, gh or Hugging Face commands)

1. [ ] Read README "Use of AI development tools" and confirm the wording about your role.
2. [ ] Check `git status`. Nothing from `data/`, `logs/`, `models/cache/`, `space/`, `.claude/` or `TEAM_PLAN.md` should be
       staged (all are gitignored). `models/*.safetensors` (16.2 MB), `docs/demo.gif` (6.5 MB) and `docs/screenshots/`
       **should** be committed.
3. [ ] Commit and push to https://github.com/Rithwik-P-26/truthlens, and make the repository **public**.
4. [ ] Build the Space with `.venv/bin/python scripts/build_space.py` and check its warnings.
5. [ ] Upload `space/` to https://huggingface.co/spaces/Rith-P/truthlens.
6. [ ] Open the Space and wait for the build. The first start downloads the 1.7 GB CLIP weights. Test one AI example,
       one real example and the hard cases; note the latency shown under a result.
7. [x] Replace `{{SPACE_LATENCY}}` in README.md (2 places) with the measured Space latency, e.g. "about N ms per image".
8. [ ] Devpost:
       - create the project and paste the DEVPOST.md sections, tagline, elevator pitch and "Built with" tags;
       - add the GitHub and Space links;
       - upload `docs/demo.gif` and a few screenshots as the gallery;
       - select Track 1 / FT-02.
9. [ ] Put the Devpost project URL into README (`{{DEVPOST_URL}}`) and push again.
10. [ ] Submit before **18:30 GMT+4 on 4 Nov**. Then check the public project page, and that every link opens while
        logged out (repo public, Space public and running).

## Final review (Phase B)

- [x] Every number in README/DEVPOST matches the v2.1 `reports/metrics.json` / `RESULTS.md`, with 32 unseen / 38 seen
      generators (v1 numbers appear only in the v1 → v2.1 comparison) (3-decimal fractions in tables,
      one-decimal % in prose)
- [x] Model descriptions match `models/config.json` (CLIP ViT-L/14 multi-layer probe + EfficientNet-B0, equal-weight
      mean of Platt-calibrated logits, band 0.35–0.65, uncertain if under 224 px)
- [x] All README relative links and image paths exist locally; mermaid blocks use standard flowchart syntax (check
      their rendering on GitHub after pushing)
- [x] App text checked against the model in the screenshots (see the docs agent's Phase B report for notes)
