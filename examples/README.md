# TruthLens demo examples

Twelve images taken from the held-out `test` and `test_unseen` splits of `data/manifest.csv`, so the model never
trained on them. Every file is already in the pipeline's normalized form (224x224 RGB JPEG q95, no metadata). None
shows an identifiable real person, and all twelve pass the v2 content filter (`data/meta/nsfw_scan.csv`:
Falconsai p(nsfw) < 0.05, SD safety-checker score < 0, no prompt-keyword flag). All come from the current
(v2, content-filtered) `data/manifest.csv`.

- **10 regular examples** (6 AI-generated, 4 real), chosen after running the final detector: each is predicted
  correctly with at least 90% confidence by the v2.1 model. Five of the six AI images come from generators the model
  **never saw in training** (Veo 3 twice, GPT Image 1.5, Wan 2.5, Seedream v4.5); one comes from a generator family
  that it did see (Stable Diffusion 1.5).
- **2 hard cases**, shown in the app on purpose so the demo is honest about its limits: a confident miss (a
  Midjourney 6 image the detector calls real; a weak unseen family, about 18% detected) and an abstention (a real
  DOCCI photo that lands in the 35–65% uncertainty band, P(AI) about 0.49).

`labels.csv` (`file,label,source,split,hard_case`) gives the ground truth the app displays after analysing an
example, in gallery order.

| File | Label | Generator / source | Split | Dataset | License | Manifest path |
|---|---|---|---|---|---|---|
| `ai_veo_3_volcano_flowerpot.jpg` | AI | veo_3 | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-NC-4.0 (OpenFake) | `data/images/openfake/ai_veo_3_7044fbd19825d5b9.jpg` |
| `real_pexels_tower_night.jpg` | Real | pexels | test | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | Pexels License (free to use), redistributed via OpenFake (CC-BY-NC-4.0) | `data/images/openfake/real_pexels_c5d4e2e466fc8451.jpg` |
| `ai_gpt_image_1_5_frog_knight.jpg` | AI | gpt_image_1_5 | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-NC-4.0 (OpenFake) | `data/images/openfake/ai_gpt_image_1_5_333e91fd372ad0dd.jpg` |
| `real_docci_bicycle.jpg` | Real | docci | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-4.0 (DOCCI, Google), redistributed via OpenFake (CC-BY-NC-4.0) | `data/images/openfake/real_docci_e2b3d8d72c93cc04.jpg` |
| `ai_wan_video_2_5_snow_fox.jpg` | AI | wan_video_2_5 | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-NC-4.0 (OpenFake) | `data/images/openfake/ai_wan_video_2_5_2b017805604eec31.jpg` |
| `real_pexels_shark.jpg` | Real | pexels | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | Pexels License (free to use), redistributed via OpenFake (CC-BY-NC-4.0) | `data/images/openfake/real_pexels_f2a08ffe05a84c67.jpg` |
| `ai_veo_3_stone_villa.jpg` | AI | veo_3 | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-NC-4.0 (OpenFake) | `data/images/openfake/ai_veo_3_1f0ba485c665700d.jpg` |
| `real_imagenet_samoyed.jpg` | Real | imagenet | test_unseen | [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) | ImageNet terms of access (non-commercial research); GenImage via Tiny-GenImage (CC-BY-NC-SA-4.0) | `data/images/genimage/real_imagenet_39cecdd7626f88c5.jpg` |
| `ai_seedream_v4_5_aurora_knight.jpg` | AI | seedream_v4_5 | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-NC-4.0 (OpenFake) | `data/images/openfake/ai_seedream_v4_5_29dd48a30a07944b.jpg` |
| `ai_sd15_red_panda.jpg` | AI | sd15 | test | [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) | CC-BY-NC-SA-4.0 (GenImage via Tiny-GenImage) | `data/images/genimage/ai_sd15_a07574c9900b67f9.jpg` |
| `hard_ai_midjourney_6_canoe_lagoon.jpg` | AI (hard case) | midjourney_6 | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-NC-4.0 (OpenFake) | `data/images/openfake/ai_midjourney_6_dbf39c39332e237a.jpg` |
| `hard_real_docci_hyacinths.jpg` | Real (hard case) | docci | test_unseen | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) | CC-BY-4.0 (DOCCI, Google), redistributed via OpenFake (CC-BY-NC-4.0) | `data/images/openfake/real_docci_3e50fe4c7a7b326c.jpg` |

License notes: these files are redistributed for non-commercial demo and research use only, with attribution to
OpenFake (Livernoche et al., ComplexDataLab, CC BY-NC 4.0), GenImage (Zhu et al., CC BY-NC-SA 4.0, via
TheKernel01/Tiny-GenImage), DOCCI (Google, CC BY 4.0), Pexels (Pexels License) and ImageNet (terms of access,
non-commercial research).
