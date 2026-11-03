# TruthLens results

_Generated 2026-11-03T11:11:49+0400 by `scripts/eval_models.py` from `reports/metrics.json`._

## Protocol

* Members are trained on TRAIN. Calibration, the ensemble and the uncertainty band are fitted on VAL. The CLIP probe variant is chosen by leave-one-family-out AUC on TRAIN/VAL. Nothing is tuned on a test split.
* TEST shares generators with training. TEST-UNSEEN holds 34 generators from families never seen in training or selection, plus new real-photo domain(s) (docci), so we report the detection rate on fakes and the false-positive rate on reals separately.
* One preprocessing policy end to end: every image (train, val, test and every upload) is centre-cropped to a square, Lanczos-resized to 224x224 and re-encoded as JPEG q95, exactly as in `normalize_image`. This discards some high-frequency evidence in large photos, but it makes these test numbers representative of the deployed app.
* Family closeness remains between some seen and unseen generators (Flux.1 seen vs Flux.2 unseen; SDXL fine-tunes, including an Illustrious-based one, vs unseen Illustrious), so a few 'unseen' families are near neighbours of training data.

## Headline (deployed ensemble)

| split | n | accuracy | F1 | ROC-AUC (95% CI) | ECE | detection rate (fakes) | FPR (reals) | coverage | selective acc. |
|---|---|---|---|---|---|---|---|---|---|
| test | 2384 | 0.982 | 0.982 | 0.998 (0.998-0.999) | 0.006 | 0.977 | 0.013 | 0.983 | 0.988 |
| test_unseen | 5326 | 0.811 | 0.776 | 0.935 (0.929-0.942) | 0.158 | 0.655 | 0.033 | 0.945 | 0.828 |

Coverage and selective accuracy refer to images that get a verdict; the rest are labelled Uncertain (calibrated confidence < 0.65).

## Members, ensemble and baselines

| system | kind | test acc | test AUC | test F1 | test ECE | unseen acc | unseen AUC | unseen F1 | unseen ECE | unseen det. rate | unseen FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TruthLens ensemble (CLIP probe + CNN artifacts) | deployed | 0.982 | 0.998 | 0.982 | 0.006 | 0.811 | 0.935 | 0.776 | 0.158 | 0.655 | 0.033 |
| CLIP probe (member) | member | 0.986 | 0.999 | 0.986 | 0.008 | 0.822 | 0.940 | 0.792 | 0.145 | 0.678 | 0.035 |
| CNN artifacts (member) | member | 0.862 | 0.933 | 0.861 | 0.019 | 0.713 | 0.797 | 0.666 | 0.131 | 0.572 | 0.146 |
| UnivFD-style probe, CLIP ViT-B/16 final embedding | baseline | 0.861 | 0.937 | 0.861 | 0.019 | 0.768 | 0.850 | 0.751 | 0.068 | 0.701 | 0.165 |
| UnivFD-style probe, CLIP ViT-L/14 final embedding | baseline | 0.898 | 0.964 | 0.899 | 0.018 | 0.780 | 0.866 | 0.761 | 0.097 | 0.703 | 0.143 |
| Spectral LR (azimuthal power spectrum + residual peaks) | baseline | 0.679 | 0.743 | 0.662 | 0.041 | 0.589 | 0.649 | 0.524 | 0.083 | 0.452 | 0.275 |
| File-metadata shortcut audit (format/size/JPEG quality LR) | baseline | 0.580 | 0.617 | 0.594 | 0.034 | 0.547 | 0.567 | 0.547 | 0.027 | 0.547 | 0.453 |
| Ensemble before calibration (mean raw logit) | ablation | 0.964 | 0.994 | 0.964 | 0.014 | 0.787 | 0.907 | 0.744 | 0.169 | 0.619 | 0.045 |
| Fast fallback ensemble (CLIP ViT-B/16 probe + CNN) | ablation | 0.966 | 0.995 | 0.966 | 0.010 | 0.802 | 0.909 | 0.767 | 0.152 | 0.652 | 0.048 |
| Stacked-LR ensemble (weights fit on VAL) | ablation | 0.987 | 0.999 | 0.987 | 0.006 | 0.823 | 0.941 | 0.792 | 0.154 | 0.673 | 0.027 |
| Ensemble + spectral member | ablation | 0.980 | 0.998 | 0.980 | 0.006 | 0.815 | 0.937 | 0.782 | 0.152 | 0.662 | 0.032 |
| Multi-layer CLIP ViT-B/16 probe (final + intermediate CLS) | ablation | 0.969 | 0.995 | 0.969 | 0.010 | 0.798 | 0.909 | 0.763 | 0.147 | 0.650 | 0.054 |

**Shortcut audit.** A logistic regression on file metadata alone (format, size, JPEG quality, bits per pixel, EXIF) reaches ROC-AUC 0.617 on test and 0.567 on test-unseen after normalisation (raw data: format alone separated the classes). The remaining signal is mostly JPEG bits per pixel, which reflects image texture.

## Per source, test_unseen (deployed ensemble)

Accuracy is the detection rate for AI sources and the specificity (1 - FPR) for real sources.

| source | label | n | accuracy | ROC-AUC vs reals | mean P(AI) |
|---|---|---|---|---|---|
| imagenet | real | 1760 | 0.993 | - | 0.013 |
| docci | real | 453 | 0.885 | - | 0.128 |
| laion | real | 252 | 0.933 | - | 0.073 |
| pexels | real | 198 | 0.975 | - | 0.043 |
| midjourney_v5 | AI | 800 | 0.604 | 0.940 | 0.594 |
| z_image_turbo | AI | 348 | 0.753 | 0.966 | 0.738 |
| flux_2_klein_9b | AI | 285 | 0.218 | 0.797 | 0.232 |
| illustrious | AI | 214 | 0.991 | 0.999 | 0.979 |
| gpt_image_1_5 | AI | 165 | 0.752 | 0.958 | 0.719 |
| midjourney_7 | AI | 116 | 0.698 | 0.952 | 0.666 |
| veo_3 | AI | 68 | 0.926 | 0.993 | 0.906 |
| ideogram_3_0 | AI | 47 | 0.702 | 0.971 | 0.718 |
| flux_2_dev | AI | 43 | 0.930 | 0.981 | 0.886 |
| seedream_v4_0 | AI | 43 | 0.791 | 0.970 | 0.753 |
| nano_banana_2 | AI | 38 | 0.684 | 0.923 | 0.610 |
| dalle_3 | AI | 37 | 0.730 | 0.954 | 0.713 |
| flux_2_klein_4b | AI | 36 | 0.083 | 0.720 | 0.112 |
| midjourney_6 | AI | 36 | 0.222 | 0.845 | 0.285 |
| wan_video_2_5 | AI | 34 | 0.941 | 0.993 | 0.940 |
| recraft_v3 | AI | 32 | 0.219 | 0.790 | 0.195 |
| seedream_v4_5 | AI | 32 | 0.688 | 0.946 | 0.654 |
| studio_ghibli_wan2_1_t2v_14b_v1_0 | AI | 30 | 0.967 | 0.991 | 0.930 |
| wan_video_2_2 | AI | 30 | 0.933 | 0.995 | 0.926 |
| imagen_4_0 | AI | 28 | 0.786 | 0.982 | 0.784 |
| flux_2_klein_base_4b | AI | 27 | 0.889 | 0.991 | 0.873 |
| gpt_image_1 | AI | 23 | 0.652 | 0.941 | 0.680 |
| gpt_image_2 | AI | 19 | 0.632 | 0.959 | 0.644 |
| sora_2 | AI | 19 | 0.684 | 0.880 | 0.652 |
| recraft_v2 | AI | 16 | 0.625 | 0.979 | 0.683 |
| lumina_17_2_25 | AI | 15 | 0.867 | 0.990 | 0.861 |
| ernie_image_turbo | AI | 14 | 0.714 | 0.979 | 0.760 |
| ideogram_2_0 | AI | 13 | 0.615 | 0.930 | 0.560 |
| frames_23_1_25 | AI | 12 | 0.750 | 0.981 | 0.721 |
| seedream_v5_0 | AI | 11 | 0.909 | 0.989 | 0.833 |
| aurora_20_1_25 | AI | 10 | 0.600 | 0.954 | 0.582 |
| halfmoon_4_4_25 | AI | 8 | 0.625 | 0.966 | 0.595 |
| nano_banana_pro | AI | 8 | 0.750 | 0.970 | 0.779 |
| ernie_image | AI | 6 | 0.833 | 0.974 | 0.794 |

## Per source, test (deployed ensemble)

Accuracy is the detection rate for AI sources and the specificity (1 - FPR) for real sources.

| source | label | n | accuracy | ROC-AUC vs reals | mean P(AI) |
|---|---|---|---|---|---|
| imagenet | real | 982 | 0.992 | - | 0.016 |
| laion | real | 135 | 0.956 | - | 0.075 |
| pexels | real | 75 | 0.987 | - | 0.034 |
| adm | AI | 197 | 1.000 | 1.000 | 0.997 |
| sd15 | AI | 197 | 0.985 | 0.999 | 0.983 |
| glide | AI | 196 | 0.995 | 1.000 | 0.993 |
| vqdm | AI | 196 | 0.985 | 0.998 | 0.970 |
| wukong | AI | 196 | 0.990 | 0.999 | 0.976 |
| cyberrealistic_xl_v9_0 | AI | 6 | 1.000 | 0.998 | 0.955 |
| flux_mvc5000 | AI | 6 | 1.000 | 0.996 | 0.900 |
| qwen_image | AI | 6 | 0.333 | 0.957 | 0.462 |
| sdxl_turbo | AI | 6 | 1.000 | 0.997 | 0.878 |
| stable_diffusion_xl_base_1_0 | AI | 6 | 1.000 | 0.999 | 0.984 |
| anima_preview | AI | 5 | 1.000 | 0.997 | 0.901 |
| anima_preview2 | AI | 5 | 1.000 | 0.999 | 0.975 |
| animagine_xl_3_0 | AI | 5 | 1.000 | 1.000 | 1.000 |
| animagine_xl_3_1 | AI | 5 | 1.000 | 1.000 | 1.000 |
| animagine_xl_4_0 | AI | 5 | 1.000 | 1.000 | 0.999 |
| cyberrealistic_pony_v15_0 | AI | 5 | 1.000 | 1.000 | 0.999 |
| cyberrealistic_pony_v16_0 | AI | 5 | 1.000 | 1.000 | 1.000 |
| flux_1_1_pro | AI | 5 | 1.000 | 0.999 | 0.988 |
| flux_realism | AI | 5 | 0.600 | 0.986 | 0.683 |
| grok_2_image_1212 | AI | 5 | 0.800 | 0.994 | 0.831 |
| hidream_i1_full | AI | 5 | 0.800 | 0.995 | 0.866 |
| kolors_v1_0 | AI | 5 | 1.000 | 0.999 | 0.991 |
| mystic | AI | 5 | 1.000 | 0.998 | 0.977 |
| novaanimexl_ilv140 | AI | 5 | 1.000 | 1.000 | 1.000 |
| one_obsession_17_red_sdxl | AI | 5 | 0.800 | 0.997 | 0.891 |
| sd_1_5 | AI | 5 | 1.000 | 0.995 | 0.847 |
| sd_1_5_epicdream | AI | 5 | 0.800 | 0.995 | 0.857 |
| sd_2_1 | AI | 5 | 1.000 | 0.998 | 0.938 |
| sd_3_5 | AI | 5 | 0.600 | 0.990 | 0.737 |
| sd_xl_v1_0_vae_fix | AI | 5 | 1.000 | 0.999 | 0.988 |
| sdxl | AI | 5 | 1.000 | 0.999 | 0.988 |
| sdxl_1_0 | AI | 5 | 1.000 | 1.000 | 0.999 |
| sdxl_realvis_v5 | AI | 5 | 0.600 | 0.985 | 0.680 |
| stable_diffusion_v1_5 | AI | 5 | 1.000 | 1.000 | 0.999 |
| chroma | AI | 4 | 1.000 | 0.996 | 0.895 |
| cyberrealistic_pony_v14_0 | AI | 4 | 1.000 | 0.999 | 0.983 |
| cyberrealistic_pony_v14_1 | AI | 4 | 1.000 | 0.999 | 0.988 |
| flux_1_dev | AI | 4 | 0.500 | 0.986 | 0.659 |
| flux_1_schnell | AI | 4 | 1.000 | 1.000 | 1.000 |
| flux_amateursnapshotphotos | AI | 4 | 1.000 | 0.999 | 0.998 |
| imagen_3_0_002 | AI | 4 | 0.750 | 0.978 | 0.658 |
| playground_v2_5_1024px_aesthetic | AI | 4 | 1.000 | 0.999 | 0.996 |
| realistic_vision_v5_1_novae | AI | 4 | 1.000 | 1.000 | 1.000 |
| sd_1_5_dreamshaper | AI | 4 | 0.750 | 0.993 | 0.818 |
| sd_turbo | AI | 4 | 1.000 | 1.000 | 1.000 |
| sdxl_epic_realism | AI | 4 | 1.000 | 0.998 | 0.926 |
| sdxl_juggernaut | AI | 4 | 1.000 | 0.997 | 0.910 |
| sdxl_touchofrealism | AI | 4 | 0.750 | 0.997 | 0.875 |
| stable_diffusion_v1_4 | AI | 4 | 1.000 | 0.997 | 0.895 |

## Robustness (deployed ensemble, perturbation then the deployed normalisation)

Subsample: 1000 images per split, stratified by class and source.

| perturbation | unseen acc | unseen AUC | unseen det. rate | unseen FPR | test acc | test AUC |
|---|---|---|---|---|---|---|
| clean | 0.828 | 0.909 | 0.702 | 0.060 | 0.954 | 0.994 |
| jpeg_q90 | 0.732 | 0.852 | 0.448 | 0.016 | 0.858 | 0.962 |
| jpeg_q75 | 0.692 | 0.833 | 0.372 | 0.022 | 0.824 | 0.956 |
| jpeg_q50 | 0.742 | 0.814 | 0.581 | 0.114 | 0.874 | 0.933 |
| downscale_0.5x | 0.630 | 0.879 | 0.973 | 0.676 | 0.637 | 0.977 |
| blur_sigma1 | 0.673 | 0.874 | 0.944 | 0.568 | 0.709 | 0.978 |

## CPU latency

torch.set_num_threads(2) on arm64 with BLAS=accelerate. Apple AMX makes ViT GEMMs much faster than on a 2-vCPU x86 Space; Space numbers must be measured there (predict() returns latency_ms).

| component | median ms | p90 ms |
|---|---|---|
| normalize_224x224 | 1 | 1 |
| clip_224x224 | 154 | 155 |
| cnn_224x224 | 65 | 65 |
| spectral_features_224x224 | 4 | 4 |
| display_spectrum_224x224 | 1 | 1 |
| predict_total_224x224 | 224 | 230 |
| normalize_1024x1024 | 3 | 3 |
| clip_1024x1024 | 154 | 154 |
| cnn_1024x1024 | 64 | 64 |
| spectral_features_1024x1024 | 4 | 4 |
| display_spectrum_1024x1024 | 1 | 1 |
| predict_total_1024x1024 | 243 | 246 |
| normalize_4032x3024 | 13 | 14 |
| clip_4032x3024 | 154 | 154 |
| cnn_4032x3024 | 63 | 64 |
| spectral_features_4032x3024 | 4 | 4 |
| display_spectrum_4032x3024 | 1 | 1 |
| predict_total_4032x3024 | 302 | 306 |
| fast_predict_total_224x224 | 109 | 110 |
| fast_predict_total_1024x1024 | 128 | 129 |
| fast_predict_total_4032x3024 | 187 | 189 |

## CLIP probe selection (leave-one-family-out on TRAIN/VAL)

Deployed: `clip_vitl14_multi`; rule: highest leave-one-family-out mean AUC on TRAIN->VAL (rounded to 3 dp; tie-break: VAL AUC).

| head | dim | CV AUC (train) | VAL AUC | LOFO mean AUC | LOFO min AUC |
|---|---|---|---|---|---|
| clip_vitl14_final | 768 | 0.9612 | 0.9657 | 0.9333 | 0.8711 |
| clip_vitl14_multi | 5888 | 0.9988 | 0.9985 | 0.9874 | 0.9589 |
| clip_vitb16_final | 512 | 0.9408 | 0.9314 | 0.8852 | 0.7713 |
| clip_vitb16_multi | 4352 | 0.9955 | 0.9952 | 0.9639 | 0.8665 |
| freq | 67 | 0.7505 | 0.7485 | 0.7078 | 0.5590 |
| meta | 12 | 0.6113 | 0.6047 | 0.5831 | 0.4004 |

## Caveats

* All sources are non-commercial research datasets. About 85% of TRAIN is GenImage (ImageNet content, 2022-era generators); modern generators contribute about 1.2k training images, so they are up-weighted by source pool (alpha = 0.5, fixed in advance).
* Many unseen generators have fewer than 50 images, so their per-source rates have wide uncertainty.
* Known failure modes from the robustness table: blur and upsampling make real photos look generated (ranking AUC stays high but the 0.5 threshold produces many false positives), and JPEG recompression hides generator traces (more missed fakes). Inputs smaller than 224 px are therefore answered with 'Uncertain' in the app; blur cannot be detected reliably and remains a limitation.
* The pre-registered ensemble (equal-weight calibrated logits) is slightly below the CLIP probe alone on clean test sets, but more robust under JPEG compression. The stacked-LR row is shown for comparison only and was not selected after seeing test results.
* Robustness numbers use a stratified subsample and perturb the 224 px normalised image. An upload degraded at full resolution is partly 'cleaned' by the downscale, so these numbers are conservative.
* CPU latency was measured on Apple silicon (AMX-accelerated matrix multiplies); a 2-vCPU x86 Space is slower. `predict()` reports `latency_ms` live.
* Calibration is fitted on VAL (seen generators). On unseen generators the probabilities are less reliable; see the test-unseen ECE and the reliability diagram.

## Figures

![cnn_training_curves](figures/cnn_training_curves.png)
![confusion_matrices](figures/confusion_matrices.png)
![per_generator_unseen_auc](figures/per_generator_unseen_auc.png)
![reliability_diagram](figures/reliability_diagram.png)
![risk_coverage](figures/risk_coverage.png)
![robustness](figures/robustness.png)
![roc_curves](figures/roc_curves.png)
