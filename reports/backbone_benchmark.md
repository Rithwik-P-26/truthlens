# Backbone benchmark (Phase A)

## CLIP image towers (224 px)

| backbone | params (M) | GFLOPs | CPU ms (2 thr, bs1) | MPS img/s fp32 | MPS img/s fp16 | fp16 min cos |
|---|---|---|---|---|---|---|
| clip_vitb32 (ViT-B-32/openai) | 87.8 | 8.8 | 26.9 | 98.0 | 274.3 | 0.999996 |
| clip_vitb16 (ViT-B-16/openai) | 86.2 | 35.1 | 115.0 | 28.9 | 84.8 | 0.999996 |
| clip_vitl14 (ViT-L-14/openai) | 304.0 | 162.0 | 362.7 | 6.3 | 19.5 | 0.999996 |

## CNN artifact detectors

| arch | params (M) | MB fp32 | GFLOPs 512 | CPU ms 512 | CPU ms 1024 | MPS train img/s 256 (fp32/fp16/bf16) |
|---|---|---|---|---|---|---|
| efficientnet_b0 | 4.0 | 15.3 | 4.0 | 222.3 | 519.4 | 42.9/48.4/44.0 |
| resnet18 | 11.2 | 42.6 | 18.9 | 74.1 | 347.6 | 31.8/45.0/48.1 |
| resnet50 | 23.5 | 89.7 | 42.7 | 321.1 | 1132.1 | 15.9/22.1/21.9 |
| convnext_tiny | 27.8 | 106.1 | 46.5 | 4332.5 | 6426.2 | 3.9/4.6/3.1 |

Spectral features: 4 ms; display spectrum: 1 ms (224x224 normalised image, CPU).

* CPU = torch.set_num_threads(2) on Apple M5 (Accelerate/AMX GEMMs: optimistic vs a 2-vCPU x86 Space).
* CNN rows were measured while the machine was overloaded by parallel data processing (load average 45-110; convnext_tiny at the peak), so their CPU/MPS numbers are pessimistic and only comparable within a row. EfficientNet-B0 was later measured at 61-66 ms per 224 px image on 2 idle CPU threads.
* CNN training throughput is batch 64 at 256 px; production training ran at 224 px.
