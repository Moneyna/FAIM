# FAIM and FAIM-Bench

This repository contains the anonymous release of **FAIM** and **FAIM-Bench**.

**FAIM** is a fidelity-aware LR/SR consistency assessment method. Given a low-resolution or degraded input image and a super-resolved output, FAIM predicts how faithfully the SR image preserves the visual evidence contained in the LR input. Unlike conventional perceptual quality metrics that mainly reward visual realism, FAIM focuses on **input consistency**, making it suitable for evaluating hallucination-prone image restoration and super-resolution methods.

**FAIM-Bench** is a benchmark for LR/SR fidelity assessment. It contains real-world LR images, SR results from multiple restoration methods, and human-annotated consistency scores.

---

## Overview

Super-resolution models can generate visually appealing details that are not necessarily supported by the LR input. FAIM is designed to measure this LR/SR faithfulness.

FAIM extracts interpretable consistency evidence from each LR/SR pair, including:

- low-frequency structural fidelity,
- semantic correspondence between LR and SR regions,
- region-conditioned texture consistency,
- optional diagnostic cues for weakly supported regions.

A lightweight learned fusion model maps these features to a final consistency score. Higher FAIM scores indicate stronger LR/SR consistency.

---

## Acknowledgement

Part of this repository builds on code from **INSID3: Training-Free In-Context Segmentation with DINOv3**:

```text
https://github.com/visinf/INSID3
```

We thank the authors for releasing their implementation. FAIM adapts the INSID3-style DINOv3-based self-mask extraction and dense correspondence components for LR/SR consistency assessment. Please also follow the license and usage terms of the original INSID3 repository.

---

## Training

To train the FAIM fusion model, run:

```bash
python train_consistency_fusion4.py  --config config4_infer.yaml --mode train
```

---

## Inference

After training, FAIM can be applied to LR/SR pairs using precomputed feature CSV files.

Example inference command for ablation variants:

```bash
for ab in wo_lowfreq wo_semantic wo_texture wo_unsupported fixed_sr_to_lr; do
python infer_list.py \
    --config outputs/ablations/${ab}/config_resolved.yaml \
    --train_script train_consistency_fusion4.py \
    --use_features_csv \
    --features_csv outputs/ablations/${ab}/features_${ab}.csv \
    --part part2 \
    --datasets RealDeg \
    --methods UPSR StableSR SeeSR CoSeR PASD DiT4SR HYPIR LucidFlux \
    --out_dir outputs/ablations/${ab}/infer
done
```

## Ablation Experiments

We provide an ablation script to train and export different FAIM variants.

### Train standard ablation variants

```bash
python ablation.py --config ablation.yaml --ablation full --mode train
python ablation.py --config ablation.yaml --ablation wo_lowfreq --mode train
python ablation.py --config ablation.yaml --ablation wo_semantic --mode train
python ablation.py --config ablation.yaml --ablation wo_texture --mode train
python ablation.py --config ablation.yaml --ablation wo_unsupported --mode train
python ablation.py --config ablation.yaml --ablation wo_residual --mode train
```

### Fixed-direction semantic matching ablation

The fixed SR-to-LR matching variant requires an export step before training:

```bash
python ablation.py --config ablation.yaml --ablation fixed_sr_to_lr --mode export
python ablation.py --config ablation.yaml --ablation fixed_sr_to_lr --mode train
```

## License

This repository is released for research purposes. Please respect the licenses and terms of use of the original datasets, pretrained models, third-party evaluation metrics, and the INSID3 codebase used by FAIM-Bench.

Parts of the implementation are adapted from:

```text
https://github.com/visinf/INSID3
```

Please refer to the INSID3 repository for its original copyright and license terms.
