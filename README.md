# Deep Learning with Attention Mechanism for Pediatric Appendicitis Detection on Plain Abdominal Radiographs

This repository contains the code for the study evaluating deep learning architectures and attention mechanisms for pediatric appendicitis classification from plain abdominal radiographs.

## Overview

A classification framework based on:
- **Preprocessing:** manual cropping + unsharp masking
- **Architectures compared:** EfficientNet-B0, ResNet50, DenseNet121, Swin-Tiny, ConvNeXt-Tiny
- **Attention mechanisms compared:** SE, CBAM, CA, PCCA
- **Evaluation:** 5-fold stratified cross-validation with bootstrap 95% CI
- **Interpretability:** Grad-CAM

Best model: **EfficientNet-B0 + SE** with aggregated AUC of 0.824 (95% CI: 0.781–0.863).

## Dataset

The dataset is not publicly available due to patient privacy and institutional ethical restrictions. Access may be requested from the corresponding author.

- 162 pediatric appendicitis cases (histopathologically confirmed)
- 206 control cases
- Single-center retrospective collection (2014–2024)

## Requirements

- Python 3.12
- PyTorch 2.x with CUDA
- torchvision
- scikit-learn
- pytorch-grad-cam
- numpy, pandas, matplotlib, PIL

## Reproducibility

All experiments use a fixed random seed of 42. Training was conducted with early stopping (patience=5 on validation loss) on a single NVIDIA Tesla T4 GPU.

## Citation

If you use this code, please cite:

> [Citation will be added upon publication]

## Contact

For data access requests or questions, please contact the corresponding author.

## License

MIT License (see LICENSE file).
