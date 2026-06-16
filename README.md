<div align="center">

# GFL-KA: GFL with Kurtosis-guided Attention for Wind Turbine Blade Defect Detection

[![Paper](https://img.shields.io/badge/Paper-WindBlade--30K%20%2F%20GFL--KA-blue)](https://github.com/SongYe-NUAA/GFL-KA)
[![Dataset](https://img.shields.io/badge/Dataset-WindBlade--30K-green)](dataset.txt)
[![Framework](https://img.shields.io/badge/MMDetection-3.x-orange)](https://github.com/open-mmlab/mmdetection)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

<p align="center">
  <b>Ye Song</b> · <b>Yiquan Wu</b> · <b>Yuqi Liu</b> · <b>Hanwen Yu</b><br/>
  <a href="https://github.com/SongYe-NUAA/GFL-KA">GitHub</a> · <a href="dataset.txt">Dataset</a>
</p>
</div>

## Abstract

Unmanned aerial vehicle (UAV)-based inspection has become a mainstream paradigm for wind turbine blade maintenance, yet practical deployments remain limited by irregular defect boundaries, complex on-site interference, the lack of large-scale standardized benchmarks, and insufficient localization accuracy of existing detectors. To address these challenges, we present three hierarchical contributions. First, we construct and release **WindBlade-30K**, the largest public benchmark for wind turbine blade defect detection to date, containing 5,168 high-resolution aerial images and 30,437 annotated instances across 17 defect categories under unified COCO-style supervision. Second, we propose the **Kurtosis-Guided Quality Predictor (KGQP)**, which augments Generalized Focal Loss v2 (GFLv2) with high-order kurtosis statistics and a lightweight Dual-Pooling Channel Attention (DPCA) mechanism for more reliable bounding-box quality estimation. Third, we introduce a **Gradient Balance Factor (GBF)** into the Quality Focal Loss, which alleviates gradient imbalance during early training without distorting the original IoU supervision. Extensive experiments on WindBlade-30K show that the proposed **GFL-KA** achieves an AP of **0.369**, a 4.2% absolute improvement over the GFLv2 baseline, while outperforming 34 one-stage, two-stage, and YOLO-series detectors and preserving a lightweight architecture suitable for UAV inspection workflows.

## Highlights

- **Benchmark dataset**: WindBlade-30K is the largest open dataset for drone-based wind turbine blade defect detection, with standardized COCO annotations and 17 defect categories.
- **Methodological contribution**: GFL-KA replaces vanilla GFLv2 localization quality estimation with a joint local-statistics, global-morphology, and direction-aware representation.
- **Training stability**: GBF stabilizes Quality Focal Loss training by asymmetric gradient modulation, reducing early-stage loss oscillation and improving convergence on medium-quality samples.
- **Strong empirical results**: GFL-KA achieves 0.369 AP, 0.565 AP50, 0.402 AP75, 0.533 AR, and 0.674 F1-score on WindBlade-30K with a ResNet-50 backbone.
- **Practical baseline coverage**: YOLO-series experiments are additionally provided and evaluated on the Ultralytics platform for industrial comparability.

## Dataset

The dataset is hosted externally because of its size. Please download it before training or evaluation.

- **Dataset name**: WindBlade-30K
- **Archive**: `WindBlade-30K.zip`
- **Download link**: see `dataset.txt`

After extracting the archive, the expected layout is:

```text
WindBlade-30K/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images/
    ├── ...
    └── ...
```

### Dataset Statistics

| Split | Images | Instances | Small | Medium | Large |
| --- | --- | --- | --- | --- | --- |
| Train | 3,618 | 21,627 | 8,558 | 7,710 | 5,359 |
| Val | 775 | 4,459 | 1,853 | 1,397 | 1,123 |
| Test | 775 | 4,437 | 1,739 | 1,572 | 1,126 |

### Defect Categories

`hole`, `leaf-opex`, `corrosion`, `stain`, `corrosion-pit`, `lightning-arrester-miss`, `degumming`, `repair`, `lightning-arrester`, `teeth`, `demould`, `painting-peel-off`, `sign`, `crack`, `dirt`, `swell`, `oil`

### Data Preparation

1. Download `WindBlade-30K.zip` using the link in `dataset.txt`.
2. Extract the archive.
3. Place the dataset under the repository root or update `data_root` in the experiment config to point to the extracted path.

### Data Sources and Citations

WindBlade-30K is built upon multiple publicly released wind turbine blade defect datasets after systematic collection, annotation correction, class normalization, and COCO-style unification. When you use this dataset or any downstream research based on it, please also cite the following original sources as appropriate.

- **DTU dataset**
```bibtex
@article{shihavuddin2019wind,
  title   = {Wind Turbine Surface Damage Detection by Deep Learning Aided Drone Inspection Analysis},
  author  = {Shihavuddin, A. and Chen, X. and Fedorov, V. and Christensen, N. N. and Riis, A. B. and Branner, K. and Dahl, A. B. and Paulsen, R. R.},
  journal = {Energies},
  volume  = {12},
  number  = {4},
  pages   = {676},
  year    = {2019}
}
```

- **YAWTSD dataset**
```bibtex
@article{sarkar2021wind,
  title   = {Wind Turbine Blade Structural State Evaluation by Hybrid Object Detector Relying on Deep Learning Models},
  author  = {Sarkar, D. and Gunturi, S. K.},
  journal = {Journal of Ambient Intelligence and Humanized Computing},
  year    = {2021}
}
```

- **Blade30 dataset**
```bibtex
@article{yang2023accurate,
  title   = {Towards Accurate Image Stitching for Drone-Based Wind Turbine Blade Inspection},
  author  = {Yang, C. and Liu, X. and Zhou, H. and Ke, Y. and See, J.},
  journal = {Renewable Energy},
  volume  = {203},
  pages   = {267--279},
  year    = {2023}
}
```

- **WCVP dataset**
```bibtex
@misc{roboflow2023wcvp,
  title   = {Wind Turbine Computer Vision Project},
  author  = {Tanishka},
  year    = {2023},
  howpublished = {\url{https://universe.roboflow.com/tanishka-p3e2d/windturbine-dmyul}}
}
```

- **WTBSDD dataset**
```bibtex
@misc{zhaowenhai2023wtbsdd,
  title   = {Wind Turbine Blade Surface Defect Dataset},
  author  = {Zhaowenhai},
  year    = {2023},
  howpublished = {\url{https://github.com/zhaowenhai2023/Wind-turbine-blade-surface-defect-dataset}}
}
```

## Method Overview

GFL-KA is built upon GFLv2 and focuses on improving localization quality estimation for dense defect detection under challenging UAV imaging conditions.

### Kurtosis-Guided Quality Predictor

KGQP refines the original GFLv2 quality predictor with three cooperative mechanisms:

- **Local statistical feature optimization**: retain the discriminative top-order bounding-box distribution statistics while fusing redundant high-order terms, reducing input dimension without losing localization cues.
- **Kurtosis-Guided Module (KGM)**: characterize global distribution morphology by high-order kurtosis statistics, adaptively modulated via learnable exponential scaling to stabilize training while remaining sensitive to boundary ambiguity.
- **Dual-Pooling Channel Attention (DPCA)**: fuse global average-pooled and max-pooled channel statistics to assign direction-aware importance across bounding-box edges and feature channels.

### Gradient Balance Factor

GBF acts directly on the Quality Focal Loss weight term. For under-confident and over-confident predictions, it applies asymmetric penalties with coefficients `δ_low = 0.25` and `δ_over = 0.1`. This preserves the original IoU target, suppresses extreme gradient outliers, and avoids uniform gradient clipping or label smoothing.

## Main Results on WindBlade-30K

ResNet-50 backbone. Evaluation follows COCO-style object detection metrics.

| Model | Params(M) | GFLOPs | AP | AP50 | AP75 | AR | AR50 | AR75 | F1-Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GFLv2 | 32.30 | 144.82 | 0.327 | 0.497 | 0.357 | 0.513 | 0.802 | 0.542 | 0.614 |
| **GFL-KA** | **31.12** | **127.08** | **0.369** | **0.565** | **0.402** | **0.533** | **0.835** | **0.555** | **0.674** |

The result table in the paper covers a broader comparison with 34 detectors, including one-stage detectors such as ATSS, DDOD, FCOS, TOOD, VFNet, Reppoints, and YOLO-series baselines (YOLOv9, YOLOv10, YOLOv11, YOLOv12), as well as two-stage detectors such as Faster R-CNN, Cascade R-CNN, Grid R-CNN, Libra R-CNN, and Dynamic R-CNN. GFL-KA achieves the best overall detection accuracy and recall while remaining lighter than the original GFLv2 baseline.

## Installation

This repository is based on MMDetection 3.x. Please use a compatible environment to avoid import errors.

```bash
git clone https://github.com/SongYe-NUAA/GFL-KA.git
cd GFL-KA

# Recommended: install via OpenMMLab MIM
pip install -U openmim
mim install -e .

# If you build manually, ensure the following runtime constraints are met
# PyTorch >= 1.8
# MMEngine >= 0.7.1, < 1.0.0
# MMCV >= 2.0.0rc4, < 2.2.0
```

## Quick Start

### Dataset Setup

Download the dataset using the link in `dataset.txt`, then verify the layout:

```bash
ls ../WindBlade-30K/annotations/train.json
ls ../WindBlade-30K/annotations/val.json
ls ../WindBlade-30K/annotations/test.json
```

### Train GFL-KA

```bash
python tools/train.py \
  configs/windturbine/windturbine_gfl-ka_r50.py \
  --work-dir runs/gfl-ka_resnet50
```

### Resume from a Checkpoint

```bash
python tools/train.py \
  configs/windturbine/windturbine_gfl-ka_r50.py \
  --work-dir runs/gfl-ka_resnet50 \
  --resume
```

### Evaluate GFL-KA

```bash
python tools/test.py \
  configs/windturbine/windturbine_gfl-ka_r50.py \
  runs/gfl-ka_resnet50/epoch_36.pth \
  --show-dir runs/gfl-ka_resnet50/vis
```

## YOLO-series Experiments

YOLO-series experiments are additionally provided for industrial baseline comparison and were evaluated on the Ultralytics platform. The repository includes MMDetection-style YOLO configs, and the same WindBlade-30K COCO annotations can be reused for YOLO-based training and benchmarking when required.

If you reproduce YOLO experiments outside MMDetection, use the WindBlade-30K COCO files under:

```text
../WindBlade-30K/annotations/train.json
../WindBlade-30K/annotations/val.json
../WindBlade-30K/annotations/test.json
```

For MMDetection-side YOLO baselines, use the dedicated configs under `configs/windturbine/` and adapt `data_root` to your dataset path.

## Repository Structure

```text
GFL-KA/
├── README.md
├── dataset.txt
├── requirements.txt
├── requirements/
│   ├── runtime.txt
│   ├── build.txt
│   └── optional.txt
├── mmdet/
│   └── models/
│       └── dense_heads/
│           └── gfocal_head.py
├── configs/
│   ├── windturbine/
│   │   └── windturbine_gfl-ka_r50.py
│   └── _base_/
│       └── datasets/
│           └── coco_detection.py
├── tools/
│   ├── train.py
│   └── test.py
└── runs/
```

## Acknowledgement

This work is supported by the open-source object detection ecosystem provided by the OpenMMLab community. We thank the maintainers and contributors of MMDetection, MMEngine, and MMCV.

## Citation

If you find this work useful for your research, please cite the paper.

```bibtex
@article{song2026windblade,
  title   = {WindBlade-30K: Towards Open-Source Benchmark for Drone-Based Wind Turbine Defect Detection},
  author  = {Song, Ye and Wu, Yiquan},
  year    = {2026}
}
```

If you use this toolbox, please also cite MMDetection.

```bibtex
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and others},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
