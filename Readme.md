├── data
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── MIMO_IB_Compare
│   └── MIMOIB_model
│       ├── Complex_MIMOIB_infoNCE_SNR5_H9_20260303_155117.png
│       └── Complex_MIMOIB_infoNCE_SNR5_H9_20260303_155117.pth
├── MIMOIB_zjl
│   ├── data
│   ├── __init__.py
│   ├── MIMOIB_infoNCE.py
│   ├── test
│   │   ├── __init__.py
│   │   └── MIMOIB_test.py
│   └── utils
│       ├── channel.py
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── channel.cpython-310.pyc
│       │   ├── channel.cpython-39.pyc
│       │   ├── __init__.cpython-310.pyc
│       │   ├── __init__.cpython-39.pyc
│       │   ├── saving.cpython-310.pyc
│       │   └── saving.cpython-39.pyc
│       └── saving.py
└── training_params
    ├── training_history.json
    ├── training_params_20260303_152142.json
    └── training_params_20260303_203425.json


# README

## Satellite MIMO Information Bottleneck – Result Visualization

This repository contains code for generating **result visualizations** based on the paper:

> **"Robust Information Bottleneck for Satellite Edge Inference over MIMO Channel"**  
> *Jielin Zhu, Graduate Student Member, IEEE*

---

## 🎯 Purpose

This is **not** the official implementation of the paper.  
Instead, this repository is created to **generate sample result images** for demonstration and presentation purposes, based on the paper's proposed **Source-Channel Robust Information Bottleneck (SC-RIB)** framework.

---

## 🛰️ Background: The Paper's Contribution

The original paper proposes **SC-RIB**, a task-oriented communication framework for satellite-edge inference that:

- Addresses the **downlink bottleneck** in LEO satellite systems
- Integrates **Variational Information Bottleneck (VIB)** with **Deep Joint Source-Channel Coding (DeepJSCC)**
- Introduces **dual robustness** mechanisms:
  - **Source Robustness**: Fisher Information regularization against input perturbations (e.g., cloud cover)
  - **Channel Robustness**: Invariance to MIMO fading without requiring CSI
- Evaluated on **two scenarios**:
  1. **MNIST classification** – for channel robustness validation
  2. **Remote Sensing Change Detection (RaVEn dataset)** – for real-world satellite imagery analysis (fires, floods, etc.)

**Semantic Communication**: Yes, this work falls under the semantic communication paradigm – it prioritizes **task-relevant feature transmission** over bit-level reconstruction, enabling efficient satellite-ground collaborative inference.

---

## 🖼️ What This Repository Does

Our goal is to **reproduce / simulate result figures** from the paper, including:

| Figure | Description |
|--------|-------------|
| **Fig. 5** | Cloud removal example using CloudNet (preprocessing) |
| **Fig. 6** | Change detection heatmaps comparison (SC-RIB vs. RaVEn baseline) |
| **Fig. 7** | AUPRC performance curves across SNR levels |
| **Fig. 8-9** | Ablation studies (Fisher Information term, channel robustness components) |

These visualizations help demonstrate:
- Robustness to **cloud occlusion** (source perturbations)
- Resilience under **varying channel SNR** and **Rician fading**
- Superiority over baselines (**DeepJSCC**, **VFE**, **RaVEn**)

---

## 📦 Output Examples

After running the code, you will get:

- `cloud_removal.png` – Original image, cloud mask, cloud-removed result
- `change_detection_heatmap.png` – SC-RIB vs. baseline comparison
- `performance_curves.png` – Accuracy/AUPRC vs. SNR plots
- `ablation_study.png` – Impact of Fisher Information and channel constraints

---
