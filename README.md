# Colorectal Cancer Segmentation with Adaptive Augmentation and Multi-Resolution Ensemble Models

[![arXiv](https://img.shields.io/badge/arXiv-2507.04681-b31b1b.svg)](https://arxiv.org/abs/2507.04681)
[![Docker](https://img.shields.io/badge/Docker-Available-blue)](https://hub.docker.com/r/mertcaglar/segm)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

A comprehensive deep learning framework for semantic segmentation of colorectal cancer histopathology images, developed for the **ICIP2025 Grand Challenge**. This repository provides tools for multi-resolution whole slide image processing, adaptive data augmentation, and ensemble model training using state-of-the-art transformer architectures.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Advanced Features](#advanced-features)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This framework addresses the challenge of **Colorectal Cancer Tumor Grade Segmentation in Digital Histopathology Images** by providing:

- **Multi-resolution WSI processing** (60x, 40x, 20x magnifications)
- **Adaptive data augmentation** powered by GPT optimization
- **DPT (Dense Prediction Transformer)** models with MaxViT backbones
- **Ensemble learning** with Top-N soft biased voting
- **Docker containerization** for reproducible experiments

The system processes whole slide images (`.svs`, `.tif`, `.tiff`) with corresponding GeoJSON annotations to generate pixel-wise segmentation masks for five distinct tissue classes.

## ✨ Features

- **🔬 Multi-Scale Processing**: Handle WSIs at 60x, 40x, and 20x resolutions with efficient patch-based processing
- **🤖 Adaptive Augmentation**: AI-powered augmentation strategy optimization using OpenAI GPT-4
- **🏗️ Transformer Backbones**: MaxViT and DPT architectures for superior segmentation performance
- **📊 Comprehensive Evaluation**: IoU, accuracy, precision, recall, and F1-score tracking with Weights & Biases integration
- **🧩 Ensemble Methods**: Top-N soft voting with morphological post-processing
- **🐳 Docker Support**: Pre-built container for easy deployment and reproducibility
- **📈 Probability Maps**: Generate and visualize class probability distributions

## 📁 Dataset Structure

The framework expects the following directory structure:

```
ICIP2025/
├── geojson/                 # Original GeoJSON annotations
├── images/                  # Original whole slide images
├── dataset/                 # Primary 60x dataset
│   ├── images/
│   │   ├── train/
│   │   ├── test/
│   │   └── validation/
│   └── annotations/
│       ├── train/
│       └── validation/
├── 20/                     # 20x downscaled dataset
├── 40/                     # 40x downscaled dataset
└── [model_outputs]/        # Training results and predictions
```

### Class Labels

| Class | Label | Color | Description |
|-------|-------|-------|-------------|
| 0 | Others | Black | Non-cancerous areas, glass plates, stroma |
| 1 | T-G1 | Green | Tumor Grade-1 (well-differentiated) |
| 2 | T-G2 | Yellow | Tumor Grade-2 (moderately differentiated) |
| 3 | T-G3 | Red | Tumor Grade-3 (poorly differentiated) |
| 4 | Normal Mucosa | Blue | Healthy tissue regions |

## 🚀 Quick Start

### Docker Deployment (Recommended)

```bash
# Pull the pre-built image
docker pull mertcaglar/segm

# Run with GPU support
docker run --gpus all -it -v $(pwd):/host -w /host mertcaglar/segm
```

### Manual Installation

```bash
# Install dependencies
pip install torch torchvision albumentations segmentation-models-pytorch
pip install opencv-python pyvips wandb tqdm matplotlib seaborn

# Clone repository
git clone https://github.com/caglarmert/ICIP2025.git
cd ICIP2025
```

### Dataset Preparation

1. **Downscale WSIs**:
   ```bash
   python dsconvert.py
   ```
   This generates 20x and 40x versions of your dataset while maintaining the original structure.

2. **Configure paths** in `dsconvert.py` to match your local directory structure.

## 🔧 Methodology

### WSI Processing Pipeline

1. **Patch Extraction**: WSIs are divided into manageable patches (default: 2048×2048)
2. **Multi-Scale Downsampling**: Patches are resized using scale factors (0.1 for 20x, 0.025 for 40x)
3. **Mask Generation**: GeoJSON annotations are converted to RGB segmentation masks
4. **Tile Reconstruction**: Processed patches are reassembled into complete images

### Adaptive Training Strategy

The system employs GPT-4 powered augmentation optimization:

```python
# Example adaptive augmentation update
if (epoch) % update_interval == 0:
    new_aug_code = query_gpt_update(metrics_history, current_augmentations)
    train_transform = A.Compose(new_aug_code)
```

## 🏗️ Model Architectures

### Primary Configuration

| Resolution | Architecture | Patch Size | Stride | Encoder | Loss Function |
|------------|--------------|------------|--------|---------|---------------|
| 60x | DPT | 224 | 56 | `maxvit_large_tf_224` | Dice |
| 40x | DPT | 512 | 64 | `maxvit_large_tf_512` | Jaccard |
| 20x | DPT | 512 | 256 | `maxvit_large_tf_512` | Dice |
| 20x | DPT | 512 | 256 | `maxvit_xlarge_tf_512` | Tversky |

### Model Backbones

- **DPT (Dense Prediction Transformer)**: Vision transformer adapted for dense prediction tasks
- **MaxViT**: Multi-axis vision transformer with excellent scaling properties
- **Pre-trained Weights**: Models initialized with ImageNet-21k/1k pre-trained weights

## ⚡ Advanced Features

### Ensemble Voting

```bash
# Generate probability maps
python infer_probs.py

# Apply Top-N soft voting
python softvote_topN.py
```

The ensemble method combines:
- **Top-N Model Selection**: Choose best models per pixel based on confidence
- **Class-biased Weighting**: Prioritize tumor classes with custom weights
- **Morphological Filtering**: Clean small artifacts and fill holes
- **Gaussian Smoothing**: Improve spatial consistency

### Adaptive Augmentation

Configure in `augmentation_strategy.py`:

```python
# Set  OpenAI API key
client = OpenAI(api_key="_API_KEY")

# Customize augmentation prompt
prompt = f"""
You are an expert in data augmentation for deep learning...
Dataset: High-resolution histopathological dataset...
"""
```

## 📊 Usage

### Training

```bash
# Start adaptive training
python adaptive_trainer.py

# Monitor with Weights & Biases
wandb login
```

Key training parameters in `adaptive_trainer.py`:
- `Adaptive_Augmentation`: Enable GPT-powered augmentation
- `update_interval`: Frequency of augmentation updates (epochs)
- `loss_criterion`: Dice, Jaccard, Tversky, Lovasz, or CrossEntropy
- `patience`: Early stopping patience

### Inference

```bash
# Generate segmentation masks
python inference.py

# Create probability maps for ensemble
python infer_probs.py
```

### Evaluation

The framework provides comprehensive metrics:
- **Pixel Accuracy**: Overall classification accuracy
- **Mean IoU**: Intersection over Union averaged across classes
- **Per-class Metrics**: Precision, recall, F1-score for each tissue type
- **Visualization**: Overlay maps and probability distributions

## 📈 Results

The multi-resolution ensemble approach demonstrates:

- **Improved Boundary Detection**: Transformer architectures capture global context
- **Class Imbalance Handling**: Custom loss functions address rare classes
- **Multi-scale Consistency**: Ensemble across resolutions improves robustness
- **Adaptive Learning**: GPT-optimized augmentations boost performance

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{bahcekapili2025colorectal,
  title={Colorectal Cancer Tumor Grade Segmentation in Digital Histopathology Images: From Giga to Mini Challenge},
  author={Bahcekapili, Alper and Arslan, Duygu and Ozdemir, Umut and Ozkirli, Berkay and Akbas, Emre and Acar, Ahmet and Akar, Gozde B and He, Bingdou and Xu, Shuoyu and Caglar, Umit Mert and others},
  journal={arXiv preprint arXiv:2507.04681},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is intended for research purposes. Please check the ICIP2025 Grand Challenge terms and conditions for usage restrictions.

---

*For questions and support, please open an issue or contact the maintainer.*
