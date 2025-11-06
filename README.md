---

# Colorectal Cancer Segmentation with Adaptive Augmentation and Multi-Resolution Ensemble Models

This repository contains tools for **downscaling Whole Slide Images (WSIs)** and **training semantic segmentation models** on the **ICIP2025 Grand Challenge** (Colorectal Cancer Tumor Grade Segmentation in Digital Histopathology Images), which includes `.svs` WSIs and their corresponding `GeoJSON` annotations.

[The Challange Paper (arXiv:2507.04681)](https://arxiv.org/abs/2507.04681)

We generate additional datasets at **40x** and **20x** magnifications to match the structure and format of the original **60x** dataset. The entire pipeline processes WSIs **patch-by-patch**, allowing flexible and scalable handling of large images.

---

## 🔄 WSI Downscaling

To downscale WSIs:

1. Run the `dsconvert.py` script.
2. Update folder paths in the script to match your local directory structure.
3. WSIs are split into patches (default: `2048x2048`), and each patch is individually downscaled based on a specified scale factor.
4. Segmentation masks are generated from `GeoJSON` annotations provided with the dataset.
5. Downscaled patches are reassembled into `.PNG` images and masks using lossless compression.

---

## 📁 Folder Structure

The expected folder structure is as follows:

```
ICIP2025/
├── geojson/
├── images/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── test/
│   │   └── validation/
│   └── annotations/
│       ├── train/
│       └── validation/
```

---

## 🐳 Docker Support

A prebuilt Docker image is available:

```bash
docker pull mertcaglar/segm
```

To run the container with GPU support and mount the current working directory:

```bash
docker run --gpus all -it -v $(pwd):/host -w /host mertcaglar/segm
```

This will launch the container and run the `adaptive_trainer.py` script.

---

## 🧪 Experimentation & Training

### 🧬 Adaptive Data Augmentation

All training experiments use **adaptive data augmentation**, optimized via the OpenAI API.
To use this feature:

* Provide your API key in `augmentation_strategy.py`.
* Adjust the prompt to define your augmentation policy.
* `adaptive_trainer.py` will then apply the generated augmentations using the [**Albumentations**](https://albumentations.ai/) library ([Buslaev et al., 2020](https://arxiv.org/abs/1809.06839)).

---

### 🛠️ Training Configurations

| Downscale | Architecture | Patch Size | Stride | Encoder                                                                         | Loss Function |
| --------- | ------------ | ---------- | ------ | ------------------------------------------------------------------------------- | ------------- |
| 60x       | DPT          | 224        | 56     | [`maxvit_large_tf_224`](https://huggingface.co/timm/maxvit_large_tf_224.in1k)   | Dice          |
| 40x       | DPT          | 512        | 64     | [`maxvit_large_tf_512`](https://huggingface.co/timm/maxvit_large_tf_512.in1k)   | Jaccard       |
| 20x       | DPT          | 512        | 256    | [`maxvit_large_tf_512`](https://huggingface.co/timm/maxvit_large_tf_512.in1k)   | Dice          |
| 20x       | DPT          | 512        | 256    | [`maxvit_xlarge_tf_512`](https://huggingface.co/timm/maxvit_xlarge_tf_512.in21k_ft_in1k) | Tversky       |
| 20x       | DPT          | 512        | 256    | [`maxvit_large_tf_512`](https://huggingface.co/timm/maxvit_large_tf_512.in1k)   | Lovasz        |

> **Backbone**: MaxViT – Tu et al., *"MaxViT: Multi-Axis Vision Transformer"* ([arXiv:2204.01697](https://arxiv.org/abs/2204.01697))
> Hugging Face models via [timm](https://huggingface.co/timm)

---

### 🔍 Probability Matrix Inference

Once models are trained using the configurations above, generate probability maps for test images using:

```bash
infer_probs.py
```

Make sure to modify the script configuration to match your model and dataset settings.

---

### 🧮 Top-N Soft Biased Voting

Use **Top-N Soft Biased Voting** to ensemble predictions across the best-performing models.
To generate final masks from the probability matrices:

```bash
softvote_topN.py
```

This method improves accuracy by aggregating predictions from multiple models using soft-weighted voting.

---

## 🧠 DPT Architecture Overview

**DPT (Dense Prediction Transformer)** is a vision transformer tailored for dense prediction tasks like semantic segmentation.

* Replaces conventional convolutional backbones with a **transformer-based encoder**, enabling **global context** at every layer.
* Intermediate transformer tokens are reconstructed into spatial feature maps and progressively **decoded via a convolutional decoder**.
* Offers **fine-grained segmentation** and superior **global consistency** compared to CNN-based approaches.

> **Dense Prediction Transformers**: Ranftl et al., *"Vision Transformers for Dense Prediction"*
> [DPT Paper (arXiv:2103.13413)](https://arxiv.org/abs/2103.13413)
> [Hugging Face Model Card](https://huggingface.co/Intel/dpt-large)

---



