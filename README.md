# ICIP2025: Histopathology Colecteral Cancer Semantic Segmentation Models and Whole Slide Image Downscaling

This repository provides tools for downscaling Whole Slide Images (WSIs) and training segmentation models on the **ICIP2025 Grand Challenge dataset**, which includes WSI `.svs` images and corresponding `GeoJSON` annotation files.

We generate downscaled datasets at **20x** and **40x** magnifications that match the structure and format of the provided **60x** dataset. All processing is done patch-by-patch using a configurable downscaling pipeline.

---

## 🔄 WSI Downscaling

To downscale WSIs:

1. Use the script `dsconvert.py`.
2. Ensure that folder paths in the script are correctly set to match your local dataset structure.
3. WSIs are split into patches (default: `2048x2048`) and each patch is downscaled individually using the specified scale factor.
4. Corresponding segmentation masks are generated from `GeoJSON` annotations provided in the challenge dataset.
5. Downscaled dataset patches are combined after scaling to obtain single `.PNG` lossless compressed image and mask pairs.
---

## 🐳 Docker Support

A ready-to-use Docker image is available on Docker Hub:

```bash
docker pull mertcaglar/segm
```

Run the container with GPU support and mount the current directory:

```bash
docker run --gpus all -it -v $(pwd):/host -w /host mertcaglar/segm
```

This will launch the container and execute the main training script: `adaptive_trainer.py`.

---

## 🧪 Experiment Configurations

### Adaptive Augmentation

All of the experiments use adaptive data augmentation policy optimization method. The adaptive augmentation requires an OpenAI supscription and API key. Use the `augmentation_strategy.py` script to provide the API key and tune the prompt. The response will be executed inside the `adaptive_trainer.py` as a list of data augmentations described in the Albumentations library.


### Model training configurations

| Downscale | Architecture | Patch Size | Stride | Encoder                | Loss Function |
| --------- | ------------ | ---------- | ------ | ---------------------- | ------------- |
| 60x       | DPT          | 224        | 56     | `maxvit_large_tf_224`  | Dice          |
| 40x       | DPT          | 512        | 64     | `maxvit_large_tf_512`  | Jaccard       |
| 20x       | DPT          | 512        | 256    | `maxvit_large_tf_512`  | Dice          |
| 20x       | DPT          | 512        | 256    | `maxvit_xlarge_tf_512` | Tversky       |
| 20x       | DPT          | 512        | 256    | `maxvit_large_tf_512`  | Lovasz        |

### Probability Matrix Inference

Best results were obtained with training these models with given configurations. After obtaining these models, we've obtained the probability matrices for all of the test images for each of the models. To obtain the probability matrices use the provided script `infer_probs.py`. Change the configuration inside the script according to your models.

### Top-N Soft Biased Voting

With probability matrices, we've applied top-n soft biased voting with the script `softvote_topN.py`. Use this script to generate final prediction masks, by ensembling the best performing models, such as the models obtained with the provided configurations.

---

## 🧠 DPT Architecture

**DPT (Dense Prediction Transformer)** is a vision transformer-based architecture designed for dense prediction tasks such as semantic segmentation.

* Instead of using convolutional backbones, DPT leverages a **transformer-based encoder** with a **global receptive field** at every stage.
* Tokens from different transformer layers are reassembled into image-like representations and progressively decoded into high-resolution predictions via a **convolutional decoder**.
* This allows for **fine-grained segmentation** and improved global consistency over traditional convolutional networks.
