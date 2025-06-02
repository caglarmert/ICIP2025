# Use NVIDIA CUDA 12.2 base image with Python pre-installed
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3
    
RUN apt-get install -y python3-pip

RUN pip install \
    opencv-python-headless==4.11.0.86 \
    numpy==2.2.6 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    albumentations==2.0.7 \
    segmentation-models-pytorch==0.5.0 \
    wandb==0.19.11 \
    tqdm==4.67.1 \
    scikit-learn==1.6.1 \
    openai==1.79.0

CMD ["python3", "adaptive_trainer.py"]

