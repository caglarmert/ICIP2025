# Use NVIDIA CUDA 12.2 base image with Python pre-installed
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3
    
RUN apt-get install -y python3-pip

RUN pip install \
    opencv-python-headless \
    numpy \
    torch torchvision torchaudio \
    albumentations \
    segmentation-models-pytorch \
    wandb \
    tqdm \
    scikit-learn \
    openai

CMD ["python3", "adaptive_trainer.py"]

