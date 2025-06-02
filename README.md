# ICIP2025

Whole Slide Image Downscaling

Use the ICIP2025 Grand Challenge dataset with Images and Geojson to obtain the downscaled images with 20x and 40x downscaling. These datasets will be identical to the provided 60x downscaled set.

use dsconvert.py, check the folder paths inside the script.

To process the WSI images in .svs format, I've used patches (default 2048) and processed whole images patch by patch, downscaling each patch with a scale factor. Each corresponding mask is obtained with the challenge dataset's geojson files.


# Docker

You can pull a readily available Docker image from the Dockerhub using:

docker pull mertcaglar/segm

and start a container from this image using:

docker run --gpus all -it -v $(pwd):/host -w /host mertcaglar/segm

The image will automatically make the current working directory available inside the container and run the adaptive_trainer.py, the main train code of this project.

Downscale, Architecture, Patch Size, Stride, Encoder, Loss Function
60x, DPT, 224, 56, maxvit_large_tf_224, Dice
40x, DPT, 512, 64, maxvit_large_tf_512, Jaccard
20x, DPT, 512, 256, maxvit_large_tf_512, Dice
20x, DPT, 512, 256, maxvit_xlarge_tf_512, Tversky
20x, DPT, 512, 256, maxvit_large_tf_512, Lovasz

DPT is a dense prediction architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks

It assembles tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combines them into full-resolution predictions using a convolutional decoder.

The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks


