# ICIP2025

Whole Slide Image Downscaling

Downscale, Architecture, Patch Size, Stride, Encoder, Loss Function
60x, DPT, 224, 56, maxvit_large_tf_224, Dice
40x, DPT, 512, 64, maxvit_large_tf_512, Jaccard
20x, DPT, 512, 256, maxvit_large_tf_512, Dice
20x, DPT, 512, 256, maxvit_xlarge_tf_512, Tversky
20x, DPT, 512, 256, maxvit_large_tf_512, Lovasz

DPT is a dense prediction architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks

It assembles tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combines them into full-resolution predictions using a convolutional decoder.

The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks


