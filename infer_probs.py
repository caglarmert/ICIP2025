import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from utils import evaluate, mask_to_class, SegmentationDataset, decode_segmap, zip_folder
from utils import COLOR2LABEL, LABEL2COLOR
from utils import DiceLoss, DiceCELoss
from tqdm import tqdm
import zipfile
import torch.nn.functional as F

# Color to class mapping
NUM_CLASSES = len(COLOR2LABEL)

val_transform = A.Compose([
    #A.CLAHE(p=1.0), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    # some models use CLAHE by default for validation as well! be careful with this parameter here!
])

patch_size = 512
stride = 64
DOWNSCALE = "20" #Dataset downscale versions
ENCODER="tu-maxvit_large_tf_512.in21k_ft_in1k"
ENCODER_WEIGHTS="in21k_ft_in1k"

if DOWNSCALE == "60":
    data_path = "dataset"
else:
    data_path = DOWNSCALE

# Model
model = smp.DPT(
    encoder_name=ENCODER,
    decoder_readout="ignore",
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=NUM_CLASSES,
)
model = torch.nn.DataParallel(model)
model = model.cuda()

# Create output folder
model_name = "best2_20_DPT_AdamW_Tversky_tu-maxvit_large_tf_512.in21k_ft_in1k"
os.makedirs(f"probs8_{model_name}", exist_ok=True)

# Load model for inference
model.load_state_dict(torch.load(f"{model_name}.pth"))
model.eval()

# Inference dataset with patch slicing
inference_ds = SegmentationDataset(f"{data_path}/images/test", mask_dir=None, transform=val_transform, inference_mode=True, patch_size=patch_size, stride=stride)
inference_loader = DataLoader(inference_ds, batch_size=1, shuffle=False, num_workers=1)

# Get dimensions of all test images
image_shapes = {i: cv2.imread(os.path.join(f"{data_path}/images/test", img)).shape[:2] for i, img in enumerate(inference_ds.images)}
final_probs = {i: np.zeros((NUM_CLASSES, h, w), dtype=np.float32) for i, (h, w) in image_shapes.items()}
vote_mask = {i: np.zeros((h, w), dtype=np.int32) for i, (h, w) in image_shapes.items()}

with torch.no_grad():
    for batch in tqdm(inference_loader):
        image_patch, img_idx, y, x = batch
        image_patch = image_patch.cuda()
        output = model(image_patch)  # [B, C, H, W]
        probs = F.softmax(output.squeeze(0), dim=0).cpu().numpy().astype(np.float32)

        img_idx, y, x = img_idx.item(), y.item(), x.item()
        final_probs[img_idx][:, y:y+patch_size, x:x+patch_size] += probs
        vote_mask[img_idx][y:y+patch_size, x:x+patch_size] += 1

for img_idx in final_probs:
    averaged_probs = final_probs[img_idx] / np.maximum(vote_mask[img_idx], 1e-5)
    filename = os.path.splitext(inference_ds.images[img_idx])[0]  # remove .png
    out_path = os.path.join(f"probs8_{model_name}", f"{filename}.npy")
    np.save(out_path, averaged_probs)

if DOWNSCALE != "60":
    from PIL import Image
    # Set paths
    reference_folder = "dataset/images/test"
    source_folder = f"probs8_{model_name}"
    target_folder = f"resized_probs8_{model_name}"
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    image_size_map = {}
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith(".png"):
            ref_path = os.path.join(reference_folder, filename)
            with Image.open(ref_path) as img:
                image_size_map[filename] = img.size  # (width, height)
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".png") and filename in image_size_map:
            src_path = os.path.join(source_folder, filename)
            with Image.open(src_path) as img:
                target_size = image_size_map[filename]
                resized_img = img.resize(target_size, resample=Image.NEAREST)
                target_path = os.path.join(target_folder, filename)
                resized_img.save(target_path)
    
    print("All matching images resized and saved to:", target_folder)
    folder_to_zip = target_folder

    # Set paths
    reference_folder = "dataset/images/test"
    source_folder = f"probs8_{model_name}"
    target_folder = f"resized_probs8_{model_name}"
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Read original image sizes (width, height)
    image_size_map = {}
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith(".png"):
            ref_path = os.path.join(reference_folder, filename)
            with Image.open(ref_path) as img:
                image_size_map[filename.replace(".png", ".npy")] = img.size  # .npy extension!
    
    # Resize each .npy matrix
    for filename in os.listdir(source_folder):
        if filename.endswith(".npy") and filename in image_size_map:
            npy_path = os.path.join(source_folder, filename)
            target_size = image_size_map[filename]  # (width, height)
    
            # Load and convert to torch tensor
            prob_map = np.load(npy_path)  # Shape: [C, H, W]
            prob_tensor = torch.from_numpy(prob_map).unsqueeze(0)  # Shape: [1, C, H, W]
    
            # Resize using bilinear interpolation
            resized = F.interpolate(
                prob_tensor,
                size=(target_size[1], target_size[0]),  # (H, W)
                mode="bilinear",
                align_corners=False
            ).squeeze(0).numpy()  # Shape: [C, H, W]
    
            # Save resized probs
            np.save(os.path.join(target_folder, filename), resized)
else:    
    folder_to_zip = f"probs8_{model_name}"
    
zip_filename = f"probs8_{model_name}.zip"

# Zip the folder
zip_folder(folder_to_zip, zip_filename)
