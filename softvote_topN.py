import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
from utils import decode_segmap 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

model_folders = [
    "resized_probs8_6682best_40_DPT_AdamW_Jaccard_tu-maxvit_large_tf_512.in21k_ft_in1k",
    "probs_67best_DPT_AdamW_Dice_tu-maxvit_large_tf_224.in21k",
    "resized_probs8_6739best_20_DPT_AdamW_Dice_tu-maxvit_large_tf_512.in21k_ft_in1k",
    "resized_probs8_best2_20_DPT_AdamW_Tversky_tu-maxvit_large_tf_512.in21k_ft_in1k"
    # Add as many as you have
]
top_n = 3  # Set your Top-N value
n_model = len(model_folders)
output_prob_folder = f"gauss_blur_favor_bias_top{top_n}_{n_model}models_morph_probs"
output_mask_folder = f"gauss_blur_favor_bias_top{top_n}_{n_model}models_morph_masks"
output_filtered_mask_folder = f"gauss_blur_favor_bias_top{top_n}_{n_model}models_morph_filtered_masks"
os.makedirs(output_prob_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_filtered_mask_folder, exist_ok=True)

# === Get list of filenames from the first model folder ===
filenames = sorted([f for f in os.listdir(model_folders[0]) if f.endswith(".npy")])

for fname in tqdm(filenames, desc=f"Top-{top_n} Voting"):
    # === Load the probability maps from all models ===
    probs_all_models = []
    for folder in model_folders:
        path = os.path.join(folder, fname)
        probs = np.load(path)  # Shape: (C, H, W)
        probs_all_models.append(probs)

    probs_all_models = np.stack(probs_all_models, axis=0)  # Shape: (M, C, H, W)
    M, C, H, W = probs_all_models.shape

    # === Compute max prob per pixel per model ===
    max_probs = probs_all_models.max(axis=1)  # Shape: (M, H, W)

    # === Initialize Top-N voted probability accumulator ===
    topn_prob_sum = np.zeros((C, H, W), dtype=np.float32)
    topn_model_counts = np.zeros((H, W), dtype=np.int32)

    for y in range(H):
        for x in range(W):
            # Get top-N model indices for this pixel
            top_indices = np.argsort(max_probs[:, y, x])[::-1][:top_n]  # Descending sort
            for idx in top_indices:
                topn_prob_sum[:, y, x] += probs_all_models[idx, :, y, x]
            topn_model_counts[y, x] = top_n

    # === Normalize summed probabilities to get soft average ===
    averaged_probs = topn_prob_sum / np.maximum(topn_model_counts, 1e-6)

    # === Save soft averaged probs ===
    np.save(os.path.join(output_prob_folder, fname), averaged_probs)

    # Custom weights for each class
    class_weights = np.array([0.8, 1.5, 2.0, 1.8, 1.7], dtype=np.float32)
    # === Optional: Smooth the averaged probability maps ===
    smoothed_probs = np.zeros_like(averaged_probs)
    for c in range(C):
        smoothed_probs[c] = cv2.GaussianBlur(averaged_probs[c], ksize=(5, 5), sigmaX=2)
    biased_probs = smoothed_probs * class_weights[:, None, None]    
    # Use averaged_probs instead of smoothed_probs for non-smoothened version
    
    # === Convert to argmax mask with bias applied ===
    mask = np.argmax(biased_probs, axis=0).astype(np.uint8)
    rgb_mask = decode_segmap(mask)  # Convert to color if needed
    filename_png = fname.replace(".npy", ".png")
    cv2.imwrite(os.path.join(output_mask_folder, filename_png), rgb_mask)

    # === Post-process with morphological filtering ===
    cleaned_mask = np.zeros_like(mask)
    
    for cls in range(1, C):  # Skip background (0)
        binary_mask = (mask == cls).astype(np.uint8)
    
        # Morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
        # Remove small connected components (area < threshold)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        min_area = 100  # You can tune this value depending on your image size
    
        for i in range(1, num_labels):  # Skip background label 0
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = cls

    rgb_mask = decode_segmap(cleaned_mask)  # Convert to color if needed
    filename_png = fname.replace(".npy", ".png")
    cv2.imwrite(os.path.join(output_filtered_mask_folder, filename_png), rgb_mask)


print(f"\n Top-{top_n} soft voting complete.")
print("Saved averaged probabilities to:", output_prob_folder)
print("Saved segmentation masks to:", output_mask_folder)


# === CONFIGURATION ===
probs_folder = output_prob_folder
test_images_folder = "dataset/images/test"
image_mask_folder = output_mask_folder
output_overlay_folder = f"{probs_folder}_overlays"
os.makedirs(output_overlay_folder, exist_ok=True)

NUM_CLASSES = 5
CLASS_NAMES = {
    0: "Others",
    1: "Tumor Grade-1",
    2: "Tumor Grade-2",
    3: "Tumor Grade-3",
    4: "Normal Mucosa"
}

# Create transparent colormaps
def create_transparent_cmap(base_cmap='viridis', alpha=0.6):
    base = plt.cm.get_cmap(base_cmap)
    new_colors = base(np.linspace(0, 1, 256))
    new_colors[:, -1] = alpha  # Set transparency
    return ListedColormap(new_colors)

transparent_cmaps = {
    cls_idx: create_transparent_cmap(alpha=0.6) 
    for cls_idx in range(NUM_CLASSES)
}

def create_complete_visualization(original_img_path, prob_matrix, mask_path, output_path):
    original_img = plt.imread(original_img_path)
    mask_img = plt.imread(mask_path) if os.path.exists(mask_path) else None
    
    plt.figure(figsize=(18, 10))
    
    # 2-6. Class Overlays
    for cls_idx in range(NUM_CLASSES):
        plt.subplot(2, 3, cls_idx+1)
        plt.imshow(original_img)
        plt.imshow(prob_matrix[cls_idx], cmap=transparent_cmaps[cls_idx])
        plt.title(f"{CLASS_NAMES[cls_idx]} Overlay")
        plt.axis('off')
    
    # 7. Final Mask (shown in the first row if space allows)
    if mask_img is not None:
        plt.subplot(2, 3, 6)  # Adjust position if needed
        plt.imshow(original_img)
        plt.imshow(mask_img, alpha=0.6)
        plt.title("Final Prediction Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# === Process all files ===
npy_files = sorted([f for f in os.listdir(probs_folder) if f.endswith('.npy')])
png_files = sorted([f for f in os.listdir(test_images_folder) if f.endswith('.png')])

for fname, img_fname in tqdm(zip(npy_files, png_files), desc="Creating visualizations"):
    probs = np.load(os.path.join(probs_folder, fname))
    C, H, W = probs.shape
    assert C == NUM_CLASSES

    base_name = fname.replace('.npy', '')
    original_img_path = os.path.join(test_images_folder, img_fname)
    mask_path = os.path.join(image_mask_folder, img_fname)
    output_path = os.path.join(output_overlay_folder, f"{base_name}_complete.png")
    
    create_complete_visualization(original_img_path, probs, mask_path, output_path)

print(f"\nComplete visualizations saved in folder: {output_overlay_folder}")
