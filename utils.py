from torch.utils.data import Dataset
import numpy as np
import torch
import zipfile
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F

# Color to class mapping
COLOR2LABEL = {
    (0, 0, 0): 0,        # Black - Others
    (0, 192, 0): 1,      # Green - Tumor Grade-1
    (32, 224, 255): 2,    # Yellow - Tumor Grade-2
    (0, 0, 255): 3,      # Red - Tumor Grade-3
    (255, 32, 0): 4       # Blue - Normal Mucosa
}

LABEL2COLOR = {
    0: [0, 0, 0],         # Others
    1: [0, 192, 0],       # Tumor Grade-1
    2: [32, 224, 255],    # Tumor Grade-2
    3: [0, 0, 255],       # Tumor Grade-3
    4: [255, 32, 0],      # Normal Mucosa
}

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Prevents division by zero

    def forward(self, inputs, targets):
        # Inputs: (batch_size, num_classes, height, width) [logits]
        # Targets: (batch_size, height, width) [class indices]

        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply softmax to logits
        probs = F.softmax(inputs, dim=1)

        # Compute intersection and union
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3))  # (B, C)
        union = torch.sum(probs + targets_onehot, dim=(2, 3))        # (B, C)

        # Dice coefficient per class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # Average across classes and batches
        loss = 1 - dice.mean()
        return loss
        
class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-6, ce_weight=1.0, dice_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # Cross-Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets)

        # Dice Loss
        num_classes = inputs.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(inputs, dim=1)
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3))
        union = torch.sum(probs + targets_onehot, dim=(2, 3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        # Combined Loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss

def evaluate(model, loader, num_classes=5):
    model.eval()
    all_preds = []
    all_labels = []
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            outputs = model(imgs) #['out']
            preds = torch.argmax(outputs, dim=1)

            # Flatten predictions and masks
            all_preds.append(preds.view(-1).cpu().numpy())
            all_labels.append(masks.view(-1).cpu().numpy())

            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Accuracy
    acc = total_correct / total_pixels

    # Per-class & macro metrics
    iou = jaccard_score(y_true, y_pred, average='macro', labels=list(range(num_classes)))
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Mean IoU: {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        "accuracy": acc,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, patch_size=224, stride=224, inference_mode=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.inference_mode = inference_mode

        self.images = sorted(os.listdir(image_dir))
        self.image_paths = [os.path.join(image_dir, img) for img in self.images]
        self.mask_paths = [os.path.join(mask_dir, img) for img in self.images] if mask_dir else None

        # Generate (image index, y, x) patch coordinates
        self.patches = []
        for idx, path in enumerate(self.image_paths):
            image = cv2.imread(path)
            h, w, _ = image.shape
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((idx, y, x))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        img_path = self.image_paths[img_idx]
        image = cv2.imread(img_path)

        image_patch = image[y:y+self.patch_size, x:x+self.patch_size]

        if self.mask_paths:
            mask = cv2.imread(self.mask_paths[img_idx])
            mask = mask_to_class(mask)
            mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        else:
            mask_patch = None

        if self.transform:
            if mask_patch is not None:
                augmented = self.transform(image=image_patch, mask=mask_patch)
                image_patch, mask_patch = augmented['image'], augmented['mask']
            else:
                image_patch = self.transform(image=image_patch)['image']

        if mask_patch is not None:
            return image_patch, mask_patch.long()
        else:
            return image_patch, img_idx, y, x  # For reconstruction in inference

def decode_segmap(label):
    """Convert a mask with class indices to RGB."""
    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cls, color in LABEL2COLOR.items():
        rgb[label == cls] = color
    return rgb

def zip_folder(folder_path, zip_path):
    """
    Zip all contents of a folder (without the parent folder itself)
    
    Args:
        folder_path (str): Path to folder to zip
        zip_path (str): Path to output zip file
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip using relative path
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

def mask_to_class(mask):
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, idx in COLOR2LABEL.items():
        label_mask[np.all(mask == rgb, axis=-1)] = idx
    return label_mask