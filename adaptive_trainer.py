import os
import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from albumentations.pytorch import ToTensorV2
from augmentation_strategy import query_gpt_initial, query_gpt_update
import wandb
from tqdm import tqdm
from utils import evaluate, mask_to_class, SegmentationDataset, decode_segmap, zip_folder
from utils import COLOR2LABEL, LABEL2COLOR
import torch.nn as nn
import torch.nn.functional as F

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

# Usage
criterion = DiceCELoss(ce_weight=1.0, dice_weight=1.0)

NUM_CLASSES = len(COLOR2LABEL)

# Training loop
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(loader)
    return avg_train_loss

Adaptive_Augmentation = True
# Data transforms
train_transform = A.Compose([
    A.SquareSymmetry(p=1),
    A.RandomToneCurve(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ChannelDropout(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Parameters
epochs = 100
batch_size = 100
patch_size = 224
stride = 224
patience = 5
best_iou = 0.0
counter = 0
loss_criterion = "CE+Dice"
OPTIMIZER = "AdamW"
ES = "ES"
AUGMENTATION = "Aug"

train_ds = SegmentationDataset('dataset/images/train', 'dataset/annotations/train', train_transform, patch_size, stride)
val_ds = SegmentationDataset('dataset/images/validation', 'dataset/annotations/validation', val_transform, patch_size, stride)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
#model = deeplabv3_resnet101(weights=None, num_classes=NUM_CLASSES)
#model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[-1] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model = model.cuda()

# Loss and optimizer
if loss_criterion == "CE":
    criterion = torch.nn.CrossEntropyLoss()
elif loss_criterion == "Dice":
    criterion = DiceLoss()
elif loss_criterion == "CE+Dice":
    criterion = DiceCELoss() 
else:
    pass
if OPTIMIZER == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
elif OPTIMIZER == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

wandb.init(project="ICIP2025", entity="caglarmert", config={
    "learning_rate": optimizer.param_groups[0]['lr'],
    "architecture": "deeplabv3_resnet50",
    "dataset": "downscaled",
    "epochs": epochs,
    "patch_size": patch_size,
    "stride": stride,
    "batch_size": train_loader.batch_size,
    "loss": loss_criterion,
    "ES_patiance": patience,
    "Augmentation": "adaptive_augmentation",
    "Optimizer": OPTIMIZER,
    "Early Stop": ES,
    "aug": AUGMENTATION,
    "Adaptive_Augmentation": Adaptive_Augmentation
})

model_name = f"aa3_{OPTIMIZER}_{ES}_{AUGMENTATION}.pth"

if Adaptive_Augmentation:
    pass
    # initial_aug_code = query_gpt_initial()
    # print("Initial Augmentations:\n", initial_aug_code)
    # exec(f"train_transform = A.Compose({initial_aug_code})")

train_ds.transform = train_transform

iou_history = []
accuracy_history = []
precision_history = []
recall_history = []
f1_history = []
prev_augs_str = []
error_history = []
update_interval = 1


for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    metrics = evaluate(model, val_loader)
    val_iou = metrics["iou"]
    scheduler.step(val_iou)
    iou_history.append(metrics["iou"])
    accuracy_history.append(metrics["accuracy"])
    precision_history.append(metrics["precision"])
    recall_history.append(metrics["recall"])
    f1_history.append(metrics["f1"])
    print(f"Validation IoU = {val_iou:.4f}")

    wandb.log({
        "train_loss": train_loss,
        "val_iou": val_iou,
        "accuracy" : metrics["accuracy"],
        "precision" : metrics["precision"],
        "recall" : metrics["recall"],
        "f1" : metrics["f1"],
        "lr": optimizer.param_groups[0]['lr']
    })
    
    if val_iou > best_iou:
        best_iou = val_iou
        counter = 0
        print("IoU improved, saving model...")
        torch.save(model.state_dict(), "best_" + model_name)
    else:
        if ES == "ES":
            counter += 1
            print(f"No improvement in IoU. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file("best_" + model_name)
                wandb.log_artifact(artifact)
                break
    if Adaptive_Augmentation:
        if (epoch + 1) % update_interval == 0:
            current_aug_str = str(train_transform)
            new_aug_code = query_gpt_update(iou_history, accuracy_history, precision_history, recall_history,
                                            f1_history, current_aug_str, prev_augs_str, error_history)
            prev_augs_str.append(current_aug_str)
            print("GPT Updated Augmentations:\n", new_aug_code)
            try:
                exec(f"train_transform = A.Compose({new_aug_code})")
                train_ds.transform = train_transform
            except Exception as e:
                print("Failed to apply new augmentations. Keeping previous.")
                print(e)
                error_history.append(e)

torch.save(model.state_dict(), model_name)

# Create output folder
output_dir = f"aa3_{OPTIMIZER}_{ES}_{AUGMENTATION}"
os.makedirs(output_dir, exist_ok=True)

# Load model for inference
model.load_state_dict(torch.load("best_" + model_name))
model.eval()

# Inference dataset with patch slicing
inference_ds = SegmentationDataset('dataset/images/test', mask_dir=None, transform=val_transform, inference_mode=True)
inference_loader = DataLoader(inference_ds, batch_size=1, shuffle=False)

# Get dimensions of all test images
image_shapes = {i: cv2.imread(os.path.join('dataset/images/test', img)).shape[:2] for i, img in enumerate(inference_ds.images)}
final_preds = {i: np.zeros((h, w), dtype=np.int64) for i, (h, w) in image_shapes.items()}
vote_mask = {i: np.zeros((h, w), dtype=np.int64) for i, (h, w) in image_shapes.items()}

with torch.no_grad():
    for batch in tqdm(inference_loader):
        image_patch, img_idx, y, x = batch
        image_patch = image_patch.cuda()
        output = model(image_patch)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        img_idx, y, x = img_idx.item(), y.item(), x.item()
        final_preds[img_idx][y:y+224, x:x+224] += pred
        vote_mask[img_idx][y:y+224, x:x+224] += 1

for img_idx in final_preds:
    mask = (final_preds[img_idx] // np.maximum(vote_mask[img_idx], 1)).astype(np.uint8)
    rgb_mask = decode_segmap(mask)
    filename = inference_ds.images[img_idx]
    cv2.imwrite(os.path.join(output_dir, filename), rgb_mask)

# Example usage:
folder_to_zip = output_dir
zip_filename = 'best_results.zip'

# Zip the folder
zip_folder(folder_to_zip, zip_filename)

# Create and log W&B artifact
artifact = wandb.Artifact(
    name='zipped_folder',
    type='results',
    description='Results'
)
artifact.add_file(zip_filename)  
wandb.log_artifact(artifact)
