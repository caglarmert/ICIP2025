import os
import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, JaccardLoss, TverskyLoss, LovaszLoss, SoftCrossEntropyLoss
from albumentations.pytorch import ToTensorV2
from augmentation_strategy import query_gpt_update
import wandb
from tqdm import tqdm
from utils import evaluate, mask_to_class, SegmentationDataset, decode_segmap, zip_folder
from utils import COLOR2LABEL, LABEL2COLOR
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = len(COLOR2LABEL)

wandb.login(key = "KEY")

def train_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(loader)
    return avg_train_loss

Adaptive_Augmentation = True
train_transform = A.Compose([
    A.SquareSymmetry(p=1),
    A.PlanckianJitter(p=0.5),
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

epochs = 100
batch_size = 64
patch_size = 224
stride = 224
patience = 10
best_iou = 0.0
counter = 0
save_best_model = False
loss_criterion = "Dice"
DOWNSCALE = "60"
ARCH = "DPT"
OPTIMIZER = "AdamW"
ES = "ES"
AUGMENTATION = "Aug"
ENCODER="tu-maxvit_large_tf_224.in21k"
ENCODER_WEIGHTS="in21k"

if DOWNSCALE == "60":
    data_path = "dataset"
else:
    data_path = DOWNSCALE


train_ds = SegmentationDataset(f"{data_path}/images/train", f"{data_path}/annotations/train", train_transform, patch_size, stride)
val_ds = SegmentationDataset(f"{data_path}/images/validation", f"{data_path}/annotations/validation", val_transform, patch_size, stride)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

# Model
model = smp.DPT(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=NUM_CLASSES,
)

model = model.cuda()

# Loss and optimizer
if loss_criterion == "CE":
    criterion = torch.nn.CrossEntropyLoss()
elif loss_criterion == "Dice":
    criterion = DiceLoss(mode = "multiclass")
elif loss_criterion == "Focal":
    criterion = FocalLoss(mode = "multiclass")
elif loss_criterion == "Jaccard":
    criterion = JaccardLoss(mode = "multiclass")
elif loss_criterion == "Tversky":
    criterion = TverskyLoss(mode = "multiclass",  alpha=0.4, beta=0.6,)
elif loss_criterion == "Lovasz":
    criterion = LovaszLoss(mode = "multiclass")
elif loss_criterion == "SCE":
    criterion = SoftCrossEntropyLoss()
else:
    pass
    
if OPTIMIZER == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
elif OPTIMIZER == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
elif OPTIMIZER == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

wandb.init(project="ICIP2025", entity="caglarmert", config={
    "learning_rate": optimizer.param_groups[0]['lr'],
    "dataset": DOWNSCALE,
    "epochs": epochs,
    "patch_size": patch_size,
    "stride": stride,
    "batch_size": train_loader.batch_size,
    "loss": loss_criterion,
    "ES_patience": patience,
    "Optimizer": OPTIMIZER,
    "Early Stop": ES,
    "aug": AUGMENTATION,
    "encoder": ENCODER,
    "encoder_weights": ENCODER_WEIGHTS,
    "architecture": ARCH,
    "Adaptive_Augmentation": Adaptive_Augmentation
})

model_name = f"{DOWNSCALE}_{ARCH}_{OPTIMIZER}_{loss_criterion}_{ENCODER}"
train_ds.transform = train_transform

iou_history = []
accuracy_history = []
precision_history = []
recall_history = []
f1_history = []
prev_augs_str = []
error_history = []
update_interval = 3


for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    metrics = evaluate(model, val_loader)
    val_iou = metrics["iou"]
    scheduler.step(val_iou)
    if (epoch) % update_interval == 0:
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
        "lr": optimizer.param_groups[0]['lr'],
        "Othr_IoU": metrics["iou_per_class"][0],
        "T-G1_IoU": metrics["iou_per_class"][1],
        "T-G2_IoU": metrics["iou_per_class"][2],
        "T-G3_IoU": metrics["iou_per_class"][3],
        "N_Mu_IoU": metrics["iou_per_class"][4]
    })
    
    if val_iou > best_iou:
        best_iou = val_iou
        counter = 0
        print("IoU improved, saving model...")
        torch.save(model.state_dict(), "best_" + model_name + ".pth")
    else:
        if ES == "ES":
            counter += 1
            print(f"No improvement in IoU. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                if save_best_model:
                    artifact = wandb.Artifact('model', type='model')
                    artifact.add_file("best_" + model_name + ".pth")
                    wandb.log_artifact(artifact)
                break
    if Adaptive_Augmentation:
        if (epoch) % update_interval == 0:
            current_aug_str = str(train_transform)
            new_aug_code = query_gpt_update(iou_history, accuracy_history, precision_history, recall_history,
                                            f1_history, current_aug_str, prev_augs_str, error_history, ARCH, loss_criterion, ENCODER)                                            
            prev_augs_str.append(current_aug_str)
            print("GPT Updated Augmentations:\n", new_aug_code)
            try:
                exec(f"train_transform = A.Compose({new_aug_code})")
                train_ds.transform = train_transform
            except Exception as e:
                print("Failed to apply new augmentations. Keeping previous.")
                print(e)
                error_history.append(e)

# Create output folder

os.makedirs(model_name, exist_ok=True)

# Load model for inference
model.load_state_dict(torch.load(f"best_{model_name}.pth"))
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
    mask = np.argmax(averaged_probs, axis=0).astype(np.uint8)
    rgb_mask = decode_segmap(mask)
    filename = inference_ds.images[img_idx]
    cv2.imwrite(os.path.join(model_name, filename), rgb_mask)

if DOWNSCALE != "60":
    # Set paths
    reference_folder = "dataset/images/test"
    source_folder = model_name
    target_folder = f"resized_{model_name}"
    
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
    
else:    
    folder_to_zip = model_name
    
zip_filename = f"{model_name}.zip"

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