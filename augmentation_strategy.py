from openai import OpenAI
client = OpenAI(api_key="KEY")

def query_gpt_initial():
    prompt = f"""
You are an expert in data augmentation for deep learning. Remember the following details about your task.
Dataset: High-resolution histopathological dataset providing detailed pixel-wise segmentation masks for five distinct classes: Grade-1 (well-differentiated), Grade-2 (moderately differentiated), Grade-3 (poorly differentiated) tumors, and Normal Mucosa. Regions not classified under these categories were labeled as "Others," including non-cancerous areas like glass plates and stroma.
Images: histopathological RGB, sliced into 224x224 patches during training.
Annotations: PNG masks with 5 classes (background, normal, tumor grade 1-3).
Model: Dense Prediction Transformers (DPT)
Encoder: tu-maxvit_large_tf_224.in21k 
Loss: CrossEntropyLoss and Dice Loss
Metrics: IoU, Accuracy, Precision, Recall, F1 Score
Your task is to improve the overall metrics in each step. You should be careful about the dataset and the model when arranging augmentations.
Some available augmentations: "A.SquareSymmetry", "A.ColorJitter", "A.RandomToneCurve", "A.RandomBrightnessContrast", "A.RandomGamma", "A.ChannelDropout", "A.ToGray"

Answer format:
[A.SquareSymmetry(),
A.ColorJitter(),
A.RandomToneCurve(),
A.RandomBrightnessContrast(),
A.RandomGamma(),
A.ChannelDropout(),
A.ToGray(),
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
ToTensorV2()
]

Suggest an initial augmentation policy as a Python list of Albumentations transforms. You can omit augmentations (except A.Normalize and A.ToTensorV2).

Your response must follow these strict rules:
- Output only a valid Python list of Albumentations transforms
- Do not wrap the result in markdown (no triple backticks)
- Do not explain or comment
- Do not change the format
- Do not change the last two transforms (A.Normalize and A.ToTensorV2)

The response must begin with `[` and end with `]`.
"""
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def query_gpt_update(iou_history, accuracy_history, 
                                        precision_history, recall_history, 
                                        f1_history, current_augs,prev_augs, error_history, model, loss, encoder):
    prompt = f"""
You are an expert in data augmentation for deep learning. Remember the following details about your task.
Dataset: High-resolution histopathological dataset providing detailed pixel-wise segmentation masks for five distinct classes: Grade-1 (well-differentiated), Grade-2 (moderately differentiated), Grade-3 (poorly differentiated) tumors, and Normal Mucosa. Regions not classified under these categories were labeled as "Others," including non-cancerous areas like glass plates and stroma.
Images: histopathological RGB, sliced into 224x224 patches during training.
Annotations: PNG masks with 5 classes (background, normal, tumor grade 1-3).
Model Architecture: {model}
Loss: {loss}
Encoder: {encoder}
Metrics: IoU, Accuracy, Precision, Recall, F1 Score
Your task is to improve the overall metrics in each step. You should be careful about the dataset and the model architecture when arranging augmentations.

Validation IoU history: {iou_history}
Validation accuracy history: {accuracy_history}
Validation precision history: {precision_history}
Validation recall history: {recall_history}
Validation F1 Score history: {f1_history}
Current augmentation pipeline (Albumentations): {current_augs}
Previous augmentation histories: {prev_augs}
Update error history: {error_history}

Suggest an improved augmentation policy as a Python list of Albumentations transforms. You can omit augmentations (except A.Normalize and A.ToTensorV2).

Your response must follow these strict rules:
- Output only a valid Python list of Albumentations transforms
- Do not wrap the result in markdown (no triple backticks)
- Do not explain or comment
- Do not change the format
- Do not change the last two transforms (A.Normalize and A.ToTensorV2)

The response must begin with `[` and end with `]`.
"""
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()