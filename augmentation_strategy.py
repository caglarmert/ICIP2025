from openai import OpenAI
client = OpenAI(api_key="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")

def query_gpt_initial():
    prompt = f"""
You are an expert in data augmentation for deep learning.

Dataset: High-resolution histopathological dataset providing detailed pixel-wise segmentation masks for five distinct classes: Grade-1 (well-differentiated), Grade-2 (moderately differentiated), Grade-3 (poorly differentiated) tumors, and Normal Mucosa. Regions not classified under these categories were labeled as "Others," including non-cancerous areas like glass plates and stroma.
Images: histopathological RGB, sliced into 224x224 patches during training.
Annotations: PNG masks with 5 classes (background, normal, tumor grade 1-3).
Model: deeplabv3_resnet101
Loss: CrossEntropyLoss"
Metric: Mean IoU
Available augmentations: "SquareSymmetry", "ColorJitter", "RandomToneCurve", "RandomBrightnessContrast", "RandomGamma", "ChannelDropout", "ToGray"

Please suggest an initial Albumentations pipeline as a Python list of transforms, respecting available augmentations.
Strict format:
[A.SquareSymmetry(p=0.5),
A.ColorJitter(p=0.5),
A.RandomToneCurve(p=0.5),
A.RandomBrightnessContrast(p=0.5),
A.RandomGamma(p=0.5),
A.ChannelDropout(p=0.5),
A.ToGray(p=0.25),
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
ToTensorV2()
]
Your response must follow these strict rules:
- Output only a valid Python list of transforms
- Do not wrap the result in markdown (no triple backticks)
- Do not explain or comment
- Do not change the format
- Do not change the last two transforms (normalize and tensor)

The response must begin with `[` and end with `]`.

"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def query_gpt_update(iou_history, accuracy_history, 
                                        precision_history, recall_history, 
                                        f1_history, current_augs,prev_augs):
    prompt = f"""
You are optimizing data augmentation for a segmentation model.

## Dataset Info:
- Histopathological patches of size 224x224 px
- 5 segmentation classes: Background (0), Tumor Grade 1 (1), Tumor Grade 2 (2), Tumor Grade 3 (3), Normal Mucosa (4)

Validation IoU history: {iou_history}
Validation accuracy history: {accuracy_history}
Validation precision history: {precision_history}
Validation recall history: {recall_history}
Validation F1 Score history: {f1_history}
Current augmentation pipeline (Albumentations): {current_augs}
Previous augmentation histories: {prev_augs}

Suggest an improved augmentation policy as a Python list of Albumentations transforms, respecting available augmentations.
Strict format:
[A.SquareSymmetry(p=0.5),
A.ColorJitter(p=0.5),
A.RandomToneCurve(p=0.5),
A.RandomBrightnessContrast(p=0.5),
A.RandomGamma(p=0.5),
A.ChannelDropout(p=0.5),
A.ToGray(p=0.25),
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
ToTensorV2()
]
Your response must follow these strict rules:
- Output only a valid Python list of transforms
- Do not wrap the result in markdown (no triple backticks)
- Do not explain or comment
- Do not change the format
- Do not change the last two transforms (normalize and tensor)

The response must begin with `[` and end with `]`.

"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()