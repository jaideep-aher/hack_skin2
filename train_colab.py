

import os
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from datasets import load_dataset
from tqdm import tqdm
import json

HF_DATASET   = "marmal88/skin_cancer"
HF_MODEL_OUT = "YOUR_HF_USERNAME/skin-lesion-efficientnet"
EPOCHS       = 15
BATCH_SIZE   = 32
LR           = 1e-4
IMAGE_SIZE   = 224
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_ABBREV = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_MAP    = {k: i for i, k in enumerate(CLASS_ABBREV)}
CLASS_NAMES  = [
    "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
    "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesions",
]

print(f"Device: {DEVICE}")

from torchvision.transforms import (
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    RandomResizedCrop, ColorJitter, ToTensor, Normalize, Resize, Compose
)

train_transform = Compose([
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.5),
    RandomRotation(180),
    RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = Compose([
    Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SkinLesionDataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"].convert("RGB")
        label = LABEL_MAP.get(item["dx"], 5)
        if self.transform:
            img = self.transform(img)
        return img, label

print("Loading HAM10000 from HuggingFace...")
raw = load_dataset(HF_DATASET, trust_remote_code=True)
splits = raw["train"].train_test_split(test_size=0.2, seed=42)

train_ds = SkinLesionDataset(splits["train"], transform=train_transform)
val_ds   = SkinLesionDataset(splits["test"],  transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

model = timm.create_model("efficientnet_b0", pretrained=True)

params = list(model.parameters())
for p in params[:int(len(params) * 0.7)]:
    p.requires_grad = False

in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256),
    nn.Dropout(0.15),
    nn.Linear(256, 7),
)
model = model.to(DEVICE)

weights = torch.tensor([1.8, 2.5, 1.2, 4.0, 2.0, 0.5, 3.5]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": LR * 0.1},
    {"params": [p for n, p in model.named_parameters() if "classifier" in n],     "lr": LR},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
history = {"train_acc": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):

    model.train()
    correct = total = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    train_acc = correct / total

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} val"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    val_acc = correct / total
    scheduler.step()

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch:2d} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  New best: {val_acc:.3f}")

print(f"\nBest val accuracy: {best_val_acc:.3f}")

