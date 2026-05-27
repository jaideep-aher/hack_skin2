

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import SkinLesionClassifier, CLASS_NAMES
from augmentations import SkinAugmentationPipeline

class HAM10000Dataset(Dataset):
    

    LABEL_MAP = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}

    def __init__(self, data_dir: str, transform=None, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        csv_path = self.data_dir / "HAM10000_metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {csv_path}\n"
                "Download HAM10000 from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
            )

        df = pd.read_csv(csv_path)
        self.image_ids = df["image_id"].tolist()
        self.labels = [self.LABEL_MAP[dx] for dx in df["dx"].tolist()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = self.labels[idx]

        img_path = self.data_dir / "images" / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = self.data_dir / "images" / f"{img_id}.JPG"

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

def load_hf_dataset(split: str = "train", augment: bool = True):
    
    try:
        from datasets import load_dataset
        print("Loading HAM10000 from Hugging Face Hub (marmal88/skin_cancer)...")
        ds = load_dataset("marmal88/skin_cancer", split=split, trust_remote_code=True)

        pipeline = SkinAugmentationPipeline()
        label_map = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}

        class HFWrapper(Dataset):
            def __init__(self, hf_dataset, augment=True):
                self.dataset = hf_dataset
                self.augment = augment

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]
                img = item["image"].convert("RGB")
                label = label_map.get(item["dx"], 5)
                if self.augment:
                    tensor = pipeline.training_transform(img)
                else:
                    tensor = pipeline.inference_transform(img)
                return tensor, label

        return HFWrapper(ds, augment=augment)

    except Exception as e:
        print(f"HuggingFace load failed: {e}")
        return None

def train_epoch(model, loader, optimizer, criterion, device, desc="Train"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels

def compute_per_class_acc(preds, labels, class_names):
    
    results = {}
    for i, name in enumerate(class_names):
        mask = [l == i for l in labels]
        if sum(mask) == 0:
            results[name] = None
            continue
        cls_preds = [p for p, m in zip(preds, mask) if m]
        cls_labels = [l for l, m in zip(labels, mask) if m]
        results[name] = sum(p == l for p, l in zip(cls_preds, cls_labels)) / len(cls_labels)
    return results

def main():
    parser = argparse.ArgumentParser(description="Train Skin Lesion Classifier")
    parser.add_argument("--data-dir", default="./data", help="Path to HAM10000 directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Use skin-specific augmentations")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Baseline: no augmentation")
    parser.add_argument("--output-dir", default="./weights")
    parser.add_argument("--use-hf", action="store_true", default=True,
                        help="Load dataset from Hugging Face Hub instead of local")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Skin Lesion Robustness - Training")
    print(f"  Device: {device} | Augment: {args.augment} | Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    pipeline = SkinAugmentationPipeline()
    dataset = None

    if args.use_hf:
        dataset = load_hf_dataset(split="train", augment=args.augment)

    if dataset is None:

        transform = pipeline.training_transform if args.augment else pipeline.inference_transform
        dataset = HAM10000Dataset(args.data_dir, transform=transform)

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    model = SkinLesionClassifier(num_classes=7, pretrained=True).to(device)

    class_weights = torch.tensor([1.8, 2.5, 1.2, 4.0, 2.0, 0.5, 3.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n],
         "lr": args.lr * 0.1},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n],
         "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0
    aug_tag = "augmented" if args.augment else "baseline"

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device,
                                            desc=f"Epoch {epoch}/{args.epochs} [train]")
        val_loss, val_acc, all_preds, all_labels = eval_epoch(model, val_loader, criterion, device,
                                                               desc=f"Epoch {epoch}/{args.epochs} [val]")
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train: {train_acc:.3f} ({train_loss:.4f}) | "
              f"Val: {val_acc:.3f} ({val_loss:.4f}) | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(os.path.join(args.output_dir, f"best_{aug_tag}.pt"))
            print(f"  ✓ New best model saved (val_acc={val_acc:.3f})")

    print(f"\nFinal validation accuracy ({aug_tag}): {best_val_acc:.3f}")

    per_class = compute_per_class_acc(all_preds, all_labels, CLASS_NAMES)
    print("\nPer-class accuracy:")
    for cls, acc in per_class.items():
        print(f"  {cls:<30} {acc:.3f}" if acc is not None else f"  {cls:<30} N/A")

    with open(os.path.join(args.output_dir, f"history_{aug_tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {args.output_dir}/history_{aug_tag}.json")

if __name__ == "__main__":
    main()
