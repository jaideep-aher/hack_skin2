

import os
from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

CLASS_NAMES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions",
]

RISK_LEVEL = {
    "Actinic Keratoses":    "Moderate Risk",
    "Basal Cell Carcinoma": "High Risk",
    "Benign Keratosis":     "Low Risk",
    "Dermatofibroma":       "Low Risk",
    "Melanoma":             "High Risk",
    "Melanocytic Nevi":     "Watch",
    "Vascular Lesions":     "Watch",
}

RISK_EMOJI = {
    "High Risk":     "🔴",
    "Moderate Risk": "⚠️",
    "Watch":         "🟡",
    "Low Risk":      "🟢",
}

HF_MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"

def get_classifier(device: str = "cpu"):
    
    print(f"Loading {HF_MODEL_ID} from HuggingFace Hub...")
    classifier = pipeline(
        "image-classification",
        model=HF_MODEL_ID,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
        top_k=7,
    )
    print("Model ready.")
    return classifier

def normalize_label(raw_label: str) -> str:
    
    label_lower = raw_label.lower()
    mapping = {
        "actinic":      "Actinic Keratoses",
        "basal":        "Basal Cell Carcinoma",
        "benign":       "Benign Keratosis",
        "keratosis":    "Benign Keratosis",
        "dermatofibr":  "Dermatofibroma",
        "melanoma":     "Melanoma",
        "melanocytic":  "Melanocytic Nevi",
        "nevi":         "Melanocytic Nevi",
        "vascular":     "Vascular Lesions",
    }
    for key, name in mapping.items():
        if key in label_lower:
            return name

    return raw_label
