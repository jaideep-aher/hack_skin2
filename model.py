import os
import base64
import io
import requests
from PIL import Image

HF_MODEL = "Anwarkh1/Skin_Cancer-Image_Classification"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanocytic nevi",
    "Melanoma",
    "Vascular lesions",
]

RISK_LEVEL = {
    "Actinic keratoses": "High",
    "Basal cell carcinoma": "High",
    "Benign keratosis-like lesions": "Low",
    "Dermatofibroma": "Low",
    "Melanocytic nevi": "Low",
    "Melanoma": "Critical",
    "Vascular lesions": "Medium",
}

RISK_EMOJI = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
    "Unknown": "⚪",
}

_LABEL_MAP = {
    "akiec": "Actinic keratoses",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis-like lesions",
    "df":    "Dermatofibroma",
    "nv":    "Melanocytic nevi",
    "mel":   "Melanoma",
    "vasc":  "Vascular lesions",
}

def normalize_label(raw: str) -> str:
    key = raw.strip().lower()
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    for full in CLASS_NAMES:
        if key in full.lower() or full.lower() in key:
            return full
    return raw

def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def get_classifier(device: str = "cpu"):
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def classifier(img: Image.Image):
        img_bytes = _pil_to_bytes(img.convert("RGB"))
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=img_bytes,
            timeout=60,
        )
        if response.status_code == 503:
            import time
            time.sleep(20)
            response = requests.post(HF_API_URL, headers=headers, data=img_bytes, timeout=120)
        response.raise_for_status()
        raw = response.json()
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "label" in raw[0]:
            return [{"label": normalize_label(item["label"]), "score": item["score"]} for item in raw]
        return [{"label": cls, "score": 1.0 / len(CLASS_NAMES)} for cls in CLASS_NAMES]

    return classifier
