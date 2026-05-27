

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch

class SkinToneShift:
    

    TONE_PROFILES = {
        "type_i_ii": {"brightness": (1.1, 1.35), "saturation": (0.7, 0.9), "hue": (-0.02, 0.02)},
        "type_iii":  {"brightness": (0.95, 1.1),  "saturation": (0.9, 1.1),  "hue": (-0.03, 0.03)},
        "type_iv":   {"brightness": (0.75, 0.95), "saturation": (1.0, 1.3),  "hue": (0.02, 0.06)},
        "type_v":    {"brightness": (0.55, 0.75), "saturation": (1.1, 1.4),  "hue": (0.04, 0.08)},
        "type_vi":   {"brightness": (0.35, 0.55), "saturation": (1.2, 1.5),  "hue": (0.06, 0.10)},
    }

    def __init__(self, tone: str = "random"):
        
        self.tone = tone

    def __call__(self, img: Image.Image) -> Image.Image:
        profile_key = (
            random.choice(list(self.TONE_PROFILES.keys()))
            if self.tone == "random"
            else self.tone
        )
        p = self.TONE_PROFILES[profile_key]

        brightness = random.uniform(*p["brightness"])
        saturation = random.uniform(*p["saturation"])
        hue        = random.uniform(*p["hue"])

        return TF.adjust_brightness(
            TF.adjust_hue(
                TF.adjust_saturation(img, saturation),
                hue
            ),
            brightness
        )

class HairArtifact:
    

    def __init__(self, num_hairs: int = 6, thickness_range: tuple = (1, 3)):
        self.num_hairs = num_hairs
        self.thickness_range = thickness_range

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        count = random.randint(2, self.num_hairs)
        for _ in range(count):

            x1 = random.randint(0, w)
            y1 = random.randint(0, h)

            angle_x = random.randint(-w // 2, w // 2)
            angle_y = random.randint(-h // 4, h // 4)
            x2, y2 = x1 + angle_x, y1 + angle_y

            darkness = random.randint(5, 40)
            color = (darkness, darkness // 2, darkness // 3)
            width = random.randint(*self.thickness_range)

            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

        return img

class VignetteEffect:
    

    def __init__(self, strength: float = 0.5):
        self.strength = strength

    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img).astype(float)
        h, w = img_array.shape[:2]

        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - self.strength * (dist / max_dist) ** 1.5

        img_array = img_array * vignette[:, :, np.newaxis]
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

class GaussianNoise:
    

    def __init__(self, std_range: tuple = (5, 25)):
        self.std_range = std_range

    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img).astype(float)
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

class FocusBlur:
    

    def __init__(self, radius_range: tuple = (0.5, 2.0)):
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class SkinAugmentationPipeline:
    

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

        self.skin_tone_shift = SkinToneShift(tone="random")
        self.hair_artifact   = HairArtifact()
        self.vignette        = VignetteEffect(strength=random.uniform(0.3, 0.6))
        self.noise           = GaussianNoise()
        self.blur            = FocusBlur()

        self.geometric = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180),
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        ])

        self.normalize = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.to_tensor = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def apply_random(self, img: Image.Image) -> Image.Image:
        
        img = img.copy().convert("RGB")
        if random.random() > 0.4:
            img = self.skin_tone_shift(img)
        if random.random() > 0.5:
            img = self.hair_artifact(img)
        if random.random() > 0.6:
            img = self.vignette(img)
        if random.random() > 0.6:
            img = self.noise(img)
        if random.random() > 0.7:
            img = self.blur(img)
        img = self.geometric(img)
        return img

    def training_transform(self, img: Image.Image) -> torch.Tensor:
        
        img = self.apply_random(img)
        return self.to_tensor(img)

    def inference_transform(self, img: Image.Image) -> torch.Tensor:
        
        return self.to_tensor(img.convert("RGB"))

    def get_named_augmentations(self, img: Image.Image) -> dict:
        
        img = img.copy().convert("RGB")
        return {
            "Original": img,
            "Light Skin Tone (I/II)": SkinToneShift(tone="type_i_ii")(img.copy()),
            "Medium Skin Tone (III)": SkinToneShift(tone="type_iii")(img.copy()),
            "Olive Skin Tone (IV)":   SkinToneShift(tone="type_iv")(img.copy()),
            "Brown Skin Tone (V)":    SkinToneShift(tone="type_v")(img.copy()),
            "Dark Skin Tone (VI)":    SkinToneShift(tone="type_vi")(img.copy()),
            "Hair Artifact":          HairArtifact(num_hairs=8)(img.copy()),
            "Vignette + Lighting":    VignetteEffect()(img.copy()),
            "Image Noise":            GaussianNoise(std_range=(15, 30))(img.copy()),
            "Focus Blur":             FocusBlur(radius_range=(1.5, 2.5))(img.copy()),
            "Full Pipeline":          self.apply_random(img.copy()),
            "Full Pipeline (v2)":     self.apply_random(img.copy()),
        }
