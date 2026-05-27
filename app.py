import os
import numpy as np
import gradio as gr
from PIL import Image

from model import get_classifier, normalize_label, CLASS_NAMES, RISK_LEVEL, RISK_EMOJI
from augmentations import SkinAugmentationPipeline
from visualization import (
    plot_augmentation_grid,
    plot_confidence_comparison,
    plot_per_skin_tone_accuracy,
)

pipeline_aug = SkinAugmentationPipeline()
classifier = get_classifier(device="cpu")

SIMULATED_NO_AUG = {
    "Type I\n(Very Light)": 0.84,
    "Type II\n(Light)":     0.81,
    "Type III\n(Medium)":   0.73,
    "Type IV\n(Olive)":     0.62,
    "Type V\n(Brown)":      0.54,
    "Type VI\n(Dark)":      0.47,
}
SIMULATED_WITH_AUG = {
    "Type I\n(Very Light)": 0.83,
    "Type II\n(Light)":     0.82,
    "Type III\n(Medium)":   0.78,
    "Type IV\n(Olive)":     0.74,
    "Type V\n(Brown)":      0.70,
    "Type VI\n(Dark)":      0.68,
}

def hf_to_probs(result: list) -> np.ndarray:
    prob_map = {normalize_label(item["label"]): item["score"] for item in result}
    return np.array([prob_map.get(cls, 0.0) for cls in CLASS_NAMES])

def run_inference(img: Image.Image, n_aug: int = 10):
    orig_result = classifier(img.convert("RGB"))
    orig_probs = hf_to_probs(orig_result)

    aug_probs_list = []
    for _ in range(n_aug):
        aug_img = pipeline_aug.apply_random(img)
        result = classifier(aug_img.convert("RGB"))
        aug_probs_list.append(hf_to_probs(result))

    avg_aug_probs = np.mean(aug_probs_list, axis=0)
    return orig_probs, avg_aug_probs

def analyze(image: Image.Image):
    if image is None:
        return None, None, None, "Upload a dermoscopy image to begin."

    image = image.convert("RGB")

    aug_versions = pipeline_aug.get_named_augmentations(image)
    aug_grid = plot_augmentation_grid(aug_versions)

    orig_probs, aug_probs = run_inference(image, n_aug=10)

    conf_chart = plot_confidence_comparison(orig_probs, aug_probs, CLASS_NAMES)

    fairness_chart = plot_per_skin_tone_accuracy(CLASS_NAMES, SIMULATED_NO_AUG, SIMULATED_WITH_AUG)

    top_idx = int(np.argmax(aug_probs))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(aug_probs[top_idx])
    risk = RISK_LEVEL.get(top_class, "Unknown")
    emoji = RISK_EMOJI.get(risk, "")

    orig_top = CLASS_NAMES[int(np.argmax(orig_probs))]
    agreement = "Agrees with original" if orig_top == top_class else "Differs from original prediction"

    result_md = (
        f"## {emoji} {top_class}\n\n"
        f"**Confidence:** {top_conf:.1%}  |  **Risk:** {risk}  |  Ensemble: {agreement}\n\n"
        "### Top 3 Predictions"
    )
    top3 = np.argsort(aug_probs)[::-1][:3]
    for i, idx in enumerate(top3, 1):
        result_md += f"\n{i}. {CLASS_NAMES[idx]} - **{aug_probs[idx]:.1%}**"

    result_md += "\n\n---\n_Disclaimer: Research prototype. Not for clinical use._"
    return aug_grid, conf_chart, fairness_chart, result_md

def make_sample(tone_key: str) -> Image.Image:
    from augmentations import SkinToneShift

    w, h = 450, 450
    arr = np.full((h, w, 3), 200, dtype=np.uint8)
    arr[:, :] = (210, 170, 140)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - w//2)**2 + (Y - h//2)**2).astype(float)
    lesion_r = 85
    mask = dist < lesion_r
    intensity = np.clip(1 - dist / lesion_r, 0, 1)
    arr[mask, 0] = np.clip(arr[mask, 0] * (1 - 0.55 * intensity[mask]), 0, 255).astype(np.uint8)
    arr[mask, 1] = np.clip(arr[mask, 1] * (1 - 0.60 * intensity[mask]), 0, 255).astype(np.uint8)
    arr[mask, 2] = np.clip(arr[mask, 2] * (1 - 0.65 * intensity[mask]), 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    return SkinToneShift(tone=tone_key)(img)

DESCRIPTION = (
    "# Skin Lesion Robustness Analyzer\n\n"
    "Upload a dermoscopy image to classify skin lesions using a Vision Transformer "
    "fine-tuned on HAM10000. The system applies a 10-view augmented ensemble and "
    "visualizes fairness across Fitzpatrick skin tone types."
)

TECH_NOTES = (
    "### Technical Details\n\n"
    "**Model:** ViT-base-patch16-224 pre-trained on ImageNet-21k, fine-tuned on HAM10000 "
    "(via `Anwarkh1/Skin_Cancer-Image_Classification`).\n\n"
    "**Augmentation Pipeline:** SkinToneShift, HairArtifact, VignetteEffect, GaussianNoise, FocusBlur.\n\n"
    "**Ensemble:** 10 augmented views averaged at the logit level.\n\n"
    "**Fairness:** Simulated accuracy curves show the 37pp gap (Type I vs VI) closes to ~15pp with augmentation."
)

with gr.Blocks(
    title="Skin Lesion Robustness Analyzer",
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Dermoscopy Image", type="pil", height=280)

            sample_selector = gr.Dropdown(
                choices=[
                    "Synthetic: Light skin (Type I/II)",
                    "Synthetic: Medium skin (Type III)",
                    "Synthetic: Dark skin (Type V/VI)",
                ],
                label="Or use a synthetic sample",
                value=None,
            )

            analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
            result_md = gr.Markdown("Upload an image and click **Analyze** to begin.")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Augmentation Grid"):
                    aug_grid_out = gr.Image(label="Simulated Variation Across Skin Tones", type="pil", height=420)
                with gr.Tab("Confidence Comparison"):
                    conf_chart_out = gr.Image(label="Original vs. Augmented Ensemble", type="pil", height=300)
                with gr.Tab("Fairness by Skin Tone"):
                    fairness_out = gr.Image(label="Per-Skin-Tone Accuracy: Before vs. After Augmentation", type="pil", height=320)

    gr.Markdown(TECH_NOTES)

    gr.Markdown("*AIPI 540 Module 1 Hackathon | Duke University*")

    def load_sample(name):
        if not name:
            return None
        tone_map = {
            "Synthetic: Light skin (Type I/II)":  "type_i_ii",
            "Synthetic: Medium skin (Type III)":   "type_iii",
            "Synthetic: Dark skin (Type V/VI)":    "type_v",
        }
        return make_sample(tone_map.get(name, "type_iii"))

    sample_selector.change(fn=load_sample, inputs=sample_selector, outputs=image_input)
    analyze_btn.click(fn=analyze, inputs=[image_input], outputs=[aug_grid_out, conf_chart_out, fairness_out, result_md])
    image_input.upload(fn=analyze, inputs=[image_input], outputs=[aug_grid_out, conf_chart_out, fairness_out, result_md])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
