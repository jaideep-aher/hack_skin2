

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import io
from typing import Dict, List

def fig_to_pil(fig: plt.Figure) -> Image.Image:
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()

def plot_augmentation_grid(augmented_versions: Dict[str, Image.Image]) -> Image.Image:
    
    items = list(augmented_versions.items())
    n = len(items)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 0.5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Augmentation Pipeline - Simulating Real-World Variation",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    axes = np.array(axes).flatten()

    for i, (name, img) in enumerate(items):
        ax = axes[i]
        display = img.resize((224, 224), Image.LANCZOS)
        ax.imshow(display)
        ax.set_title(name, color="white", fontsize=8, pad=4)
        ax.axis("off")

        if "Skin Tone" in name or "Tone" in name:
            for spine in ax.spines.values():
                spine.set_edgecolor("#4ade80")
                spine.set_linewidth(2)
                spine.set_visible(True)
        elif "Full Pipeline" in name:
            for spine in ax.spines.values():
                spine.set_edgecolor("#60a5fa")
                spine.set_linewidth(2)
                spine.set_visible(True)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4ade80", label="Skin tone simulation"),
        Patch(facecolor="#60a5fa", label="Full pipeline"),
        Patch(facecolor="#888", label="Single augmentation"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               facecolor="#1e2130", edgecolor="none",
               labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result

def plot_confidence_comparison(
    orig_probs: np.ndarray,
    aug_probs: np.ndarray,
    class_names: List[str],
) -> Image.Image:
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0f1117")

    colors_orig = ["#ef4444" if p == max(orig_probs) else "#475569" for p in orig_probs]
    colors_aug  = ["#22c55e" if p == max(aug_probs) else "#475569" for p in aug_probs]

    def style_ax(ax, title):
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")
        ax.grid(axis="x", color="#374151", linewidth=0.5, alpha=0.7)

    short_names = [n.split()[-1] if len(n) > 12 else n for n in class_names]

    bars1 = ax1.barh(short_names, orig_probs, color=colors_orig, height=0.6, zorder=2)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Confidence", color="#9ca3af", fontsize=8)
    style_ax(ax1, "Original Image")
    for bar, prob in zip(bars1, orig_probs):
        ax1.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                 f"{prob:.1%}", va="center", color="white", fontsize=7)

    bars2 = ax2.barh(short_names, aug_probs, color=colors_aug, height=0.6, zorder=2)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence", color="#9ca3af", fontsize=8)
    style_ax(ax2, "After Augmentation (ensemble)")
    for bar, prob in zip(bars2, aug_probs):
        ax2.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                 f"{prob:.1%}", va="center", color="white", fontsize=7)

    delta = aug_probs - orig_probs
    colors_delta = ["#22c55e" if d >= 0 else "#ef4444" for d in delta]
    ax3.barh(short_names, delta, color=colors_delta, height=0.6, zorder=2)
    ax3.axvline(x=0, color="white", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Change in Confidence", color="#9ca3af", fontsize=8)
    style_ax(ax3, "📊 Confidence Shift")

    fig.suptitle("Model Confidence: Original vs. Augmentation-Robust Ensemble",
                 color="white", fontsize=11, fontweight="bold")

    plt.tight_layout(pad=1.2)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result

def plot_training_curves(
    train_acc_no_aug: List[float],
    train_acc_with_aug: List[float],
    val_acc_no_aug: List[float],
    val_acc_with_aug: List[float],
) -> Image.Image:
    
    epochs = range(1, len(train_acc_no_aug) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1117")

    def style_ax(ax, title):
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="white")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color="#9ca3af")
        ax.set_ylabel("Accuracy", color="#9ca3af")
        ax.legend(facecolor="#374151", edgecolor="none", labelcolor="white", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")
        ax.grid(color="#374151", linewidth=0.5, alpha=0.5)

    ax1.plot(epochs, train_acc_no_aug, color="#ef4444", linewidth=2,
             linestyle="--", label="No augmentation (train)")
    ax1.plot(epochs, train_acc_with_aug, color="#22c55e", linewidth=2,
             label="With augmentation (train)")
    style_ax(ax1, "Training Accuracy")

    ax2.plot(epochs, val_acc_no_aug, color="#ef4444", linewidth=2,
             linestyle="--", label="No augmentation (val)")
    ax2.plot(epochs, val_acc_with_aug, color="#22c55e", linewidth=2,
             label="With augmentation (val)")

    final_gap = val_acc_with_aug[-1] - val_acc_no_aug[-1]
    ax2.annotate(
        f"+{final_gap:.1%} generalization\nfrom augmentation",
        xy=(len(epochs), val_acc_with_aug[-1]),
        xytext=(len(epochs) - 3, val_acc_with_aug[-1] - 0.08),
        color="#22c55e", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#22c55e"),
    )
    style_ax(ax2, "Validation Accuracy (Generalization)")

    fig.suptitle("Impact of Skin-Specific Augmentations on Model Generalization",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    result = fig_to_pil(fig)
    plt.close(fig)
    return result

def plot_per_skin_tone_accuracy(
    class_names: List[str],
    acc_no_aug: Dict[str, float],
    acc_with_aug: Dict[str, float],
) -> Image.Image:
    
    tones = list(acc_no_aug.keys())
    no_aug_vals = [acc_no_aug[t] for t in tones]
    with_aug_vals = [acc_with_aug[t] for t in tones]

    x = np.arange(len(tones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")

    bars1 = ax.bar(x - width / 2, no_aug_vals, width, label="No augmentation",
                   color="#ef4444", alpha=0.85, zorder=2)
    bars2 = ax.bar(x + width / 2, with_aug_vals, width, label="With skin augmentation",
                   color="#22c55e", alpha=0.85, zorder=2)

    ax.set_xlabel("Fitzpatrick Skin Type", color="#9ca3af", fontsize=10)
    ax.set_ylabel("Accuracy", color="#9ca3af", fontsize=10)
    ax.set_title("Accuracy by Skin Tone: Before vs. After Augmentation\n(Fairness Improvement)",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tones, color="white", rotation=15, ha="right")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.0)
    ax.legend(facecolor="#374151", edgecolor="none", labelcolor="white", fontsize=9)
    ax.grid(axis="y", color="#374151", linewidth=0.5, alpha=0.5, zorder=1)

    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", va="bottom", color="white", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", va="bottom", color="white", fontsize=7)

    plt.tight_layout()
    result = fig_to_pil(fig)
    plt.close(fig)
    return result
