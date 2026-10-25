
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Output directory
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Colors
BLUE = "#E3F0FF"
GREEN = "#E5F5E9"
ORANGE = "#FFF0D9"
PURPLE = "#EEE7FF"
GRAY = "#F1F3F5"


def setup(title, height=11):
    fig, ax = plt.subplots(figsize=(11, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)
    return fig, ax


def box(ax, x, y, w, h, title, detail="", color=BLUE):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.3,
        edgecolor="#526277",
        facecolor=color
    )
    ax.add_patch(patch)

    text = title if not detail else f"{title}\n{detail}"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.4,
        color="#172033"
    )


def arrow(ax, x1, y1, x2, y2):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color="#526277"
    )
    ax.add_patch(patch)


def save(fig, name):
    fig.savefig(
        OUTPUT / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        OUTPUT / f"{name}.svg",
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)
    print(f"[OK] {name}.png and {name}.svg")


def draw_tinyvit():
    fig, ax = setup("TinyViT RGB-D Gated Fusion", height=12)

    # Two independent input branches
    box(ax, 0.6, 10.5, 3.8, 0.85,
        "RGB Image", "(B, 3, 224, 224)", BLUE)
    box(ax, 5.6, 10.5, 3.8, 0.85,
        "Depth Map", "(B, 1, 224, 224)", BLUE)

    box(ax, 0.6, 8.9, 3.8, 0.85,
        "TinyViT RGB Encoder",
        "(B, 320, 7, 7)", GREEN)
    box(ax, 5.6, 8.9, 3.8, 0.85,
        "Repeat 1 to 3 channels",
        "TinyViT Depth Encoder", GREEN)

    arrow(ax, 2.5, 10.5, 2.5, 9.75)
    arrow(ax, 7.5, 10.5, 7.5, 9.75)

    # Both branches enter the fusion stage
    box(ax, 2.0, 7.15, 6.0, 0.95,
        "Depth-Gated RGB Features",
        "Gate = Sigmoid(Conv1x1(Depth)) | RGB x Gate", ORANGE)

    arrow(ax, 2.5, 8.9, 3.8, 8.1)
    arrow(ax, 7.5, 8.9, 6.2, 8.1)

    box(ax, 2.0, 5.4, 6.0, 0.95,
        "RGB and Depth Tokenization",
        "49 gated RGB tokens + 49 depth tokens", PURPLE)

    arrow(ax, 5, 7.15, 5, 6.35)

    box(ax, 2.0, 3.65, 6.0, 0.95,
        "Token Concatenation",
        "1 learnable fusion token + 98 tokens = 99 tokens", PURPLE)

    arrow(ax, 5, 5.4, 5, 4.6)

    box(ax, 2.0, 1.9, 6.0, 0.95,
        "Transformer Encoder",
        "1 layer | 8 attention heads | 320 features", GREEN)

    arrow(ax, 5, 3.65, 5, 2.85)

    box(ax, 2.0, 0.2, 6.0, 0.9,
        "Classification Head",
        "Fusion token -> LayerNorm -> Linear -> 7 logits", BLUE)

    arrow(ax, 5, 1.9, 5, 1.1)

    save(fig, "tinyvit_rgbd_gated_fusion")


def draw_efficientnet():
    fig, ax = setup("EfficientNetV2-S RGB-D + CBAM", height=12)

    stages = [
        ("RGB-D Input", "(B, 4, 224, 224)", BLUE),
        ("Input Projection", "Conv 1x1: 4 to 3 channels + ReLU", ORANGE),
        ("EfficientNetV2-S", "Feature extraction | 1280 channels", GREEN),
        ("CBAM Attention", "Channel attention -> Spatial attention", PURPLE),
        ("Convolutional Head", "Conv 1x1: 1280 to 256 + ReLU", ORANGE),
        ("Global Average Pooling", "AdaptiveAvgPool2d(1) + Flatten", GRAY),
        ("Classification", "Linear 256 -> 128 -> 7 logits", BLUE),
    ]

    y_positions = [10.4, 8.7, 7.0, 5.3, 3.6, 1.9, 0.2]

    for i, ((title, detail, color), y) in enumerate(
        zip(stages, y_positions)
    ):
        box(ax, 2.0, y, 6.0, 0.95, title, detail, color)

        if i < len(stages) - 1:
            arrow(ax, 5, y, 5, y_positions[i + 1] + 0.95)

    save(fig, "efficientnetv2_cbam_rgbd")


if __name__ == "__main__":
    draw_tinyvit()
    draw_efficientnet()
    print("Both architecture diagrams generated successfully.")
