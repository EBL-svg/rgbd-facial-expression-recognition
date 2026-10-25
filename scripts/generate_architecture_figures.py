from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, xy, w, h, text, fontsize=10, fc="#F4F6F8", ec="#2F3B52", lw=1.5):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return box


def add_arrow(ax, start, end, text=None, fontsize=9):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.4,
        color="#333333",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.02, text, ha="center", va="bottom", fontsize=fontsize)
    return arrow


def save_figure(fig, filename):
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def draw_tinyvit_figure():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "TinyViT RGB-D Gated Fusion Architecture",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Inputs
    add_box(ax, (0.04, 0.70), 0.13, 0.10, "RGB Input\n(B, 3, 224, 224)", fc="#EAF3FF")
    add_box(ax, (0.04, 0.28), 0.13, 0.10, "Depth Input\n(B, 1, 224, 224)", fc="#EAF3FF")

    # Depth repeat
    add_box(ax, (0.22, 0.28), 0.14, 0.10, "Repeat depth\n1 → 3 channels", fc="#FFF4E6")

    # Encoders
    add_box(
        ax,
        (0.42, 0.66),
        0.16,
        0.14,
        "TinyViT RGB Encoder\npretrained=True/False\nnum_classes=0\nglobal_pool=''\nOutput: (B, 320, 7, 7)",
        fc="#E8F7EC",
    )
    add_box(
        ax,
        (0.42, 0.24),
        0.16,
        0.14,
        "TinyViT Depth Encoder\npretrained=True/False\nnum_classes=0\nglobal_pool=''\nOutput: (B, 320, 7, 7)",
        fc="#E8F7EC",
    )

    # Gate
    add_box(
        ax,
        (0.64, 0.24),
        0.14,
        0.14,
        "Gate module\nConv 1×1\n320 → 320\nSigmoid\nMask: (B, 320, 7, 7)",
        fc="#FCECEC",
    )

    add_box(
        ax,
        (0.64, 0.58),
        0.14,
        0.14,
        "Elementwise gating\nGated RGB =\nRGB features × mask\n(B, 320, 7, 7)",
        fc="#FCECEC",
    )

    # Tokenization
    add_box(
        ax,
        (0.80, 0.60),
        0.14,
        0.10,
        "Flatten + permute\nGated RGB tokens\n(B, 49, 320)",
        fc="#F4F0FF",
    )
    add_box(
        ax,
        (0.80, 0.26),
        0.14,
        0.10,
        "Flatten + permute\nDepth tokens\n(B, 49, 320)",
        fc="#F4F0FF",
    )

    # Fusion token and transformer
    add_box(
        ax,
        (0.80, 0.82),
        0.14,
        0.08,
        "Learnable fusion token\n(B, 1, 320)",
        fc="#FFFADB",
    )

    add_box(
        ax,
        (0.96, 0.48),
        0.18,
        0.18,
        "Concatenate tokens\n[B, 1+49+49, 320]\n=\nFusion token + RGB tokens + Depth tokens\n→ Transformer Encoder\n1 layer, 8 heads",
        fontsize=9,
        fc="#EAF7FA",
    )

    add_box(
        ax,
        (1.18, 0.48),
        0.14,
        0.18,
        "Select token 0\nLayerNorm\nLinear(320 → 7)\nOutput logits\n(B, 7)",
        fc="#EAF3FF",
    )

    # Arrows
    add_arrow(ax, (0.17, 0.75), (0.42, 0.75))
    add_arrow(ax, (0.17, 0.33), (0.22, 0.33))
    add_arrow(ax, (0.36, 0.33), (0.42, 0.33))
    add_arrow(ax, (0.58, 0.33), (0.64, 0.33))
    add_arrow(ax, (0.58, 0.73), (0.64, 0.65))
    add_arrow(ax, (0.78, 0.33), (0.78, 0.62), text="mask")
    add_arrow(ax, (0.78, 0.65), (0.80, 0.65))
    add_arrow(ax, (0.58, 0.31), (0.80, 0.31))
    add_arrow(ax, (0.87, 0.70), (0.96, 0.60))
    add_arrow(ax, (0.87, 0.31), (0.96, 0.54))
    add_arrow(ax, (0.87, 0.86), (0.96, 0.62))
    add_arrow(ax, (1.14, 0.57), (1.18, 0.57))

    # Notes
    ax.text(
        0.04,
        0.06,
        "Code-grounded details: depth is repeated from 1 to 3 channels; both streams use TinyViT-5M; "
        "depth features generate a gating mask; fusion uses a learnable token and a Transformer encoder.",
        fontsize=9,
        ha="left",
        va="center",
    )

    # extend limits because final boxes go beyond x=1
    ax.set_xlim(0, 1.35)

    save_figure(fig, "tinyvit_rgbd_gated_fusion.png")


def draw_efficientnet_figure():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 1.28)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "EfficientNetV2-S RGB-D + CBAM Architecture",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    add_box(ax, (0.04, 0.42), 0.14, 0.16, "RGB-D Input\n(B, 4, 224, 224)", fc="#EAF3FF")

    add_box(
        ax,
        (0.24, 0.40),
        0.16,
        0.20,
        "Projection layer\nConv 1×1\n4 → 3 channels\n+ ReLU",
        fc="#FFF4E6",
    )

    add_box(
        ax,
        (0.46, 0.36),
        0.18,
        0.28,
        "EfficientNetV2-S\nfeature extractor\n(pretrained ImageNet)\nbase_model.features\nOutput channels: 1280",
        fc="#E8F7EC",
    )

    add_box(
        ax,
        (0.70, 0.38),
        0.14,
        0.24,
        "CBAM Block\nChannel attention:\nAvgPool + MaxPool + MLP\nthen\nSpatial attention:\nConv on [avg, max]",
        fontsize=9,
        fc="#FCECEC",
    )

    add_box(
        ax,
        (0.90, 0.40),
        0.14,
        0.20,
        "Final conv\nConv 1×1\n1280 → 256\n+ ReLU",
        fc="#F4F0FF",
    )

    add_box(
        ax,
        (1.08, 0.42),
        0.14,
        0.16,
        "AdaptiveAvgPool2d(1)\nFlatten",
        fc="#FFFADB",
    )

    add_box(
        ax,
        (1.08, 0.14),
        0.14,
        0.18,
        "Classifier\nLinear(256 → 128)\nReLU\nLinear(128 → 7)\nOutput logits",
        fc="#EAF7FA",
    )

    add_arrow(ax, (0.18, 0.50), (0.24, 0.50))
    add_arrow(ax, (0.40, 0.50), (0.46, 0.50))
    add_arrow(ax, (0.64, 0.50), (0.70, 0.50))
    add_arrow(ax, (0.84, 0.50), (0.90, 0.50))
    add_arrow(ax, (1.04, 0.50), (1.08, 0.50))
    add_arrow(ax, (1.15, 0.42), (1.15, 0.32))

    ax.text(
        0.04,
        0.08,
        "Code-grounded details: the model first projects the 4-channel RGB-D tensor to 3 channels, "
        "extracts features with EfficientNetV2-S, applies CBAM attention, then classifies 7 emotions.",
        fontsize=9,
        ha="left",
        va="center",
    )

    save_figure(fig, "efficientnetv2_cbam_rgbd.png")


def main():
    draw_tinyvit_figure()
    draw_efficientnet_figure()
    print("Architecture figures generated successfully.")


if __name__ == "__main__":
    main()