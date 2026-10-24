
import random

import numpy as np
import torch
from PIL import Image

from src.data.rgbd_dataset import PairedTransform


def create_synthetic_pair():
    """Create two spatially aligned synthetic images."""
    height, width = 180, 260

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[40:140, 65:195] = 255

    rgb = np.stack([mask, mask, mask], axis=-1)

    return Image.fromarray(rgb), Image.fromarray(mask)


def test_validation_transform():
    rgb, depth = create_synthetic_pair()

    transform = PairedTransform(size=224, train=False)
    rgb_tensor, depth_tensor = transform(rgb, depth)

    assert rgb_tensor.shape == (3, 224, 224)
    assert depth_tensor.shape == (1, 224, 224)

    assert torch.isfinite(rgb_tensor).all()
    assert torch.isfinite(depth_tensor).all()

    assert 0 <= depth_tensor.min().item()
    assert depth_tensor.max().item() <= 1

    # Undo ImageNet normalization for alignment comparison.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    rgb_restored = rgb_tensor * std + mean

    rgb_mask = rgb_restored[0] > 0.5
    depth_mask = depth_tensor[0] > 0.5

    assert torch.equal(rgb_mask, depth_mask), (
        "RGB and depth are not spatially aligned."
    )

    print("[PASS] Validation transform")


def test_training_transform():
    rgb, depth = create_synthetic_pair()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    transform = PairedTransform(size=224, train=True)
    rgb_tensor, depth_tensor = transform(rgb, depth)

    assert rgb_tensor.shape == (3, 224, 224)
    assert depth_tensor.shape == (1, 224, 224)
    assert torch.isfinite(rgb_tensor).all()
    assert torch.isfinite(depth_tensor).all()

    print("[PASS] Training transform")


def test_deterministic_validation():
    rgb, depth = create_synthetic_pair()

    transform = PairedTransform(size=224, train=False)

    first_rgb, first_depth = transform(rgb, depth)
    second_rgb, second_depth = transform(rgb, depth)

    assert torch.equal(first_rgb, second_rgb)
    assert torch.equal(first_depth, second_depth)

    print("[PASS] Deterministic validation")


if __name__ == "__main__":
    test_validation_transform()
    test_training_transform()
    test_deterministic_validation()

    print("\nAll preprocessing tests passed.")
