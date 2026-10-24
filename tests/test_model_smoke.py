
import torch

from src.models.tinyvit_fusion import TinyViT_Gated_Fusion
from src.models.efficientnet_cbam import RGBDEfficientNet


def check_output(name, output, batch_size, num_classes):
    expected_shape = (batch_size, num_classes)

    assert tuple(output.shape) == expected_shape, (
        f"{name}: expected {expected_shape}, "
        f"got {tuple(output.shape)}"
    )

    assert torch.isfinite(output).all().item(), (
        f"{name}: output contains NaN or Inf"
    )

    print(f"[PASS] {name}")
    print(f"       Output shape: {tuple(output.shape)}")
    print(f"       Scores: {output[0].tolist()}")


def test_tinyvit():
    print("\nTesting TinyViT RGB-D Fusion...")

    model = TinyViT_Gated_Fusion(
        num_classes=7,
        pretrained=False,
    )
    model.eval()

    rgb = torch.rand(1, 3, 224, 224)
    depth = torch.rand(1, 1, 224, 224)

    with torch.inference_mode():
        output = model(rgb, depth)

    check_output("TinyViT", output, 1, 7)


def test_efficientnet():
    print("\nTesting EfficientNetV2 + CBAM...")

    model = RGBDEfficientNet(
        num_classes=7,
        pretrained=False,
    )
    model.eval()

    rgb = torch.rand(1, 3, 224, 224)
    depth = torch.rand(1, 1, 224, 224)

    rgbd = torch.cat((rgb, depth), dim=1)

    with torch.inference_mode():
        output = model(rgbd)

    check_output("EfficientNetV2", output, 1, 7)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(2)

    test_tinyvit()
    test_efficientnet()

    print("\nAll model smoke tests passed.")
