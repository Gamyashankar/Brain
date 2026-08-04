from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .model import load_checkpoint
from .utils import ensure_dir, get_device, load_config


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(path: str | Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    original = np.asarray(image.resize((image_size, image_size)))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
        ]
    )
    return transform(image).unsqueeze(0), original


def occlusion_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        baseline_prob = torch.softmax(model(image), dim=1)[0, target_class].item()

    _, _, height, width = image.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    fill_value = ((0.0 - IMAGENET_MEAN) / IMAGENET_STD).to(device)
    positions = [(y, x) for y in range(0, height, stride) for x in range(0, width, stride)]

    for y, x in tqdm(positions, leave=False):
        occluded = image.clone()
        y2 = min(y + patch_size, height)
        x2 = min(x + patch_size, width)
        occluded[:, :, y:y2, x:x2] = fill_value

        with torch.no_grad():
            prob = torch.softmax(model(occluded), dim=1)[0, target_class].item()

        drop = max(0.0, baseline_prob - prob)
        heatmap[y:y2, x:x2] += drop
        counts[y:y2, x:x2] += 1.0

    heatmap = heatmap / np.maximum(counts, 1.0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def save_overlay(original: np.ndarray, heatmap: np.ndarray, output_path: str | Path) -> dict[str, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap_u8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * original + 0.45 * colored)

    base_name = output_path.stem.replace("_xai", "")
    input_path = output_path.with_name(f"{base_name}_input.png")
    heatmap_path = output_path.with_name(f"{base_name}_heatmap.png")
    overlay_path = output_path.with_name(f"{base_name}_overlay.png")

    Image.fromarray(np.uint8(original)).save(input_path)
    Image.fromarray(np.uint8(colored)).save(heatmap_path)
    Image.fromarray(np.uint8(overlay)).save(overlay_path)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.axis("off")
    plt.title("Input")
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.title("Swin heatmap")
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Overlay")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return {
        "combined": str(output_path),
        "input": str(input_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
    }


def explain(config_path: str, checkpoint_path: str, image_path: str) -> None:
    config = load_config(config_path)
    device = get_device()
    model, classes, _ = load_checkpoint(checkpoint_path, config, device)

    image_size = int(config["dataset"]["image_size"])
    image_tensor, original = preprocess_image(image_path, image_size)
    with torch.no_grad():
        probs = torch.softmax(model(image_tensor.to(device)), dim=1)[0]
        target_class = int(torch.argmax(probs).item())

    heatmap = occlusion_map(
        model,
        image_tensor,
        target_class=target_class,
        patch_size=int(config["xai"]["occlusion_patch_size"]),
        stride=int(config["xai"]["occlusion_stride"]),
        device=device,
    )

    output_dir = ensure_dir(config["outputs"]["explanations"])
    stem = Path(image_path).stem
    output_path = Path(output_dir) / f"{stem}_{classes[target_class]}_xai.png"
    save_overlay(original, heatmap, output_path)
    print(f"Prediction: {classes[target_class]} ({probs[target_class].item():.4f})")
    print(f"Saved explanation: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explain(args.config, args.checkpoint, args.image)
