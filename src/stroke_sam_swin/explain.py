from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def grad_cam_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """Grad-CAM class-activation map (Selvaraju et al.) on the last Swin stage.

    A forward hook captures the last block's post-norm output (B, H, W, C); the
    class logit gradient is pooled per channel and used to weight the activations,
    then upsampled to the input resolution. Implemented with raw hooks so it works
    with the Swin transformer's (H, W, C) token layout.
    """
    model.eval()
    image = image.to(device)

    activation: dict[str, torch.Tensor] = {}
    gradient: dict[str, torch.Tensor] = {}

    def _forward_hook_fn(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activation["value"] = output

    def _backward_hook_fn(
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        gradient["value"] = grad_output[0]

    layer = model.layers[-1].blocks[-1].norm2
    forward_handle = layer.register_forward_hook(_forward_hook_fn)
    backward_handle = layer.register_full_backward_hook(_backward_hook_fn)

    image = image.detach().clone().requires_grad_(True)
    output = model(image)
    output[0, target_class].backward(retain_graph=True)
    forward_handle.remove()
    backward_handle.remove()

    if "value" not in activation or "value" not in gradient:
        raise RuntimeError("No activations/gradients captured for Grad-CAM target layer.")

    acts = activation["value"]  # (B, L, C) token sequence
    grads = gradient["value"]  # (B, L, C)

    batch, length, channels = acts.shape
    side = int(round(length ** 0.5))
    acts = acts.reshape(batch, side, side, channels).permute(0, 3, 1, 2)  # (B, C, H, W)
    grads = grads.reshape(batch, side, side, channels).permute(0, 3, 1, 2).clamp_min(0.0)

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam,
        size=(image.shape[2], image.shape[3]),
        mode="bilinear",
        align_corners=False,
    )

    heatmap = cam.squeeze().detach().cpu().numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def shap_pixel_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    original: np.ndarray,
    target_class: int,
    device: torch.device,
    grid_size: int = 8,
    n_samples: int = 3,
) -> np.ndarray:
    """SHAP pixel-level attribution via Shapley value sampling (captum).

    The image is split into a grid of superpixel regions; Shapley values are
    estimated by permutation sampling over those regions and the attribution is
    returned at full input resolution. This is the practical, CPU-friendly
    adaptation of SHAP for 224x224 images (the shap library's pixel hierarchy is
    intractable at this resolution).
    """
    from captum.attr import ShapleyValueSampling

    model.eval()
    image = image.to(device)

    _, _, height, width = image.shape
    step_y = int(np.ceil(height / grid_size))
    step_x = int(np.ceil(width / grid_size))
    ys, xs = np.mgrid[0:height, 0:width]
    regions = (xs // step_x + (ys // step_y) * grid_size).astype(np.int64)
    feature_mask = (
        torch.tensor(regions, dtype=torch.long, device=device)
        .unsqueeze(0)
        .repeat(1, image.shape[1], 1, 1)
    )

    explainer = ShapleyValueSampling(model)
    attributions = explainer.attribute(
        image,
        target=target_class,
        feature_mask=feature_mask,
        n_samples=n_samples,
    )

    heatmap = np.abs(attributions.detach().cpu().numpy()[0]).sum(axis=0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def lime_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
    grid_size: int = 6,
    n_samples: int = 150,
    kernel_width: float = 25.0,
    ridge_alpha: float = 0.05,
) -> np.ndarray:
    """LIME-style explanation: local linear surrogate model over superpixels.

    Follows the LIME (Local Interpretable Model-Agnostic Explanations) algorithm:
    the image is segmented into a grid of superpixels, random subsets are masked
    with a blurred background, and a distance-weighted ridge regression is fitted
    on the model's target-class probability. The fitted coefficients become the
    per-superpixel attribution map.
    """
    model.eval()
    image = image.to(device)

    _, _, height, width = image.shape
    step_y = int(np.ceil(height / grid_size))
    step_x = int(np.ceil(width / grid_size))
    ys, xs = np.mgrid[0:height, 0:width]
    region = xs // step_x + (ys // step_y) * grid_size  # (H, W) region ids
    n_features = grid_size * grid_size

    fill = F.avg_pool2d(image, kernel_size=grid_size, stride=grid_size)  # (1, C, g, g)
    fill_full = torch.zeros_like(image)
    reg = torch.tensor(region, device=device)
    for row in range(grid_size):
        for col in range(grid_size):
            region_id = row * grid_size + col
            pixels = reg == region_id
            fill_full[0, :, pixels] = fill[0, :, row, col].unsqueeze(1)

    present = torch.tensor(region, device=device)  # reuse as id map

    rng = np.random.RandomState(42)
    perturbed = rng.randint(0, 2, size=(n_samples - 1, n_features)).astype(np.float32)
    perturbed = np.vstack([np.ones((1, n_features), dtype=np.float32), perturbed])
    n_total = perturbed.shape[0]

    probabilities = np.zeros(n_total, dtype=np.float32)
    with torch.no_grad():
        for index in range(n_total):
            keep = perturbed[index][present] == 1.0  # (H, W) present features
            masked = image.clone()
            masked[0, :, keep] = image[0, :, keep]
            masked[0, :, ~keep] = fill_full[0, :, ~keep]
            probs = torch.softmax(model(masked), dim=1)[0, target_class].item()
            probabilities[index] = probs

    distances = 1.0 - perturbed.mean(axis=1)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    feature_means = perturbed.mean(axis=0)
    feature_stds = perturbed.std(axis=0)
    feature_stds[feature_stds == 0] = 1.0
    design = (perturbed - feature_means) / feature_stds
    design = np.hstack([np.ones((n_total, 1)), design])

    weighted = np.diag(weights)
    xtwx = design.T @ weighted @ design + ridge_alpha * np.eye(design.shape[1])
    xtwy = design.T @ weighted @ probabilities
    beta = np.linalg.solve(xtwx, xtwy)
    coefficients = beta[1:]

    heatmap = np.zeros((height, width), dtype=np.float32)
    for feature_id in range(n_features):
        heatmap[region == feature_id] = coefficients[feature_id]

    heatmap = np.maximum(heatmap, 0.0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def build_combined_mask(
    heatmaps: dict[str, np.ndarray],
    threshold_ratio: float = 0.5,
) -> np.ndarray:
    """Consensus mask from LIME + Grad-CAM + SHAP attributions.

    Averages the normalized maps and thresholds the result to a binary mask
    highlighting regions that all methods consistently consider important.
    """
    sources = [heatmaps[key] for key in ("lime", "gradcam", "shap") if heatmaps.get(key) is not None]
    if not sources:
        if not heatmaps:
            return None
        return np.zeros_like(next(iter(heatmaps.values())), dtype=bool)
    consensus = np.mean(np.stack(sources), axis=0)
    if consensus.max() > 0:
        return consensus >= threshold_ratio * consensus.max()
    return np.zeros_like(consensus, dtype=bool)


def save_explanations(
    original: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    output_path: str | Path,
    is_normal: bool = False,
) -> dict[str, str]:
    """Persist per-method heatmaps, overlays, a consensus mask and a 6-panel figure.

    Panels: Input | LIME | Grad-CAM / SHAP | Combined Mask | Final Overlay.
    Falls back gracefully when not all methods are present. When the prediction is
    normal (no stroke), the consensus highlight is drawn in blue instead of red so
    it is not mistaken for a stroke region.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    highlight_rgb = np.array([0, 120, 255]) if is_normal else np.array([255, 0, 0])
    region_label = "Analyzed Region" if is_normal else "Stroke Region"
    overlay_label = "Analysis Overlay" if is_normal else "Final Overlay"

    base_name = output_path.stem.replace("_xai", "")
    input_path = output_path.with_name(f"{base_name}_input.png")
    Image.fromarray(np.uint8(original)).save(input_path)

    artifacts: dict[str, str] = {"combined": str(output_path), "input": str(input_path)}

    panels: list[tuple[str, np.ndarray]] = [("Input", np.uint8(original))]

    perturbation_key = "lime" if heatmaps.get("lime") is not None else "occlusion"
    perturbation_label = "LIME" if perturbation_key == "lime" else "Occlusion"

    for key, label in (
        (perturbation_key, perturbation_label),
        ("gradcam", "Grad-CAM"),
        ("shap", "SHAP"),
    ):
        heatmap = heatmaps.get(key)
        if heatmap is None:
            continue
        heatmap_u8 = np.uint8(255 * heatmap)
        colored = cv2.cvtColor(cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.55 * original + 0.45 * colored)

        heatmap_path = output_path.with_name(f"{base_name}_{key}_heatmap.png")
        overlay_path = output_path.with_name(f"{base_name}_{key}_overlay.png")
        Image.fromarray(np.uint8(colored)).save(heatmap_path)
        Image.fromarray(np.uint8(overlay)).save(overlay_path)
        artifacts[f"{key}_heatmap"] = str(heatmap_path)
        artifacts[f"{key}_overlay"] = str(overlay_path)
        panels.append((label, overlay))

    mask = build_combined_mask(heatmaps)
    if mask.any():
        grayscale = cv2.cvtColor(np.uint8(original), cv2.COLOR_RGB2GRAY)
        grayscale_rgb = np.stack([grayscale] * 3, axis=-1)
        mask_view = np.where(mask[..., None], highlight_rgb, grayscale_rgb).astype(np.uint8)
        mask_overlay = np.uint8(0.6 * original + 0.4 * highlight_rgb * mask[..., None])

        mask_path = output_path.with_name(f"{base_name}_combined_mask.png")
        overlay_path = output_path.with_name(f"{base_name}_combined_overlay.png")
        Image.fromarray(mask_view).save(mask_path)
        Image.fromarray(mask_overlay).save(overlay_path)
        artifacts["combined_mask"] = str(mask_path)
        artifacts["combined_overlay"] = str(overlay_path)
        panels.append((region_label, mask_view))
        panels.append((overlay_label, mask_overlay))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return artifacts


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

    xai = config.get("xai", {})
    heatmaps: dict[str, np.ndarray] = {}

    if xai.get("lime", {}).get("enabled", True):
        try:
            lime_settings = xai.get("lime", {})
            heatmaps["lime"] = lime_map(
                model,
                image_tensor,
                target_class,
                device,
                grid_size=int(lime_settings.get("grid_size", 6)),
                n_samples=int(lime_settings.get("n_samples", 150)),
            )
        except Exception as exc:
            print(f"LIME failed: {exc}")
    else:
        heatmaps["occlusion"] = heatmap

    if xai.get("grad_cam", {}).get("enabled", True):
        try:
            heatmaps["gradcam"] = grad_cam_map(model, image_tensor, target_class, device)
        except Exception as exc:
            print(f"Grad-CAM failed: {exc}")
    if xai.get("shap", {}).get("enabled", True):
        try:
            shap_settings = xai.get("shap", {})
            heatmaps["shap"] = shap_pixel_map(
                model,
                image_tensor,
                original,
                target_class,
                device,
                grid_size=int(shap_settings.get("grid_size", 8)),
                n_samples=int(shap_settings.get("n_samples", 3)),
            )
        except Exception as exc:
            print(f"SHAP failed: {exc}")

    combined = build_combined_mask(heatmaps)
    if combined is not None and combined.any():
        artifacts = save_explanations(
            original,
            heatmaps,
            output_path,
            is_normal=(classes[target_class] == "normal"),
        )
    else:
        artifacts = save_overlay(original, heatmap, output_path)

    print(f"Prediction: {classes[target_class]} ({probs[target_class].item():.4f})")
    print(f"Saved explanation: {output_path}")
    for key, path in artifacts.items():
        print(f"  {key}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explain(args.config, args.checkpoint, args.image)
