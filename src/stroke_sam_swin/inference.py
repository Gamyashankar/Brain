from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

_MODEL_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, list[str], dict[str, Any]]] = {}

from .explain import (
    build_combined_mask,
    grad_cam_map,
    lime_map,
    occlusion_map,
    preprocess_image,
    save_explanations,
    save_overlay,
    shap_pixel_map,
)
from .localization import localize_lesion
from .model import load_checkpoint
from .utils import ensure_dir, get_device, load_config


def checkpoint_ready(checkpoint_path: str | Path) -> bool:
    return Path(checkpoint_path).exists()


def load_cached_checkpoint(config_path: str | Path, checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    config = load_config(config_path)
    checkpoint_path = Path(checkpoint_path)
    cache_key = (str(checkpoint_path), str(device))
    if cache_key not in _MODEL_CACHE:
        model, classes, checkpoint = load_checkpoint(str(checkpoint_path), config, device)
        _MODEL_CACHE[cache_key] = (model, classes, checkpoint)
    model, classes, checkpoint = _MODEL_CACHE[cache_key]
    model.to(device)
    model.eval()
    return model, classes, checkpoint


def analyze_scan(
    config_path: str | Path,
    checkpoint_path: str | Path,
    image_path: str | Path,
    generate_explanation: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return {
            "status": "model_missing",
            "prediction": "Training required",
            "confidence": 0.0,
            "probabilities": pd.DataFrame(
                [{"class": class_name, "probability": 0.0} for class_name in config["dataset"]["classes"]]
            ),
            "explanation_path": None,
        }

    device = get_device()
    model, classes, _ = load_cached_checkpoint(config_path, checkpoint_path, device)
    image_size = int(config["dataset"]["image_size"])
    image_tensor, original = preprocess_image(image_path, image_size)

    with torch.no_grad():
        probs = torch.softmax(model(image_tensor.to(device)), dim=1)[0].cpu()

    predicted_index = int(torch.argmax(probs).item())
    explanation_path = None
    lesion = None

    explanation_artifacts = None
    if generate_explanation:
        output_dir = ensure_dir(config["outputs"]["explanations"])
        explanation_path = Path(output_dir) / f"{Path(image_path).stem}_{classes[predicted_index]}_xai.png"
        xai = config.get("xai", {})
        try:
            heatmaps: dict[str, Any] = {}

            if xai.get("lime", {}).get("enabled", True):
                try:
                    lime_settings = xai.get("lime", {})
                    heatmaps["lime"] = lime_map(
                        model,
                        image_tensor,
                        predicted_index,
                        device,
                        grid_size=int(lime_settings.get("grid_size", 6)),
                        n_samples=int(lime_settings.get("n_samples", 150)),
                    )
                except Exception as exc:
                    print(f"LIME failed: {exc}")
            else:
                try:
                    heatmaps["occlusion"] = occlusion_map(
                        model=model,
                        image=image_tensor,
                        target_class=predicted_index,
                        patch_size=int(xai.get("occlusion_patch_size", config["xai"]["occlusion_patch_size"])),
                        stride=int(xai.get("occlusion_stride", config["xai"]["occlusion_stride"])),
                        device=device,
                    )
                except Exception as exc:
                    print(f"Occlusion failed: {exc}")

            if xai.get("grad_cam", {}).get("enabled", True):
                try:
                    heatmaps["gradcam"] = grad_cam_map(model, image_tensor, predicted_index, device)
                except Exception as exc:
                    print(f"Grad-CAM failed: {exc}")
            if xai.get("shap", {}).get("enabled", True):
                try:
                    shap_settings = xai.get("shap", {})
                    heatmaps["shap"] = shap_pixel_map(
                        model,
                        image_tensor,
                        original,
                        predicted_index,
                        device,
                        grid_size=int(shap_settings.get("grid_size", 8)),
                        n_samples=int(shap_settings.get("n_samples", 3)),
                    )
                except Exception as exc:
                    print(f"SHAP failed: {exc}")

            combined = build_combined_mask(heatmaps)
            if combined is not None and combined.any():
                explanation_artifacts = save_explanations(
                    original,
                    heatmaps,
                    explanation_path,
                    is_normal=(classes[predicted_index] == "normal"),
                )
            else:
                occlusion_heatmap = heatmaps.get("lime") or heatmaps.get("occlusion")
                if occlusion_heatmap is not None:
                    explanation_artifacts = save_overlay(original, occlusion_heatmap, explanation_path)

            if combined is not None and classes[predicted_index] != "normal":
                base_name = f"{Path(image_path).stem}_{classes[predicted_index]}"
                lesion = localize_lesion(original, combined, output_dir, base_name)
                if lesion is not None and explanation_artifacts:
                    explanation_artifacts["lesion_overlay"] = lesion["overlay_path"]
        except Exception as exc:
            print(f"Explanation pipeline failed: {exc}")
        explanation_path = (
            Path(explanation_artifacts["combined"]) if explanation_artifacts else None
        )

    predicted_label = classes[predicted_index]
    if predicted_label == "stroke":
        predicted_label = "stroke"

    return {
        "status": "ok",
        "prediction": classes[predicted_index],
        "confidence": float(probs[predicted_index].item()),
        "probabilities": pd.DataFrame(
            [
                {"class": class_name, "probability": round(float(probability), 6)}
                for class_name, probability in zip(classes, probs.tolist())
            ]
        ),
        "explanation_path": str(explanation_path) if explanation_path else None,
        "explanation_artifacts": explanation_artifacts,
        "lesion": lesion,
    }
