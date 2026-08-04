from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import ensure_dir, load_config


def load_sam(config: dict):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError("Install segment-anything before running SAM preprocessing.") from exc

    sam_cfg = config["sam"]
    checkpoint = Path(sam_cfg["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    sam = sam_model_registry[sam_cfg["model_type"]](checkpoint=str(checkpoint))
    return SamAutomaticMaskGenerator(sam)


def choose_mask(masks: list[dict], image_shape: tuple[int, int], min_ratio: float, max_ratio: float) -> np.ndarray | None:
    height, width = image_shape
    image_area = height * width
    center = np.array([width / 2.0, height / 2.0])
    candidates = []

    for mask in masks:
        segmentation = mask["segmentation"].astype(bool)
        area_ratio = segmentation.sum() / image_area
        if area_ratio < min_ratio or area_ratio > max_ratio:
            continue

        ys, xs = np.where(segmentation)
        if len(xs) == 0:
            continue
        centroid = np.array([xs.mean(), ys.mean()])
        centrality = np.linalg.norm((centroid - center) / np.array([width, height]))
        score = area_ratio - 0.35 * centrality
        candidates.append((score, segmentation))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image

    height, width = image.shape[:2]
    x1 = max(0, xs.min() - padding)
    y1 = max(0, ys.min() - padding)
    x2 = min(width, xs.max() + padding)
    y2 = min(height, ys.max() + padding)
    return image[y1:y2, x1:x2]


def preprocess_with_sam(config_path: str) -> None:
    config = load_config(config_path)
    dataset_root = Path(config["dataset"]["root"])
    output_root = Path("data/processed/sam_roi")
    mask_generator = load_sam(config)
    sam_cfg = config["sam"]

    image_paths = [
        path
        for split in ("train", "val", "test")
        for path in (dataset_root / split).glob("*/*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]

    if not image_paths:
        raise FileNotFoundError(f"No images found under {dataset_root}")

    for image_path in tqdm(image_paths):
        relative = image_path.relative_to(dataset_root)
        target_image = output_root / relative
        target_mask = output_root / "masks" / relative.with_suffix(".png")
        ensure_dir(target_image.parent)
        ensure_dir(target_mask.parent)

        image = np.asarray(Image.open(image_path).convert("RGB"))
        masks = mask_generator.generate(image)
        selected = choose_mask(
            masks,
            image_shape=image.shape[:2],
            min_ratio=float(sam_cfg["min_mask_area_ratio"]),
            max_ratio=float(sam_cfg["max_mask_area_ratio"]),
        )

        if selected is None:
            cropped = image
            selected = np.zeros(image.shape[:2], dtype=bool)
        else:
            cropped = crop_to_mask(image, selected, padding=int(sam_cfg["crop_padding"]))

        Image.fromarray(cropped).save(target_image)
        cv2.imwrite(str(target_mask), np.uint8(selected) * 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_with_sam(args.config)
