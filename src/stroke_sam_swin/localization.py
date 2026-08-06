from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import ensure_dir

# Approximate in-plane pixel spacing used to convert pixel area to mm^2.
# Real CT slice spacing is typically ~0.45-0.6 mm/pixel at native resolution;
# we assume a flat 0.5 mm/pixel since the model operates on resized images.
DEFAULT_PIXEL_SPACING_MM = 0.5


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest 8-connected component of a binary mask."""
    if mask is None or mask.size == 0:
        return mask
    mask_u8 = np.uint8(mask > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def brain_mask(original: np.ndarray) -> np.ndarray:
    """Rough brain-tissue mask (bright CT tissue vs dark background)."""
    gray = cv2.cvtColor(np.uint8(original), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _largest_component(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def clean_lesion_mask(mask: np.ndarray) -> np.ndarray:
    """Denoise a binary mask and keep the largest connected region as the lesion."""
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=bool)
    mask_u8 = np.uint8(mask > 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return _largest_component(mask_u8) > 0


def compute_lesion_stats(
    lesion: np.ndarray,
    brain: np.ndarray,
    pixel_mm: float = DEFAULT_PIXEL_SPACING_MM,
) -> dict[str, float]:
    """Lesion area (mm^2), % of brain affected, and pixel centroid."""
    lesion = np.asarray(lesion, dtype=bool)
    if not lesion.any():
        return {"area_mm2": 0.0, "pct_brain": 0.0, "centroid_y": 0.0, "centroid_x": 0.0}
    area_px = int(lesion.sum())
    area_mm2 = float(area_px * pixel_mm * pixel_mm)
    brain_px = max(int(np.asarray(brain, dtype=bool).sum()), 1)
    pct_brain = float(area_px / brain_px * 100.0)
    ys, xs = np.nonzero(lesion)
    return {
        "area_mm2": round(area_mm2, 1),
        "pct_brain": round(pct_brain, 2),
        "centroid_y": round(float(np.mean(ys)), 1),
        "centroid_x": round(float(np.mean(xs)), 1),
    }


def map_brain_region(stats: dict[str, float], brain: np.ndarray) -> str:
    """Approximate anatomical region from the lesion centroid.

    Heuristic axial-slice atlas: the centroid is normalized inside the brain
    bounding box and mapped to a lobe using anterior (top) / posterior (bottom)
    and medial / lateral position. This is a simplified demonstration atlas --
    not a full MNI / Harvard-Oxford registration.
    """
    if not stats or not stats.get("area_mm2"):
        return "Not localized"
    cx = stats["centroid_x"]
    cy = stats["centroid_y"]
    ys, xs = np.nonzero(np.asarray(brain, dtype=bool))
    if len(ys) == 0:
        return "Not localized"
    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())
    h = max(y_max - y_min, 1.0)
    w = max(x_max - x_min, 1.0)
    ny = (cy - y_min) / h  # 0 = anterior/top, 1 = posterior/bottom
    nx = (cx - x_min) / w  # 0 = image-left, 1 = image-right

    if nx < 0.5:
        side = "Left"
    elif nx > 0.5:
        side = "Right"
    else:
        side = "Midline"

    if ny < 0.28:
        lobe = "Frontal Lobe"
    elif ny < 0.60:
        if nx < 0.24 or nx > 0.76:
            lobe = "Temporal Lobe"
        elif ny < 0.42:
            lobe = "Parietal Lobe"
        else:
            lobe = "Basal Ganglia / Deep White Matter"
    elif ny < 0.78:
        lobe = "Occipital Lobe"
    elif nx < 0.28 or nx > 0.72:
        lobe = "Cerebellum"
    else:
        lobe = "Brainstem"

    return f"{side} {lobe}" if side != "Midline" else lobe


def draw_lesion_overlay(original: np.ndarray, lesion: np.ndarray, output_path: str | Path) -> str:
    """Draw a red contour + semi-transparent red fill over the lesion."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = np.uint8(original).copy()
    mask_u8 = np.uint8(lesion > 0)

    red = np.zeros_like(base)
    red[mask_u8 > 0] = (255, 0, 0)
    blended = cv2.addWeighted(base, 0.7, red, 0.3, 0)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 0, 0), 2)
    Image.fromarray(blended).save(output_path)
    return str(output_path)


def localize_lesion(
    original: np.ndarray,
    combined_mask: np.ndarray,
    output_dir: str | Path,
    base_name: str,
) -> dict[str, object] | None:
    """Build the lesion localization payload from the XAI consensus mask."""
    lesion = clean_lesion_mask(combined_mask)
    if not lesion.any():
        return None
    brain = brain_mask(original)
    stats = compute_lesion_stats(lesion, brain)
    region = map_brain_region(stats, brain)
    overlay_path = draw_lesion_overlay(original, lesion, ensure_dir(output_dir) / f"{base_name}_lesion.png")
    return {
        "detected": True,
        "region": region,
        "area_mm2": stats["area_mm2"],
        "pct_brain": stats["pct_brain"],
        "centroid": f"({stats['centroid_x']:.0f}, {stats['centroid_y']:.0f})",
        "overlay_path": overlay_path,
    }
