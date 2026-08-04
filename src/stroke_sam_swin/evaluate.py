from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .data import build_loaders
from .model import load_checkpoint
from .utils import ensure_dir, get_device, load_config, save_json


def evaluate(config_path: str, checkpoint_path: str) -> None:
    config = load_config(config_path)
    device = get_device()
    _, val_loader, test_loader, _ = build_loaders(config)
    loader = test_loader or val_loader

    model, classes, _ = load_checkpoint(checkpoint_path, config, device)
    all_probs: list[list[float]] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(probs.argmax(dim=1).cpu().tolist())
            all_targets.extend(labels.tolist())

    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(all_targets, all_preds).tolist()
    payload = {
        "classes": classes,
        "classification_report": report,
        "confusion_matrix": matrix,
        "probabilities": np.asarray(all_probs).tolist(),
    }

    report_dir = ensure_dir(config["outputs"]["reports"])
    save_json(payload, Path(report_dir) / "evaluation.json")
    print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))
    print("Confusion matrix:")
    print(np.asarray(matrix))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.checkpoint)
