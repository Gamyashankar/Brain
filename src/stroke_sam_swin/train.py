from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm

from .data import build_loaders
from .model import build_model, save_checkpoint
from .utils import ensure_dir, get_device, load_config, save_json, set_seed


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    mixed_precision: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    scaler = torch.cuda.amp.GradScaler(enabled=training and mixed_precision and device.type == "cuda")

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def train(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    device = get_device()

    train_loader, val_loader, _, classes = build_loaders(config)
    model = build_model(config, num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    checkpoint_dir = ensure_dir(config["outputs"]["checkpoints"])
    report_dir = ensure_dir(config["outputs"]["reports"])
    best_f1 = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        print(f"Epoch {epoch}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            mixed_precision=bool(config["training"].get("mixed_precision", True)),
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(row)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                str(Path(checkpoint_dir) / "best_model.pt"),
                model,
                classes,
                config,
                {"best_epoch": epoch, **val_metrics},
            )

    save_json({"classes": classes, "history": history}, Path(report_dir) / "training_history.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
