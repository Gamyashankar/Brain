from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import build_loaders
from .model import build_model, save_checkpoint
from .utils import ensure_dir, get_device, load_config, load_json, save_json, set_seed


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


def train(config_path: str, resume: bool = False, epochs_target: int | None = None) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    device = get_device()

    total_epochs = int(config["training"]["epochs"])
    if epochs_target is None:
        epochs_target = total_epochs

    train_loader, val_loader, _, classes = build_loaders(config)
    model = build_model(config, num_classes=len(classes)).to(device)

    targets = np.asarray(train_loader.dataset.targets)
    counts = np.bincount(targets, minlength=len(classes)).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32, device=device)
    print(f"Class weights: {dict(zip(classes, class_weights.cpu().tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    checkpoint_dir = ensure_dir(config["outputs"]["checkpoints"])
    report_dir = ensure_dir(config["outputs"]["reports"])
    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
    start_epoch = 1
    best_f1 = -1.0

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if list(checkpoint.get("classes", [])) != classes:
            raise RuntimeError(
                f"Cannot resume: checkpoint trained on {checkpoint.get('classes')} "
                f"but the dataset now has {classes}. Retrain from scratch first "
                "(run without --resume)."
            )
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_f1 = float(checkpoint["metrics"].get("best_f1", -1.0))
        start_epoch = int(checkpoint["metrics"].get("best_epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch} (target {epochs_target}), best val F1 = {best_f1:.4f}")

    history: list[dict[str, float | int]] = []
    if resume and (Path(report_dir) / "training_history.json").exists():
        history = load_json(Path(report_dir) / "training_history.json").get("history", [])

    for epoch in range(start_epoch, epochs_target + 1):
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
        scheduler.step()

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
                {"best_epoch": epoch, "best_f1": best_f1, **val_metrics},
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
            )

        save_checkpoint(
            str(Path(checkpoint_dir) / "latest_model.pt"),
            model,
            classes,
            config,
            {"best_epoch": epoch, "best_f1": best_f1, **val_metrics},
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
        )

    save_json({"classes": classes, "history": history}, Path(report_dir) / "training_history.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Run training up to this cumulative epoch (default: config epochs)")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing best_model.pt checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, resume=args.resume, epochs_target=args.epochs)
