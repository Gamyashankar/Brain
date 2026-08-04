from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_dataset(root: str | Path, split: str, image_size: int, train: bool) -> datasets.ImageFolder:
    split_root = Path(root) / split
    if not split_root.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_root}")
    return datasets.ImageFolder(split_root, transform=build_transforms(image_size, train=train))


def build_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader | None, list[str]]:
    dataset_cfg = config["dataset"]
    root = dataset_cfg["root"]
    image_size = int(dataset_cfg["image_size"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(dataset_cfg.get("num_workers", 0))

    train_ds = build_dataset(root, "train", image_size, train=True)
    val_ds = build_dataset(root, "val", image_size, train=False)

    test_root = Path(root) / "test"
    test_ds = build_dataset(root, "test", image_size, train=False) if test_root.exists() else None

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, train_ds.classes
