from __future__ import annotations

import timm
import torch
from torch import nn


def build_model(config: dict, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    model_cfg = config["model"]
    use_pretrained = bool(model_cfg.get("pretrained", True)) if pretrained is None else pretrained
    return timm.create_model(
        model_cfg["name"],
        pretrained=use_pretrained,
        num_classes=num_classes,
    )


def save_checkpoint(path: str, model: nn.Module, classes: list[str], config: dict, metrics: dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, config: dict, device: torch.device) -> tuple[nn.Module, list[str], dict]:
    checkpoint = torch.load(path, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(config, num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, classes, checkpoint
