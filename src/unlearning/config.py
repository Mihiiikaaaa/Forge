"""Configuration and device helpers for unlearning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class UnlearningConfig:
    method: str = "ga"
    model_path: str = "models/tofu_baseline"
    forget_path: str = "data/processed/forget_1"
    retain_path: str = "data/processed/retain_1"
    output_dir: str | None = None
    seed: int = 42
    device: str = "auto"
    max_length: int = 512
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    logging_steps: int = 10
    beta: float = 0.1
    target_layer: int = -1
    retain_coefficient: float = 1.0
    target_coefficient: float = 1.0
    target_norm: float = 1.0

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir or f"models/unlearned/{self.method}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.resolved_output_dir())
        return data


def load_config_file(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Unlearning config file must contain a mapping.")
    return data


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
