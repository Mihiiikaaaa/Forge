"""A lightweight representation-misdirection unlearning objective."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as functional

from .base import UnlearningObjective


def final_token_representation(hidden_states, attention_mask, target_layer: int):
    """Select the final non-padding token representation at a transformer layer."""
    layers = len(hidden_states) - 1  # hidden_states[0] is embedding output
    layer = target_layer if target_layer >= 0 else layers + target_layer
    if not 0 <= layer < layers:
        raise ValueError(f"target_layer must be in [-{layers}, {layers - 1}], got {target_layer}")
    states = hidden_states[layer + 1]
    last_indices = attention_mask.sum(dim=1).to(torch.long).sub(1)
    return states[torch.arange(states.shape[0], device=states.device), last_indices]


class RMUObjective(UnlearningObjective):
    """Misdirect forget representations while anchoring retain representations.

    This is a lightweight RMU-style approximation: it uses a fixed seeded random
    vector at one hidden layer and a frozen baseline to preserve retain examples.
    It intentionally does not claim equivalence to a particular published RMU
    training recipe.
    """

    def __init__(self, reference_model, target_layer, retain_coefficient, target_coefficient, target_norm, seed):
        self.reference_model = copy.deepcopy(reference_model).eval()
        self.target_layer = target_layer
        self.retain_coefficient = retain_coefficient
        self.target_coefficient = target_coefficient
        hidden_size = reference_model.config.hidden_size
        generator = torch.Generator(device="cpu").manual_seed(seed)
        target = torch.randn(hidden_size, generator=generator)
        self.target = target / target.norm() * target_norm
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    def to(self, device: torch.device) -> "RMUObjective":
        self.reference_model.to(device)
        self.target = self.target.to(device)
        return self

    def loss(self, model, forget_batch, retain_batch=None):
        if retain_batch is None:
            raise ValueError("RMU requires retain batches.")
        forget_outputs = model(**forget_batch, output_hidden_states=True, use_cache=False)
        retain_outputs = model(**retain_batch, output_hidden_states=True, use_cache=False)
        forget_repr = final_token_representation(
            forget_outputs.hidden_states, forget_batch["attention_mask"], self.target_layer
        )
        retain_repr = final_token_representation(
            retain_outputs.hidden_states, retain_batch["attention_mask"], self.target_layer
        )
        with torch.no_grad():
            reference_outputs = self.reference_model(
                **retain_batch, output_hidden_states=True, use_cache=False
            )
            reference_repr = final_token_representation(
                reference_outputs.hidden_states, retain_batch["attention_mask"], self.target_layer
            )
        target_loss = functional.mse_loss(forget_repr, self.target.expand_as(forget_repr))
        retain_loss = functional.mse_loss(retain_repr, reference_repr)
        total = self.target_coefficient * target_loss + self.retain_coefficient * retain_loss
        return total, {
            "rmu_target_loss": float(target_loss.detach()),
            "rmu_retain_loss": float(retain_loss.detach()),
        }
