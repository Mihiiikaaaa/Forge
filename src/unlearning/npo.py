"""Negative Preference Optimisation (NPO) objective for causal language models."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as functional

from .base import UnlearningObjective, sequence_log_prob


class NPOObjective(UnlearningObjective):
    """Move forget-answer likelihood below a frozen baseline reference.

    The loss is ``softplus(beta * (log p_model - log p_reference))``.  Its
    minimisation penalises a model that still assigns the forget answer at least
    as much probability as the original baseline.
    """

    def __init__(self, reference_model: torch.nn.Module, beta: float):
        self.reference_model = copy.deepcopy(reference_model).eval()
        self.beta = beta
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    def to(self, device: torch.device) -> "NPOObjective":
        self.reference_model.to(device)
        return self

    def loss(self, model, forget_batch, retain_batch=None):
        current = model(**forget_batch, use_cache=False)
        with torch.no_grad():
            reference = self.reference_model(**forget_batch, use_cache=False)
        current_log_prob = sequence_log_prob(current.logits, forget_batch["labels"])
        reference_log_prob = sequence_log_prob(reference.logits, forget_batch["labels"])
        delta = current_log_prob - reference_log_prob
        loss = functional.softplus(self.beta * delta).mean()
        return loss, {
            "npo_loss": float(loss.detach()),
            "log_prob_delta": float(delta.detach().mean()),
        }
