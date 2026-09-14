"""Shared objective interface and causal-LM likelihood utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


def sequence_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return masked, answer-token log likelihood for every batch item."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_log_probs = torch.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * valid).sum(dim=-1)


class UnlearningObjective(ABC):
    """An objective used by :class:`UnlearningTrainer`.

    The trainer owns optimisation; objectives only calculate differentiable losses.
    """

    @abstractmethod
    def loss(
        self,
        model: torch.nn.Module,
        forget_batch: dict[str, torch.Tensor],
        retain_batch: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return a scalar loss and scalar values for logging."""
