"""Gradient-ascent unlearning objective."""

from __future__ import annotations

import torch

from .base import UnlearningObjective


class GradientAscentObjective(UnlearningObjective):
    """Maximise answer-token cross-entropy on the forget examples."""

    def loss(self, model, forget_batch, retain_batch=None):
        outputs = model(**forget_batch, use_cache=False)
        forget_loss = outputs.loss
        # Optimisers minimise; negating loss implements gradient ascent.
        return -forget_loss, {"forget_loss": float(forget_loss.detach())}
