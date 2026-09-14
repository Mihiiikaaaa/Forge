"""Reusable data preparation and optimisation loop for all baselines."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from .config import UnlearningConfig

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AnswerOnlyCollator:
    """Tokenise TOFU records and mask prompt tokens from the LM objective."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skipped_examples = 0
        self.padding_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, padding=True, return_tensors="pt"
        )

    def __call__(self, examples):
        features = []
        for example in examples:
            prompt = f"Question: {example['question']}\nAnswer:"
            text = f"{prompt} {example['answer']}{self.tokenizer.eos_token}"
            encoded = self.tokenizer(text, truncation=True, max_length=self.max_length)
            prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
            labels = list(encoded["input_ids"])
            for index in range(min(len(prompt_ids), len(labels))):
                labels[index] = -100
            # Causal LM loss predicts labels after a one-token shift. A prompt
            # that consumes max_length therefore has no usable answer target.
            if not any(label != -100 for label in labels[1:]):
                self.skipped_examples += 1
                continue
            encoded["labels"] = labels
            features.append(encoded)
        if not features:
            return None
        return self.padding_collator(features)


def load_tofu_datasets(config: UnlearningConfig):
    forget_dataset = load_from_disk(config.forget_path)
    retain_dataset = load_from_disk(config.retain_path)
    required = {"question", "answer"}
    for name, dataset in (("forget", forget_dataset), ("retain", retain_dataset)):
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(f"{name} dataset is missing columns: {sorted(missing)}")
        if not len(dataset):
            raise ValueError(f"{name} dataset is empty")
    return forget_dataset, retain_dataset


class UnlearningTrainer:
    """Optimise an ``UnlearningObjective`` without duplicating method loops."""

    def __init__(self, model, tokenizer, objective, config: UnlearningConfig, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.objective = objective
        self.config = config
        self.device = device

    @staticmethod
    def _optimizer_step(optimizer, model, accumulated_steps: int) -> None:
        """Average accumulated microbatch gradients before one optimiser step."""
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_steps)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def train(self, forget_dataset, retain_dataset) -> list[dict[str, float]]:
        collator = AnswerOnlyCollator(self.tokenizer, self.config.max_length)
        loader_args = dict(
            batch_size=self.config.batch_size, collate_fn=collator, num_workers=0,
        )
        forget_loader = DataLoader(forget_dataset, shuffle=True, **loader_args)
        retain_loader = DataLoader(retain_dataset, shuffle=True, **loader_args)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        history = []
        global_step = 0
        retain_iterator = iter(retain_loader)
        skipped_forget_batches = 0

        def next_retain_batch():
            """Return a valid retain batch, restarting after one loader pass."""
            nonlocal retain_iterator
            for _ in range(len(retain_loader)):
                try:
                    batch = next(retain_iterator)
                except StopIteration:
                    retain_iterator = iter(retain_loader)
                    batch = next(retain_iterator)
                if batch is not None:
                    return batch
            raise ValueError(
                "All retain examples were skipped because truncation removed their answer tokens. "
                "Increase max_length or inspect the retain dataset."
            )

        for epoch in range(self.config.epochs):
            accumulated_steps = 0
            for forget_batch in forget_loader:
                if forget_batch is None:
                    skipped_forget_batches += 1
                    continue
                retain_batch = next_retain_batch()
                forget_batch = {key: value.to(self.device) for key, value in forget_batch.items()}
                retain_batch = {key: value.to(self.device) for key, value in retain_batch.items()}
                loss, metrics = self.objective.loss(self.model, forget_batch, retain_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss.item()}")
                loss.backward()
                global_step += 1
                accumulated_steps += 1
                if accumulated_steps == self.config.gradient_accumulation_steps:
                    self._optimizer_step(optimizer, self.model, accumulated_steps)
                    accumulated_steps = 0
                record = {"step": global_step, "epoch": epoch + 1, "training_loss": float(loss.detach()), **metrics}
                history.append(record)
                if global_step % self.config.logging_steps == 0:
                    LOGGER.info("epoch=%d step=%d training_loss=%.6f %s", epoch + 1, global_step, record["training_loss"], metrics)
            # Apply a final partial accumulation at each epoch boundary.
            if accumulated_steps:
                self._optimizer_step(optimizer, self.model, accumulated_steps)
        if collator.skipped_examples:
            LOGGER.warning(
                "Skipped %d example(s) whose answers were fully truncated; increase max_length to include them.",
                collator.skipped_examples,
            )
        if skipped_forget_batches:
            LOGGER.warning("Skipped %d all-invalid forget batch(es).", skipped_forget_batches)
        if not history:
            raise ValueError(
                "No valid forget examples remained after truncation. Increase max_length or inspect the forget dataset."
            )
        return history
