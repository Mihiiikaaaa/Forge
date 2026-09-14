"""CLI entry point for GA, NPO and RMU TOFU unlearning experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Allows both `python src/unlearning/run_unlearning.py` and module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.unlearning.config import UnlearningConfig, load_config_file, select_device
from src.unlearning.ga import GradientAscentObjective
from src.unlearning.npo import NPOObjective
from src.unlearning.rmu import RMUObjective
from src.unlearning.trainer import UnlearningTrainer, load_tofu_datasets, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run a TOFU unlearning baseline.")
    parser.add_argument("--method", choices=("ga", "npo", "rmu"), required=True)
    parser.add_argument("--config", help="Optional YAML file; CLI values take precedence.")
    for name, argument_type in (
        ("model_path", str), ("forget_path", str), ("retain_path", str), ("output_dir", str),
        ("seed", int), ("device", str), ("max_length", int), ("epochs", int),
        ("batch_size", int), ("gradient_accumulation_steps", int), ("learning_rate", float),
        ("weight_decay", float), ("logging_steps", int), ("beta", float), ("target_layer", int),
        ("retain_coefficient", float), ("target_coefficient", float), ("target_norm", float),
    ):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=argument_type, default=None)
    return parser.parse_args()


def build_config(args) -> UnlearningConfig:
    # Keep output_dir as None here so its method-specific default is resolved
    # only after the required --method value has been applied.
    values = asdict(UnlearningConfig())
    values.update(load_config_file(args.config))
    values["method"] = args.method
    values.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
    return UnlearningConfig(**values)


def ensure_safe_output(config: UnlearningConfig) -> None:
    baseline = Path(config.model_path).resolve()
    output = config.resolved_output_dir().resolve()
    if output == baseline or baseline in output.parents:
        raise ValueError("output_dir cannot be the baseline directory or a path inside it; the baseline is read-only.")


def make_objective(config, model, device):
    if config.method == "ga":
        return GradientAscentObjective()
    if config.method == "npo":
        return NPOObjective(model, config.beta).to(device)
    return RMUObjective(
        model, config.target_layer, config.retain_coefficient, config.target_coefficient,
        config.target_norm, config.seed,
    ).to(device)


def main():
    args = parse_args()
    config = build_config(args)
    ensure_safe_output(config)
    if min(config.epochs, config.batch_size, config.gradient_accumulation_steps, config.logging_steps) < 1:
        raise ValueError("epochs, batch_size, gradient_accumulation_steps, and logging_steps must be positive.")
    if config.beta <= 0 or config.retain_coefficient < 0 or config.target_coefficient <= 0 or config.target_norm <= 0:
        raise ValueError("beta, target_coefficient, and target_norm must be positive; retain_coefficient cannot be negative.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(config.seed)
    device = select_device(config.device)
    print(f"Method: {config.method}\nDevice: {device}")
    if not Path(config.model_path).is_dir():
        raise FileNotFoundError(f"Baseline model directory not found: {config.model_path}")
    forget_dataset, retain_dataset = load_tofu_datasets(config)
    print(f"Forget dataset size: {len(forget_dataset)}\nRetain dataset size: {len(retain_dataset)}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    objective = make_objective(config, model, device)
    trainer = UnlearningTrainer(model, tokenizer, objective, config, device)
    history = trainer.train(forget_dataset, retain_dataset)
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "unlearning_config.json", "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    final_loss = history[-1]["training_loss"] if history else float("nan")
    print(f"Training loss: {final_loss:.6f}\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
