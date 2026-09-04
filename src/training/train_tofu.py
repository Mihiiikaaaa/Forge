import os
import random
import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TRAIN_PATH = "data/processed/retain_1"
OUTPUT_DIR = "models/tofu_baseline"

SEED = 42
MAX_LENGTH = 512


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_example(example):
    text = (
        "Question: "
        + example["question"]
        + "\nAnswer: "
        + example["answer"]
    )

    return {"text": text}


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def main():

    set_seed(SEED)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    print("Loading TOFU training data...")

    dataset = load_from_disk(TRAIN_PATH)

    print("Number of training examples:", len(dataset))

    dataset = dataset.map(
        format_example
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=2,

        gradient_accumulation_steps=8,

        learning_rate=2e-5,

        logging_steps=10,

        save_strategy="epoch",

        report_to="none",

        seed=SEED,

        fp16=torch.cuda.is_available(),

        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting TOFU training...")

    trainer.train()

    print("\nSaving trained model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n===================================")
    print("TOFU BASELINE TRAINING COMPLETE")
    print("===================================")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()