import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "models/tofu_learned"

FORGET_PATH = "data/processed/forget_1"
RETAIN_PATH = "data/processed/retain_1"

OUTPUT_PATH = "results/raw/baseline/pre_unlearning_results.json"

MAX_LENGTH = 512
MAX_NEW_TOKENS = 80


def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print("Using device:", device)

    return tokenizer, model, device


def generate_answer(model, tokenizer, device, question):
    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


def evaluate_dataset(model, tokenizer, device, dataset, name, limit=40):
    results = []

    limit = min(limit, len(dataset))

    print(f"\nEvaluating {name}: {limit} examples")

    for i in range(limit):
        question = dataset[i]["question"]
        expected = dataset[i]["answer"]

        generated = generate_answer(
            model,
            tokenizer,
            device,
            question
        )

        results.append({
            "id": i,
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated
        })

        print(f"\n[{i + 1}/{limit}]")
        print("Question:", question)
        print("Expected:", expected)
        print("Generated:", generated)

    return results


def main():
    tokenizer, model, device = load_model()

    print("\nLoading datasets...")

    forget_dataset = load_from_disk(FORGET_PATH)
    retain_dataset = load_from_disk(RETAIN_PATH)

    forget_results = evaluate_dataset(
        model,
        tokenizer,
        device,
        forget_dataset,
        "FORGET SET"
    )

    retain_results = evaluate_dataset(
        model,
        tokenizer,
        device,
        retain_dataset,
        "RETAIN SET"
    )

    results = {
        "model": MODEL_PATH,
        "forget_set_size": len(forget_dataset),
        "retain_set_size": len(retain_dataset),
        "forget_evaluation": forget_results,
        "retain_evaluation": retain_results
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===================================")
    print("PRE-UNLEARNING EVALUATION COMPLETE")
    print("===================================")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()