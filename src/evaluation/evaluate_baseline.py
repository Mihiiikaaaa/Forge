from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import json
from pathlib import Path
import torch


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

FORGET_PATH = "data/processed/forget_1"
RETAIN_PATH = "data/processed/retain_1"

OUTPUT_PATH = "results/raw/baseline/baseline_results.json"

NUM_EXAMPLES = 10


def generate_response(model, tokenizer, question):
    """
    Generate a deterministic response for a question.
    """

    inputs = tokenizer(
        question,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


def evaluate_dataset(model, tokenizer, dataset, dataset_name):
    """
    Evaluate a small sample from a dataset.
    """

    results = []

    limit = min(NUM_EXAMPLES, len(dataset))

    for i in range(limit):

        question = dataset[i]["question"]
        expected_answer = dataset[i]["answer"]

        response = generate_response(
            model,
            tokenizer,
            question
        )

        results.append({
            "index": i,
            "question": question,
            "expected_answer": expected_answer,
            "model_response": response
        })

        print(f"\n[{dataset_name}] Example {i + 1}/{limit}")
        print("Question:", question)
        print("Expected:", expected_answer)
        print("Model:", response)

    return results


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    print("Loading forget set...")

    forget_dataset = load_from_disk(
        FORGET_PATH
    )

    print("Loading retain set...")

    retain_dataset = load_from_disk(
        RETAIN_PATH
    )

    print("\nEvaluating forget set...")

    forget_results = evaluate_dataset(
        model,
        tokenizer,
        forget_dataset,
        "FORGET"
    )

    print("\nEvaluating retain set...")

    retain_results = evaluate_dataset(
        model,
        tokenizer,
        retain_dataset,
        "RETAIN"
    )

    results = {
        "model": MODEL_NAME,
        "num_examples_per_split": NUM_EXAMPLES,
        "forget": forget_results,
        "retain": retain_results
    }

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4
        )

    print("\n===================================")
    print("BASELINE EVALUATION COMPLETE")
    print("===================================")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()