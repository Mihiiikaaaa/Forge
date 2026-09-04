import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from answer_likelihood import answer_nll


MODEL_PATH = "models/tofu_learned"

FORGET_PATH = "data/processed/forget_1"
RETAIN_PATH = "data/processed/retain_1"

OUTPUT_PATH = "results/raw/baseline/pre_unlearning_likelihood.json"


def load_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    print("Using device:", device)

    return tokenizer, model, device


def evaluate_dataset(model, tokenizer, device, dataset, name):

    results = []

    print(f"\nEvaluating {name}")
    print("Examples:", len(dataset))

    for i in range(len(dataset)):

        question = dataset[i]["question"]
        answer = dataset[i]["answer"]

        nll = answer_nll(
            model,
            tokenizer,
            device,
            question,
            answer
        )

        results.append({
            "id": i,
            "question": question,
            "answer": answer,
            "answer_nll": nll
        })

        print(
            f"[{i + 1}/{len(dataset)}] "
            f"NLL = {nll:.4f}"
        )

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
    print("LIKELIHOOD EVALUATION COMPLETE")
    print("===================================")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()