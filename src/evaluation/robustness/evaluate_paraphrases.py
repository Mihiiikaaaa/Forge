import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.evaluation.answer_likelihood import (
    calculate_answer_likelihood,
    get_device
)


def load_model(model_path):

    device = get_device()

    print("Device:", device)
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return model, tokenizer, device


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True
    )

    parser.add_argument(
        "--paraphrase-path",
        default="data/processed/forget_paraphrases.json"
    )

    parser.add_argument(
        "--output",
        default="results/raw/robustness/paraphrase_results.json"
    )

    args = parser.parse_args()

    model, tokenizer, device = load_model(
        args.model_path
    )

    with open(
        args.paraphrase_path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    results = []

    for item in data:

        target_result = {
            "id": item["id"],
            "original_question": item["original_question"],
            "paraphrases": []
        }

        for paraphrase in item["paraphrases"]:

            text = paraphrase["text"].strip()

            if not text:
                continue

            if not paraphrase["semantically_equivalent"]:
                continue

            result = calculate_answer_likelihood(
                model=model,
                tokenizer=tokenizer,
                question=text,
                answer=item["expected_answer"],
                device=device
            )

            target_result["paraphrases"].append(
                {
                    "question": text,
                    "likelihood": result["likelihood"],
                    "normalized_log_likelihood":
                        result["normalized_log_likelihood"]
                }
            )

        results.append(target_result)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n===================================")
    print("PARAPHRASE EVALUATION COMPLETE")
    print("===================================")
    print("Targets:", len(results))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()