import json
from pathlib import Path

from datasets import load_from_disk


FORGET_PATH = "data/processed/forget_1"
OUTPUT_PATH = "data/processed/forget_paraphrases.json"


def create_paraphrase_template(question):
    """
    Create paraphrase slots for a question.

    The actual paraphrases will be written/generated separately.
    This keeps the evaluation dataset explicit and auditable.
    """

    return {
        "original": question,
        "paraphrases": []
    }


def main():

    print("Loading forget dataset...")

    dataset = load_from_disk(FORGET_PATH)

    results = []

    print("Number of forget examples:", len(dataset))

    for i in range(len(dataset)):

        question = dataset[i]["question"]
        answer = dataset[i]["answer"]

        item = {
            "id": i,
            "original_question": question,
            "expected_answer": answer,
            "paraphrases": []
        }

        results.append(item)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===================================")
    print("PARAPHRASE DATASET CREATED")
    print("===================================")
    print("Examples:", len(results))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()