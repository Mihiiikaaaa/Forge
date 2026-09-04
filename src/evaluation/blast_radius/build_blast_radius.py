import json
from pathlib import Path

from datasets import load_from_disk


FORGET_PATH = "data/processed/forget_1"
OUTPUT_PATH = "data/processed/blast_radius.json"


def create_template(example_id, question, answer):

    return {
        "forget_id": example_id,

        "target": {
            "question": question,
            "answer": answer
        },

        "related": [],

        "moderately_related": [],

        "unrelated": []
    }


def main():

    print("Loading forget dataset...")

    dataset = load_from_disk(FORGET_PATH)

    blast_radius_data = []

    for i in range(len(dataset)):

        question = dataset[i]["question"]
        answer = dataset[i]["answer"]

        item = create_template(
            i,
            question,
            answer
        )

        blast_radius_data.append(item)

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        json.dump(
            blast_radius_data,
            f,
            indent=4
        )

    print("\n===================================")
    print("BLAST RADIUS TEMPLATE CREATED")
    print("===================================")
    print("Targets:", len(blast_radius_data))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()