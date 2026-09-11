import json
from pathlib import Path
from datasets import load_from_disk


FORGET_PATH = "data/processed/forget_1"
OUTPUT_PATH = "data/processed/forget_paraphrases.json"

NUM_PARAPHRASES = 4


def create_template(example_id, question, answer):
    return {
        "id": example_id,
        "original_question": question,
        "expected_answer": answer,
        "paraphrases": [
            {
                "text": "",
                "human_verified": False,
                "semantically_equivalent": False
            }
            for _ in range(NUM_PARAPHRASES)
        ]
    }


def main():
    print("Loading forget dataset...")

    dataset = load_from_disk(FORGET_PATH)

    print("Forget examples:", len(dataset))

    results = []

    for i in range(len(dataset)):
        question = dataset[i]["question"]
        answer = dataset[i]["answer"]

        results.append(
            create_template(
                example_id=i,
                question=question,
                answer=answer
            )
        )

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n===================================")
    print("PARAPHRASE DATASET TEMPLATE CREATED")
    print("===================================")
    print("Targets:", len(results))
    print("Paraphrases per target:", NUM_PARAPHRASES)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()