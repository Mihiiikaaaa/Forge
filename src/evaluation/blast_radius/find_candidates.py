import json
from pathlib import Path

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


FORGET_PATH = "data/processed/forget_1"
RETAIN_PATH = "data/processed/retain_1"

OUTPUT_PATH = (
    "data/processed/blast_radius_candidates.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 20


def main():

    print("Loading datasets...")

    forget_dataset = load_from_disk(
        FORGET_PATH
    )

    retain_dataset = load_from_disk(
        RETAIN_PATH
    )

    print(
        "Forget examples:",
        len(forget_dataset)
    )

    print(
        "Retain examples:",
        len(retain_dataset)
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    retain_questions = [
        item["question"]
        for item in retain_dataset
    ]

    print("\nEncoding retain questions...")

    retain_embeddings = model.encode(
        retain_questions,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    results = []

    print("\nMining candidates...")

    for target_id in range(
        len(forget_dataset)
    ):

        target_question = (
            forget_dataset[target_id]["question"]
        )

        target_embedding = model.encode(
            [target_question],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            target_embedding,
            retain_embeddings
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        candidates = []

        for index in ranked_indices[:TOP_K]:

            candidates.append(
                {
                    "retain_id": int(index),
                    "question": retain_questions[index],
                    "similarity": float(
                        similarities[index]
                    ),
                    "category": "unreviewed"
                }
            )

        results.append(
            {
                "forget_id": target_id,
                "target_question": target_question,
                "candidates": candidates
            }
        )

        if (target_id + 1) % 5 == 0:
            print(
                f"Processed "
                f"{target_id + 1}/"
                f"{len(forget_dataset)}"
            )

    output_path = Path(OUTPUT_PATH)

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
    print("BLAST RADIUS CANDIDATES CREATED")
    print("===================================")

    print(
        "Targets:",
        len(results)
    )

    print(
        "Candidates per target:",
        TOP_K
    )

    print(
        "Total candidate relationships:",
        len(results) * TOP_K
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()