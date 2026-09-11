import json
from pathlib import Path

import numpy as np


INPUT_PATH = (
    "data/processed/blast_radius_candidates.json"
)

OUTPUT_PATH = (
    "results/raw/blast_radius/"
    "blast_radius_scores.json"
)


CATEGORIES = [
    "related",
    "moderately_related",
    "unrelated"
]


def mean(values):

    if not values:
        return None

    return float(
        np.mean(values)
    )


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    results = []

    for target in data:

        category_scores = {
            category: []
            for category in CATEGORIES
        }

        for candidate in target["candidates"]:

            category = candidate["category"]

            if category not in CATEGORIES:
                continue

            category_scores[category].append(
                candidate["similarity"]
            )

        result = {
            "forget_id": target["forget_id"],
            "target_question":
                target["target_question"],
            "semantic_similarity": {
                category: mean(
                    category_scores[category]
                )
                for category in CATEGORIES
            },
            "candidate_counts": {
                category: len(
                    category_scores[category]
                )
                for category in CATEGORIES
            }
        }

        results.append(result)

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
    print("BLAST RADIUS ANALYSIS")
    print("===================================")

    print(
        "Targets:",
        len(results)
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()