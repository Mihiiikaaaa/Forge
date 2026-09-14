import json
from collections import defaultdict
from pathlib import Path

INPUT_PATH = (
    "data/processed/blast_radius_candidates_reviewed.json"
)


def main():
    path = Path(INPUT_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    with open(path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    bands = defaultdict(list)
    categories = defaultdict(list)

    total = 0

    for target in targets:
        for candidate in target.get("candidates", []):
            category = candidate.get(
                "category",
                "unreviewed"
            )

            if category == "unreviewed":
                continue

            similarity_band = candidate.get(
                "similarity_band",
                "unknown"
            )

            similarity = candidate.get(
                "similarity"
            )

            total += 1

            bands[similarity_band].append(similarity)
            categories[category].append(similarity)

    print("\n======================================")
    print("SEMANTIC DISTANCE ANALYSIS")
    print("======================================")

    print("\nReviewed candidates:", total)

    print("\nSimilarity bands:")

    for band in [
        "high",
        "medium",
        "low"
    ]:
        values = bands.get(band, [])

        if values:
            mean_similarity = (
                sum(values) / len(values)
            )

            print(
                f"{band:10s}: "
                f"{len(values):4d} candidates | "
                f"mean similarity = "
                f"{mean_similarity:.4f}"
            )
        else:
            print(
                f"{band:10s}: 0 candidates"
            )

    print("\nHuman semantic categories:")

    for category in [
        "related",
        "moderately_related",
        "unrelated",
        "exclude"
    ]:
        values = categories.get(category, [])

        if values:
            mean_similarity = (
                sum(values) / len(values)
            )

            print(
                f"{category:20s}: "
                f"{len(values):4d} candidates | "
                f"mean similarity = "
                f"{mean_similarity:.4f}"
            )
        else:
            print(
                f"{category:20s}: 0 candidates"
            )


if __name__ == "__main__":
    main()