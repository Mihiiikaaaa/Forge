import json
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = (
    "data/processed/blast_radius_candidates_reviewed.json"
)


VALID_CATEGORIES = {
    "related",
    "moderately_related",
    "unrelated",
    "exclude",
    "unreviewed"
}


def main():
    path = Path(INPUT_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    with open(path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    category_counts = Counter()
    target_counts = defaultdict(Counter)

    total_candidates = 0

    for target in targets:
        forget_id = target.get("forget_id")

        candidates = target.get("candidates", [])

        for candidate in candidates:
            total_candidates += 1

            category = candidate.get(
                "category",
                "unreviewed"
            )

            if category not in VALID_CATEGORIES:
                category = "unreviewed"

            category_counts[category] += 1
            target_counts[forget_id][category] += 1

    reviewed = (
        category_counts["related"]
        + category_counts["moderately_related"]
        + category_counts["unrelated"]
        + category_counts["exclude"]
    )

    unreviewed = category_counts["unreviewed"]

    print("\n======================================")
    print("BLAST-RADIUS ANNOTATION ANALYSIS")
    print("======================================")

    print("\nOverall:")
    print("Targets:", len(targets))
    print("Total candidates:", total_candidates)
    print("Reviewed:", reviewed)
    print("Unreviewed:", unreviewed)

    print("\nCategory counts:")

    for category in [
        "related",
        "moderately_related",
        "unrelated",
        "exclude",
        "unreviewed"
    ]:
        count = category_counts[category]

        percentage = (
            count / total_candidates * 100
            if total_candidates > 0
            else 0
        )

        print(
            f"{category:20s}: "
            f"{count:4d} "
            f"({percentage:6.2f}%)"
        )

    print("\nReviewed category distribution:")

    if reviewed > 0:
        for category in [
            "related",
            "moderately_related",
            "unrelated",
            "exclude"
        ]:
            count = category_counts[category]

            percentage = count / reviewed * 100

            print(
                f"{category:20s}: "
                f"{count:4d} "
                f"({percentage:6.2f}%)"
            )

    print("\nPer-target reviewed counts:")

    targets_with_annotations = 0

    for forget_id in sorted(
        target_counts,
        key=lambda x: int(x)
    ):
        counts = target_counts[forget_id]

        target_reviewed = (
            counts["related"]
            + counts["moderately_related"]
            + counts["unrelated"]
            + counts["exclude"]
        )

        if target_reviewed > 0:
            targets_with_annotations += 1

            print(
                f"Target {forget_id}: "
                f"{target_reviewed} reviewed | "
                f"R={counts['related']} "
                f"M={counts['moderately_related']} "
                f"U={counts['unrelated']} "
                f"E={counts['exclude']}"
            )

    print(
        "\nTargets with at least one annotation:",
        targets_with_annotations
    )


if __name__ == "__main__":
    main()