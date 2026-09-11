import argparse
import json
from pathlib import Path


INPUT_PATH = "data/processed/blast_radius_candidates_banded.json"
OUTPUT_PATH = "data/processed/blast_radius_candidates_reviewed.json"


VALID_CATEGORIES = {
    "related",
    "moderately_related",
    "unrelated",
    "exclude"
}


MAPPING = {
    "r": "related",
    "m": "moderately_related",
    "u": "unrelated",
    "e": "exclude"
}


def load_data():
    """
    Load the original banded candidate file.

    If a reviewed file already exists, load that instead so
    previous annotations are preserved.
    """

    if Path(OUTPUT_PATH).exists():
        print("Loading existing reviewed data...")
        path = OUTPUT_PATH
    else:
        print("No reviewed file found.")
        print("Starting from banded candidates...")
        path = INPUT_PATH

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    """
    Save the current review progress.
    """

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def review_candidate(candidate):
    """
    Ask the reviewer to classify one candidate.
    """

    print("\n" + "-" * 80)

    print(
        "Retain ID:",
        candidate["retain_id"]
    )

    print(
        "Similarity:",
        f"{candidate['similarity']:.4f}"
    )

    print(
        "Similarity band:",
        candidate.get(
            "similarity_band",
            "unknown"
        )
    )

    print("\nQuestion:")
    print(candidate["question"])

    print("\nCategories:")
    print("r = related")
    print("m = moderately related")
    print("u = unrelated")
    print("e = exclude")

    while True:

        choice = input(
            "\nCategory: "
        ).strip().lower()

        if choice in MAPPING:
            return MAPPING[choice]

        print(
            "Invalid choice."
        )

        print(
            "Please enter r, m, u, or e."
        )


def get_review_counts(data):

    counts = {
        "related": 0,
        "moderately_related": 0,
        "unrelated": 0,
        "exclude": 0,
        "unreviewed": 0
    }

    for target in data:

        for candidate in target["candidates"]:

            category = candidate.get(
                "category",
                "unreviewed"
            )

            if category in VALID_CATEGORIES:
                counts[category] += 1
            else:
                counts["unreviewed"] += 1

    return counts


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--targets",
        type=int,
        default=None,
        help="Number of targets to inspect"
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Maximum candidates per target"
    )

    args = parser.parse_args()

    data = load_data()

    total_targets = len(data)

    total_candidates = sum(
        len(target["candidates"])
        for target in data
    )

    counts = get_review_counts(data)

    print("\n===================================")
    print("BLAST RADIUS CANDIDATE REVIEW")
    print("===================================")

    print(
        "Total targets:",
        total_targets
    )

    print(
        "Total candidates:",
        total_candidates
    )

    print(
        "Already reviewed:",
        total_candidates - counts["unreviewed"]
    )

    print(
        "Remaining:",
        counts["unreviewed"]
    )

    print("\nCurrent distribution:")

    for category in [
        "related",
        "moderately_related",
        "unrelated",
        "exclude"
    ]:
        print(
            f"{category}:",
            counts[category]
        )

    # If no limits are provided, review everything.
    target_limit = (
        args.targets
        if args.targets is not None
        else total_targets
    )

    candidates_limit = (
        args.candidates
        if args.candidates is not None
        else None
    )

    targets = data[:target_limit]

    print("\n===================================")
    print("REVIEW SETTINGS")
    print("===================================")

    print(
        "Targets in this session:",
        len(targets)
    )

    if candidates_limit is None:
        print(
            "Candidates per target: ALL"
        )
    else:
        print(
            "Candidates per target:",
            candidates_limit
        )

    print(
        "\nAlready reviewed candidates "
        "will be skipped automatically."
    )

    print(
        "Press Ctrl+C at any time to stop."
    )

    for target in targets:

        print("\n" + "=" * 90)

        print(
            f"TARGET {target['forget_id']}"
        )

        print("\nTarget question:")
        print(target["target_question"])

        candidates = target["candidates"]

        if candidates_limit is not None:
            candidates = candidates[
                :candidates_limit
            ]

        for candidate in candidates:

            # Resume support:
            # Do not ask again if already reviewed.
            if candidate.get("category") in VALID_CATEGORIES:

                continue

            category = review_candidate(
                candidate
            )

            candidate["category"] = category

            save_data(data)

            print(
                "\nSaved:",
                category
            )

    counts = get_review_counts(data)

    print("\n===================================")
    print("REVIEW SESSION COMPLETE")
    print("===================================")

    print(
        "Reviewed:",
        total_candidates - counts["unreviewed"]
    )

    print(
        "Remaining:",
        counts["unreviewed"]
    )

    print(
        "\nSaved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()