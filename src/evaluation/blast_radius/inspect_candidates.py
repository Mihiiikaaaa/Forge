import json

INPUT_PATH = "data/processed/blast_radius_candidates.json"

SHOW_TARGETS = 5
SHOW_CANDIDATES = 5


def main():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n===================================")
    print("BLAST RADIUS CANDIDATE INSPECTION")
    print("===================================")

    print(
        f"Showing {SHOW_TARGETS} targets "
        f"with {SHOW_CANDIDATES} candidates each."
    )

    for target in data[:SHOW_TARGETS]:

        print("\n" + "=" * 90)

        print(
            f"TARGET {target['forget_id']}"
        )

        print(
            f"\nTarget question:\n"
            f"{target['target_question']}"
        )

        print("\nTop candidates:")

        for rank, candidate in enumerate(
            target["candidates"][:SHOW_CANDIDATES],
            start=1
        ):

            print("\n" + "-" * 80)

            print(
                f"Rank: {rank}"
            )

            print(
                f"Retain ID: "
                f"{candidate['retain_id']}"
            )

            print(
                f"Similarity: "
                f"{candidate['similarity']:.4f}"
            )

            print(
                f"Question:\n"
                f"{candidate['question']}"
            )


if __name__ == "__main__":
    main()