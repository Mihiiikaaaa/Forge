import json
from pathlib import Path


INPUT_PATH = "data/processed/forget_paraphrases.json"

EXPECTED_PARAPHRASES = 4


def main():

    path = Path(INPUT_PATH)

    if not path.exists():
        print("ERROR: paraphrase dataset not found.")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = []
    valid_count = 0

    for item in data:

        item_id = item["id"]

        if len(item["paraphrases"]) != EXPECTED_PARAPHRASES:
            errors.append(
                f"ID {item_id}: incorrect number of paraphrases"
            )
            continue

        for index, paraphrase in enumerate(item["paraphrases"]):

            text = paraphrase.get("text", "").strip()

            if not text:
                errors.append(
                    f"ID {item_id}, paraphrase {index + 1}: empty"
                )

            if not paraphrase.get("human_verified", False):
                errors.append(
                    f"ID {item_id}, paraphrase {index + 1}: "
                    "not human verified"
                )

            if not paraphrase.get(
                "semantically_equivalent", False
            ):
                errors.append(
                    f"ID {item_id}, paraphrase {index + 1}: "
                    "not marked semantically equivalent"
                )

        if not any(
            str(item_id) in error for error in errors
        ):
            valid_count += 1

    print("\n===================================")
    print("PARAPHRASE VALIDATION")
    print("===================================")

    print("Targets:", len(data))
    print("Valid targets:", valid_count)
    print("Errors:", len(errors))

    if errors:
        print("\nProblems:")

        for error in errors[:30]:
            print("-", error)

        if len(errors) > 30:
            print(
                f"... and {len(errors) - 30} more"
            )

    else:
        print("\nAll paraphrase entries are valid.")


if __name__ == "__main__":
    main()