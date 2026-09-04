import json


INPUT_PATH = "data/processed/forget_paraphrases.json"

REQUIRED_TYPES = 4


def main():

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    errors = 0

    for item in data:

        example_id = item["id"]
        paraphrases = item["paraphrases"]

        if len(paraphrases) != REQUIRED_TYPES:
            print(
                f"Example {example_id}: "
                f"expected {REQUIRED_TYPES} paraphrases, "
                f"found {len(paraphrases)}"
            )

            errors += 1

        for p in paraphrases:

            if not isinstance(p, str):
                print(
                    f"Example {example_id}: "
                    "paraphrase is not a string"
                )

                errors += 1

            elif len(p.strip()) < 10:
                print(
                    f"Example {example_id}: "
                    "paraphrase is too short"
                )

                errors += 1

    print("\n===================================")

    if errors == 0:
        print("PARAPHRASE VALIDATION PASSED")
    else:
        print("PARAPHRASE VALIDATION FAILED")
        print("Errors:", errors)

    print("===================================")


if __name__ == "__main__":
    main()