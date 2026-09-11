import json
from pathlib import Path


INPUT_PATH = "data/processed/forget_paraphrases.json"


def save_data(data):
    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    path = Path(INPUT_PATH)

    if not path.exists():
        print("Paraphrase file not found.")
        print("Run paraphrases.py first.")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n===================================")
    print("PARAPHRASE REVIEW")
    print("===================================")
    print("Targets:", len(data))

    for item in data:

        print("\n" + "=" * 80)
        print("TARGET ID:", item["id"])
        print("=" * 80)

        print("\nOriginal question:")
        print(item["original_question"])

        print("\nExpected answer:")
        print(item["expected_answer"])

        for i, paraphrase in enumerate(item["paraphrases"]):

            print("\n-----------------------------------")
            print(f"Paraphrase {i + 1}")
            print("-----------------------------------")

            current = paraphrase["text"]

            if current:
                print("Current:", current)
            else:
                print("Current: EMPTY")

            new_text = input(
                "Enter paraphrase "
                "(leave blank to keep current): "
            ).strip()

            if new_text:
                paraphrase["text"] = new_text

            verify = input(
                "Is this semantically equivalent? [y/n]: "
            ).strip().lower()

            if verify == "y":
                paraphrase["human_verified"] = True
                paraphrase["semantically_equivalent"] = True
            else:
                paraphrase["human_verified"] = False
                paraphrase["semantically_equivalent"] = False

            save_data(data)

        print("\nProgress saved.")

    print("\n===================================")
    print("REVIEW COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()