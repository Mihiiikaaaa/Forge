import json
from pathlib import Path


INPUT_PATH = "data/processed/indirect_inference.json"

REQUIRED_TYPES = {
    "contextual",
    "attribute_based",
    "multi_hop"
}


def validate_entry(entry, errors):
    entry_id = entry.get("id")

    if "original_question" not in entry:
        errors.append(
            f"Target {entry_id}: missing original_question"
        )

    if "target_answer" not in entry:
        errors.append(
            f"Target {entry_id}: missing target_answer"
        )

    questions = entry.get("indirect_questions")

    if not isinstance(questions, list):
        errors.append(
            f"Target {entry_id}: indirect_questions must be a list"
        )
        return

    if len(questions) != 3:
        errors.append(
            f"Target {entry_id}: expected 3 indirect questions, "
            f"found {len(questions)}"
        )

    seen_types = set()

    for index, item in enumerate(questions):
        item_type = item.get("type")

        if item_type not in REQUIRED_TYPES:
            errors.append(
                f"Target {entry_id}, question {index}: "
                f"invalid type '{item_type}'"
            )
        else:
            seen_types.add(item_type)

        question = item.get("question", "").strip()
        expected_answer = item.get("expected_answer", "").strip()

        if not question:
            errors.append(
                f"Target {entry_id}, question {index}: "
                "question is empty"
            )

        if not expected_answer:
            errors.append(
                f"Target {entry_id}, question {index}: "
                "expected_answer is empty"
            )

        if item.get("human_verified") is not True:
            errors.append(
                f"Target {entry_id}, question {index}: "
                "human_verified is not True"
            )

        if item.get(
            "semantically_targets_forgotten_knowledge"
        ) is not True:
            errors.append(
                f"Target {entry_id}, question {index}: "
                "semantic verification is not True"
            )

    missing_types = REQUIRED_TYPES - seen_types

    for missing_type in missing_types:
        errors.append(
            f"Target {entry_id}: missing type '{missing_type}'"
        )


def main():
    path = Path(INPUT_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = []

    if not isinstance(data, list):
        raise ValueError(
            "Indirect inference file must contain a list."
        )

    for entry in data:
        validate_entry(entry, errors)

    total_targets = len(data)
    total_questions = total_targets * 3

    verified_questions = 0

    for entry in data:
        for item in entry.get("indirect_questions", []):
            if (
                item.get("human_verified") is True
                and item.get(
                    "semantically_targets_forgotten_knowledge"
                ) is True
                and item.get("question", "").strip()
                and item.get("expected_answer", "").strip()
            ):
                verified_questions += 1

    print("\n======================================")
    print("INDIRECT INFERENCE VALIDATION")
    print("======================================")
    print("Targets:", total_targets)
    print("Expected questions:", total_questions)
    print("Verified questions:", verified_questions)
    print("Errors:", len(errors))

    if errors:
        print("\nFirst 20 errors:")
        for error in errors[:20]:
            print("-", error)

        print(
            "\nValidation FAILED. "
            "Do not use this dataset for experiments."
        )
    else:
        print("\nValidation PASSED.")


if __name__ == "__main__":
    main()