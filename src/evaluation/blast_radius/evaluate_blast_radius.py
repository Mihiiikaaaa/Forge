import json
from pathlib import Path

INPUT_PATH = (
    "data/processed/blast_radius_candidates_reviewed.json"
)

OUTPUT_PATH = (
    "results/raw/blast_radius/blast_radius_scores.json"
)


def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    results = []

    total_candidates = 0
    reviewed_candidates = 0

    for target in targets:
        forget_id = target["forget_id"]
        target_question = target["target_question"]

        target_result = {
            "forget_id": forget_id,
            "target_question": target_question,
            "candidates": []
        }

        for candidate in target.get("candidates", []):
            total_candidates += 1

            category = candidate.get(
                "category",
                "unreviewed"
            )

            if category == "unreviewed":
                continue

            reviewed_candidates += 1

            target_result["candidates"].append({
                "retain_id": candidate["retain_id"],
                "question": candidate["question"],
                "similarity": candidate["similarity"],
                "similarity_band": candidate.get(
                    "similarity_band"
                ),
                "category": category,

                # Filled after actual model evaluation
                "before_likelihood": None,
                "after_likelihood": None,
                "damage": None
            })

        results.append(target_result)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n======================================")
    print("BLAST-RADIUS EVALUATION TEMPLATE")
    print("======================================")
    print("Targets:", len(targets))
    print("Total candidates:", total_candidates)
    print("Reviewed candidates:", reviewed_candidates)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()