import json
from pathlib import Path

FORGET_PATH = "data/processed/forget_1"
RETAIN_PATH = "data/processed/retain_1"

PARAPHRASE_PATH = (
    "data/processed/forget_paraphrases.json"
)

INDIRECT_PATH = (
    "data/processed/indirect_inference.json"
)

BLAST_RADIUS_PATH = (
    "data/processed/blast_radius_candidates_reviewed.json"
)

OUTPUT_PATH = (
    "data/processed/evaluation_manifest.json"
)


def count_json_records(path):
    path = Path(path)

    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return len(data)


def count_nested_candidates(path):
    path = Path(path)

    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    total = 0
    reviewed = 0

    for target in targets:
        for candidate in target.get("candidates", []):
            total += 1

            if candidate.get(
                "category",
                "unreviewed"
            ) != "unreviewed":
                reviewed += 1

    return {
        "targets": len(targets),
        "total_candidates": total,
        "reviewed_candidates": reviewed
    }


def main():
    blast_info = count_nested_candidates(
        BLAST_RADIUS_PATH
    )

    manifest = {
        "forget_set": {
            "path": FORGET_PATH,
            "examples": 40
        },

        "retain_set": {
            "path": RETAIN_PATH,
            "examples": 3960
        },

        "paraphrases": {
            "path": PARAPHRASE_PATH,
            "targets": count_json_records(
                PARAPHRASE_PATH
            ),
            "paraphrases_per_target": 4
        },

        "indirect_inference": {
            "path": INDIRECT_PATH,
            "targets": count_json_records(
                INDIRECT_PATH
            ),
            "questions_per_target": 3
        },

        "blast_radius": {
            "path": BLAST_RADIUS_PATH,
            **blast_info
        }
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=4
        )

    print("\n======================================")
    print("FORGE EVALUATION MANIFEST")
    print("======================================")

    print(
        "Forget examples:",
        manifest["forget_set"]["examples"]
    )

    print(
        "Retain examples:",
        manifest["retain_set"]["examples"]
    )

    print(
        "Paraphrase targets:",
        manifest["paraphrases"]["targets"]
    )

    print(
        "Indirect targets:",
        manifest["indirect_inference"]["targets"]
    )

    print(
        "Blast-radius targets:",
        manifest["blast_radius"]["targets"]
    )

    print(
        "Blast-radius candidates:",
        manifest["blast_radius"]["total_candidates"]
    )

    print(
        "Blast-radius reviewed:",
        manifest["blast_radius"]["reviewed_candidates"]
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()