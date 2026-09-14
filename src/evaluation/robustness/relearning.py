import json
from pathlib import Path


OUTPUT_PATH = "data/processed/relearning_plan.json"

RELEARNING_STEPS = [
    0,
    1,
    3,
    5,
    10
]


def create_relearning_plan():
    return {
        "experiment": {
            "name": "forgetting_relearning_resistance",
            "description": (
                "Measure how quickly forgotten knowledge returns "
                "after controlled retraining on the forget set."
            ),
            "forget_percentage": 1,
            "seed": 42
        },
        "steps": [
            {
                "relearning_steps": step,
                "checkpoint": None,
                "forget_knowledge_score": None,
                "retain_utility": None
            }
            for step in RELEARNING_STEPS
        ],
        "metrics": {
            "forget_knowledge_score": (
                "Model likelihood or equivalent knowledge-recovery "
                "measure on the forget set."
            ),
            "retain_utility": (
                "Utility measured on the retained evaluation set."
            ),
            "relearning_resistance": None
        }
    }


def main():
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plan = create_relearning_plan()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            plan,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n======================================")
    print("RELEARNING EXPERIMENT PLAN CREATED")
    print("======================================")
    print("Relearning steps:", RELEARNING_STEPS)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()