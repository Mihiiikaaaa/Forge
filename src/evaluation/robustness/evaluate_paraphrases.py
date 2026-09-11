import json
from pathlib import Path

from answer_likelihood import calculate_answer_likelihood


INPUT_PATH = (
    "data/processed/forget_paraphrases.json"
)

OUTPUT_PATH = (
    "results/raw/robustness/paraphrase_results.json"
)


def evaluate(model, tokenizer, device):

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    results = []

    for item in data:

        target_result = {
            "id": item["id"],
            "original_question": item[
                "original_question"
            ],
            "paraphrases": []
        }

        for paraphrase in item["paraphrases"]:

            if not paraphrase.get(
                "human_verified",
                False
            ):
                continue

            if not paraphrase.get(
                "semantically_equivalent",
                False
            ):
                continue

            result = calculate_answer_likelihood(
                model=model,
                tokenizer=tokenizer,
                question=paraphrase["text"],
                answer=item["expected_answer"],
                device=device
            )

            target_result["paraphrases"].append(
                {
                    "text": paraphrase["text"],
                    "likelihood": result[
                        "likelihood"
                    ],
                    "normalized_log_likelihood":
                        result[
                            "normalized_log_likelihood"
                        ]
                }
            )

        results.append(target_result)

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "Paraphrase evaluation saved to:",
        OUTPUT_PATH
    )