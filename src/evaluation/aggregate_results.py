import json
from pathlib import Path

import numpy as np


RESULTS_DIR = "results/raw/experiments"
OUTPUT_PATH = "results/tables/summary.json"


def safe_mean(values):

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        np.mean(values)
    )


def main():

    directory = Path(
        RESULTS_DIR
    )

    files = list(
        directory.glob("*.json")
    )

    experiments = []

    for file in files:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            experiments.append(
                json.load(f)
            )

    summary = []

    for experiment in experiments:

        forgetting = experiment[
            "forgetting"
        ]

        retention = experiment[
            "retention"
        ]

        summary.append(
            {
                "experiment":
                    experiment[
                        "experiment"
                    ]["name"],

                "model":
                    experiment[
                        "experiment"
                    ]["model"],

                "forget_percentage":
                    experiment[
                        "experiment"
                    ]["forget_percentage"],

                "forgetting_score":
                    forgetting[
                        "forgetting_score"
                    ],

                "relative_drop":
                    forgetting[
                        "relative_drop"
                    ],

                "retention_percentage":
                    retention[
                        "retention_percentage"
                    ]
            }
        )

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
            summary,
            f,
            indent=4
        )

    print("\n===================================")
    print("RESULT AGGREGATION")
    print("===================================")

    print(
        "Experiments:",
        len(summary)
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()