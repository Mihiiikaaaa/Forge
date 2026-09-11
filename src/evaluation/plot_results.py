import json
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = (
    "results/tables/summary.json"
)

OUTPUT_DIR = Path(
    "results/plots"
)


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not data:

        print("No experiment results found.")
        return

    names = [
        item["experiment"]
        for item in data
    ]

    forgetting = [
        item["forgetting_score"]
        if item["forgetting_score"] is not None
        else 0
        for item in data
    ]

    retention = [
        item["retention_percentage"]
        if item["retention_percentage"] is not None
        else 0
        for item in data
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Forgetting comparison

    plt.figure(figsize=(10, 6))

    plt.bar(
        names,
        forgetting
    )

    plt.xlabel(
        "Experiment"
    )

    plt.ylabel(
        "Forgetting Score"
    )

    plt.title(
        "Forgetting Effectiveness"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "forgetting_comparison.png",
        dpi=300
    )

    plt.close()

    # Retention comparison

    plt.figure(figsize=(10, 6))

    plt.bar(
        names,
        retention
    )

    plt.xlabel(
        "Experiment"
    )

    plt.ylabel(
        "Retention (%)"
    )

    plt.title(
        "Knowledge Retention"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "retention_comparison.png",
        dpi=300
    )

    plt.close()

    print("\n===================================")
    print("PLOTS CREATED")
    print("===================================")

    print(
        "Output directory:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()