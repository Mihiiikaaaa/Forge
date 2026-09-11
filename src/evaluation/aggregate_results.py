import json
from pathlib import Path


INPUT_DIR = Path("results/raw/experiments")
OUTPUT_PATH = Path("results/tables/summary.json")


def main():

    if not INPUT_DIR.exists():
        print("No experiment results found.")
        return

    files = sorted(
        INPUT_DIR.glob("*.json")
    )

    if not files:
        print("No experiment result files found.")
        return

    experiments = []

    for file in files:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        experiments.append(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            experiments,
            f,
            indent=4
        )

    print("===================================")
    print("RESULTS AGGREGATED")
    print("===================================")

    print(
        "Experiments:",
        len(experiments)
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()