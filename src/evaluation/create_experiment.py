import argparse
import json
from pathlib import Path

from experiment_schema import create_experiment_result


OUTPUT_DIR = Path("results/raw/experiments")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--forget",
        type=int,
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    result = create_experiment_result(
        experiment_name=args.name,
        model_name=args.model,
        forget_percentage=args.forget,
        seed=args.seed
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / f"{args.name}.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=4
        )

    print("Experiment template created.")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()