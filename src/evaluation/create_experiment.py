import argparse
import json
from pathlib import Path

from src.evaluation.experiment_schema import (
    create_experiment_result
)


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
        "--forget-percentage",
        type=int,
        default=1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    result = create_experiment_result(
        experiment_name=args.name,
        model_name=args.model,
        forget_percentage=
            args.forget_percentage,
        seed=args.seed
    )

    output_path = Path(args.output)

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
            result,
            f,
            indent=4
        )

    print(
        "Experiment template created:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()