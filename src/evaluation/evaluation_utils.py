import json
from pathlib import Path


def save_results(results, output_path):
    """
    Save evaluation results as JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


def load_results(path):
    """
    Load evaluation results from JSON.
    """

    with open(path, "r") as f:
        return json.load(f)