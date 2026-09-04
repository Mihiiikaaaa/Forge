from datasets import load_dataset, Dataset
import random
import json
import os

DATASET_NAME = "locuslab/TOFU"
SEED = 42

OUTPUT_DIR = "data/processed"


def create_split(dataset, forget_percentage):
    random.seed(SEED)

    total = len(dataset)
    forget_size = int(total * forget_percentage / 100)

    indices = list(range(total))
    random.shuffle(indices)

    forget_indices = indices[:forget_size]
    retain_indices = indices[forget_size:]

    forget_set = dataset.select(forget_indices)
    retain_set = dataset.select(retain_indices)

    return forget_set, retain_set


def save_dataset(dataset, path):
    dataset.save_to_disk(path)


def main():
    print("Loading TOFU...")

    dataset = load_dataset(DATASET_NAME)["train"]

    print(f"Total examples: {len(dataset)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    statistics = {
        "dataset": DATASET_NAME,
        "total_examples": len(dataset),
        "seed": SEED,
        "splits": {}
    }

    for percentage in [1, 5, 10]:

        print(f"\nCreating {percentage}% forget set...")

        forget_set, retain_set = create_split(
            dataset,
            percentage
        )

        forget_path = f"{OUTPUT_DIR}/forget_{percentage}"
        retain_path = f"{OUTPUT_DIR}/retain_{percentage}"

        save_dataset(forget_set, forget_path)
        save_dataset(retain_set, retain_path)

        statistics["splits"][f"{percentage}%"] = {
            "forget_examples": len(forget_set),
            "retain_examples": len(retain_set)
        }

        print(f"Forget set: {len(forget_set)}")
        print(f"Retain set: {len(retain_set)}")

    stats_path = f"{OUTPUT_DIR}/dataset_statistics.json"

    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=4)

    print("\nDataset preparation complete!")
    print(f"Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()