from datasets import load_dataset

DATASET_NAME = "locuslab/TOFU"


def main():
    print("Loading TOFU dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully!")
    print(dataset)

    for split in dataset:
        print(f"\n--- {split} ---")
        print("Number of examples:", len(dataset[split]))
        print("Columns:", dataset[split].column_names)

        print("\nFirst example:")
        print(dataset[split][0])


if __name__ == "__main__":
    main()