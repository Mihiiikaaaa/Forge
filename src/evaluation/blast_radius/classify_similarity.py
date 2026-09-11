import json
from pathlib import Path


INPUT_PATH = (
    "data/processed/blast_radius_candidates.json"
)

OUTPUT_PATH = (
    "data/processed/blast_radius_candidates_banded.json"
)


def similarity_band(similarity):

    if similarity >= 0.85:
        return "high"

    if similarity >= 0.70:
        return "medium"

    return "low"


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    high = 0
    medium = 0
    low = 0

    for target in data:

        for candidate in target["candidates"]:

            similarity = candidate["similarity"]

            band = similarity_band(similarity)

            candidate["similarity_band"] = band

            if band == "high":
                high += 1

            elif band == "medium":
                medium += 1

            else:
                low += 1

    output_path = Path(OUTPUT_PATH)

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
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n===================================")
    print("SIMILARITY BANDS CREATED")
    print("===================================")

    print("High similarity:", high)
    print("Medium similarity:", medium)
    print("Low similarity:", low)

    print(
        "\nSaved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()