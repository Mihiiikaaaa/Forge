import json
import math
from pathlib import Path
from collections import defaultdict


CANDIDATE_PATH = (
    "data/processed/blast_radius_candidates_reviewed.json"
)

OUTPUT_PATH = (
    "results/raw/blast_radius/blast_radius_results.json"
)


VALID_CATEGORIES = {
    "related",
    "moderately_related",
    "unrelated"
}


def calculate_damage(before_nll, after_nll):
    """
    Positive value = model became worse after unlearning.
    Zero = no change.
    Negative value = model improved after unlearning.
    """
    if before_nll is None or after_nll is None:
        return None

    return after_nll - before_nll


def summarize(values):
    values = [
        value for value in values
        if value is not None and math.isfinite(value)
    ]

    if not values:
        return {
            "count": 0,
            "mean_damage": None,
            "median_damage": None,
            "max_damage": None,
            "percent_harmed": None
        }

    values_sorted = sorted(values)

    mean_damage = sum(values) / len(values)

    middle = len(values) // 2

    if len(values) % 2 == 0:
        median_damage = (
            values_sorted[middle - 1] +
            values_sorted[middle]
        ) / 2
    else:
        median_damage = values_sorted[middle]

    harmed = sum(1 for value in values if value > 0)

    return {
        "count": len(values),
        "mean_damage": mean_damage,
        "median_damage": median_damage,
        "max_damage": max(values),
        "percent_harmed": (
            harmed / len(values) * 100
        )
    }


def load_candidates():
    path = Path(CANDIDATE_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {CANDIDATE_PATH}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(candidates):
    category_values = defaultdict(list)
    distance_values = defaultdict(list)

    evaluated_candidates = []

    for candidate in candidates:
        category = candidate.get("category")

        if category not in VALID_CATEGORIES:
            continue

        before_nll = candidate.get("before_nll")
        after_nll = candidate.get("after_nll")

        damage = calculate_damage(
            before_nll,
            after_nll
        )

        candidate_result = dict(candidate)
        candidate_result["damage"] = damage

        evaluated_candidates.append(candidate_result)

        if damage is not None:
            category_values[category].append(damage)

            similarity = candidate.get(
                "cosine_similarity"
            )

            if similarity is not None:
                similarity = float(similarity)

                if similarity >= 0.85:
                    band = "high"
                elif similarity >= 0.70:
                    band = "medium"
                else:
                    band = "low"

                distance_values[band].append(damage)

    category_summary = {
        category: summarize(values)
        for category, values in category_values.items()
    }

    similarity_summary = {
        band: summarize(values)
        for band, values in distance_values.items()
    }

    return {
        "experiment": {
            "name": "FORGE_blast_radius",
            "metric": (
                "Collateral damage is measured as "
                "post-unlearning NLL minus pre-unlearning NLL."
            ),
            "positive_damage_means": (
                "The candidate knowledge became less usable "
                "after unlearning."
            )
        },
        "summary_by_category": category_summary,
        "summary_by_similarity_band": similarity_summary,
        "num_evaluated_candidates": len(evaluated_candidates),
        "candidates": evaluated_candidates
    }


def main():
    candidates = load_candidates()

    print("Candidates loaded:", len(candidates))

    results = evaluate(candidates)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n======================================")
    print("BLAST RADIUS EVALUATOR READY")
    print("======================================")
    print(
        "Evaluated candidates:",
        results["num_evaluated_candidates"]
    )
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()