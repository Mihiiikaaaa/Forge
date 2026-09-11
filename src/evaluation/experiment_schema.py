def create_experiment_result(
    experiment_name,
    model_name,
    forget_percentage,
    seed
):

    return {
        "experiment": {
            "name": experiment_name,
            "model": model_name,
            "forget_percentage":
                forget_percentage,
            "seed": seed
        },

        "forgetting": {
            "before_likelihood": None,
            "after_likelihood": None,
            "relative_drop": None,
            "forgetting_score": None
        },

        "retention": {
            "before_likelihood": None,
            "after_likelihood": None,
            "retention_percentage": None
        },

        "robustness": {
            "paraphrase_likelihood": None,
            "paraphrase_drop": None,
            "indirect_inference": None,
            "relearning_resistance": None
        },

        "blast_radius": {
            "related_damage": None,
            "moderately_related_damage": None,
            "unrelated_damage": None
        },

        "utility": {
            "overall_retention": None
        }
    }