import numpy as np


def mean(values):
    if not values:
        return 0.0

    return float(np.mean(values))


def median(values):
    if not values:
        return 0.0

    return float(np.median(values))


def standard_deviation(values):
    if len(values) < 2:
        return 0.0

    return float(np.std(values, ddof=1))


def percentile(values, p):
    if not values:
        return 0.0

    return float(np.percentile(values, p))


def summarize(values):
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0
        }

    return {
        "mean": mean(values),
        "median": median(values),
        "std": standard_deviation(values),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75)
    }