import numpy as np


def mean(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return float(
        np.mean(values)
    )


def median(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return float(
        np.median(values)
    )


def standard_deviation(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return float(
        np.std(values)
    )


def percentile(values, q):

    values = np.asarray(
        values,
        dtype=float
    )

    return float(
        np.percentile(values, q)
    )


def summarize(values):

    return {
        "mean": mean(values),
        "median": median(values),
        "std": standard_deviation(values),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75)
    }