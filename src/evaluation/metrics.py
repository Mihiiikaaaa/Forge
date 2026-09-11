def accuracy_score(correct, total):
    if total == 0:
        return 0.0

    return (correct / total) * 100


def forgetting_score(before, after):
    """
    Absolute decrease in target knowledge likelihood.
    Higher = stronger forgetting.
    """

    return before - after


def relative_drop(before, after):
    """
    Percentage decrease from before to after.
    """

    if before == 0:
        return 0.0

    return ((before - after) / before) * 100


def retention_score(before, after):
    """
    Percentage of original knowledge retained.
    """

    if before == 0:
        return 0.0

    return (after / before) * 100


def truth_ratio(forget_probability, retain_probability):
    """
    Ratio between forgotten and retained knowledge probability.

    Lower values indicate stronger separation.
    """

    if retain_probability == 0:
        return 0.0

    return forget_probability / retain_probability


def model_utility(before, after):
    """
    Relative utility retained after unlearning.
    """

    if before == 0:
        return 0.0

    return (after / before) * 100