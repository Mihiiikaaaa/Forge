def accuracy_score(correct, total):
    """
    Calculate accuracy as a percentage.
    """
    if total == 0:
        return 0.0

    return (correct / total) * 100


def forgetting_score(before, after):
    """
    Measures reduction in performance on the forget set.

    Higher = more forgetting.
    """
    return before - after


def retention_score(after):
    """
    Measures retained performance after unlearning.
    """
    return after


def relative_drop(before, after):
    """
    Percentage drop from before to after.
    """
    if before == 0:
        return 0.0

    return ((before - after) / before) * 100