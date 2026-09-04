def accuracy_score(correct, total):
    if total == 0:
        return 0.0
    return (correct / total) * 100


def forgetting_score(before, after):
    """
    Measures how much target knowledge was forgotten.

    Higher value = more forgetting.
    """
    return before - after


def retention_score(before, after):
    """
    Measures how much knowledge was retained.

    Higher value = better retention.
    """
    if before == 0:
        return 0.0

    return (after / before) * 100


def relative_drop(before, after):
    """
    Percentage decrease from before to after.
    """
    if before == 0:
        return 0.0

    return ((before - after) / before) * 100