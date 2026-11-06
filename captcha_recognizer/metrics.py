"""Exact-match and per-character scores.

A prediction is correct only when every character matches. A shorter string
that merely occurs inside the filename is not a hit.
"""


def exact_matches(pairs) -> float:
    """Fraction of pairs whose two strings are equal."""
    if not pairs:
        return 0.0
    hits = sum(predicted == expected for predicted, expected in pairs)
    return hits / len(pairs)


def character_accuracy(pairs) -> float:
    """Mean share of positions that match, across every pair."""
    correct = 0
    total = 0
    for predicted, expected in pairs:
        width = max(len(predicted), len(expected))
        total += width
        correct += sum(left == right for left, right in zip(predicted, expected))
    if total == 0:
        return 0.0
    return correct / total
