"""Exact-match and per-character scores for 5-character predictions."""


def exact_match(predicted: str, truth: str) -> bool:
    """True only when every character matches."""
    return predicted == truth


def per_character_accuracy(predicted: str, truth: str) -> float:
    """Fraction of positions that match. A one-character miss is 0.8, not a hit."""
    if not truth:
        return 0.0
    hits = sum(left == right for left, right in zip(predicted, truth))
    return hits / len(truth)


def score_pairs(pairs) -> dict:
    """Score ``(predicted, truth)`` pairs.

    ``exact_match`` is the fraction of strings that match completely.
    ``per_character`` is the mean per-string character accuracy.
    """
    total = len(pairs)
    if total == 0:
        return {
            "exact_match": 0.0,
            "per_character": 0.0,
            "total": 0,
            "exact_correct": 0,
        }
    exact_correct = sum(exact_match(predicted, truth) for predicted, truth in pairs)
    per_character = sum(
        per_character_accuracy(predicted, truth) for predicted, truth in pairs
    ) / total
    return {
        "exact_match": exact_correct / total,
        "per_character": per_character,
        "total": total,
        "exact_correct": exact_correct,
    }
