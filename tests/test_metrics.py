"""Exact match is stricter than per-character accuracy."""

from captcha_recognizer.labels import label_from_filename
from captcha_recognizer.metrics import per_character_accuracy, score_pairs


def test_near_miss_is_not_an_exact_match():
    scores = score_pairs([("abcde", "abcde"), ("abcde", "abcdf")])
    assert scores["exact_match"] == 0.5
    assert scores["exact_correct"] == 1
    assert per_character_accuracy("abcde", "abcdf") == 0.8
    assert scores["per_character"] == (1.0 + 0.8) / 2


def test_filename_label_is_the_prefix_not_a_substring():
    # "1ab2" occurs inside this name. The label is the whole prefix.
    assert label_from_filename("1ab2s_num286.jpg") == "1ab2s"
    assert label_from_filename("CMQVA_num717_1.png") == "CMQVA"


if __name__ == "__main__":
    test_near_miss_is_not_an_exact_match()
    test_filename_label_is_the_prefix_not_a_substring()
    print("metric tests passed")
