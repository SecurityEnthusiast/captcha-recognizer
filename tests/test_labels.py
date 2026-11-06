"""Filename labels do not need TensorFlow."""

from captcha_recognizer.labels import label_from_filename, one_hot_heads


def test_label_is_the_prefix_before_the_underscore():
    assert label_from_filename("1ab2s_num286.jpg") == "1ab2s"
    assert label_from_filename("CMQVA_num717_1.png") == "CMQVA"
    assert label_from_filename("abcde.png") == "abcde"


def test_one_hot_heads_match_character_positions():
    heads = one_hot_heads("ab", "ab", 2)
    assert len(heads) == 2
    assert heads[0].tolist() == [1.0, 0.0]
    assert heads[1].tolist() == [0.0, 1.0]


if __name__ == "__main__":
    test_label_is_the_prefix_before_the_underscore()
    test_one_hot_heads_match_character_positions()
    print("label tests passed")
