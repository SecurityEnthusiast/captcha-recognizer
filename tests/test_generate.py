"""The generator can repeat a glyph, and the demo bar is strict."""

import random


def test_labels_can_repeat_and_use_the_requested_length():
    from captcha_recognizer.generate import random_label

    rng = random.Random(0)
    labels = [random_label(rng, "ab", 5) for _ in range(30)]
    assert all(len(label) == 5 for label in labels)
    assert any(len(set(label)) < len(label) for label in labels)


def test_generated_images_match_the_model_size(tmp_path):
    from PIL import Image

    from captcha_recognizer.data_loader import label_from_filename
    from captcha_recognizer.generate import generate

    paths = generate(
        tmp_path,
        2,
        chars_num=4,
        char_set="abc",
        seed=1,
        width=128,
        height=48,
    )
    assert len(paths) == 2
    with Image.open(paths[0]) as image:
        assert image.size == (128, 48)
    assert len(label_from_filename(paths[0].name)) == 4


def test_learning_bar_requires_a_loss_drop_and_better_than_chance():
    from captcha_recognizer.demo import learning_bar_passed

    assert learning_bar_passed(1.0, 0.5, 0.2, 62) is True
    assert learning_bar_passed(1.0, 1.0, 0.2, 62) is False
    assert learning_bar_passed(1.0, 0.5, 1 / 62, 62) is False


def test_holdout_is_not_used_as_training_data():
    from captcha_recognizer.demo import split_files

    files = [f"img_{index}.png" for index in range(10)]
    train_files, holdout_files = split_files(files, 0.2, seed=3)
    assert holdout_files
    assert train_files
    assert set(train_files).isdisjoint(holdout_files)
    assert set(train_files) | set(holdout_files) == set(files)
