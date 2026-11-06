"""Decode and score without treating a substring as a full hit."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


def test_decode_uses_the_class_axis():
    from captcha_recognizer.decode import decode_heads

    head = np.zeros((1, 5), dtype=np.float32)
    head[0, 4] = 1.0
    text, confidence = decode_heads([head] * 5, "abcde")
    assert text == "eeeee"
    assert confidence == 1.0


def test_substring_is_not_an_exact_match():
    from captcha_recognizer.metrics import character_accuracy, exact_matches

    pairs = [("1ab2", "1ab2s")]
    assert exact_matches(pairs) == 0.0
    assert character_accuracy(pairs) == 4 / 5


def test_predict_returns_a_string_and_a_confidence(tmp_path):
    from PIL import Image

    from captcha_recognizer import Config, create_model, predict

    path = tmp_path / "aaaaa_num0.png"
    Image.fromarray(np.zeros((48, 128), dtype=np.uint8), mode="L").save(path)
    config = Config()
    model = create_model(config.model, learning_rate=1e-3)
    text, confidence = predict(path, model, config)

    assert len(text) == 5
    assert set(text) <= set(config.model.char_set)
    assert 0.0 <= confidence <= 1.0


def test_inference_is_deterministic():
    import tensorflow as tf

    from captcha_recognizer import ModelConfig, create_model

    model = create_model(ModelConfig(), learning_rate=1e-3)
    sample = tf.random.uniform((1, 48, 128, 1))
    first = model(sample, training=False)["char_0_output"].numpy()
    second = model(sample, training=False)["char_0_output"].numpy()
    np.testing.assert_allclose(first, second)
