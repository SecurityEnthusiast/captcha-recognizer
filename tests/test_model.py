"""The five-head network, dropout, and a saved round trip."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


def test_each_character_reads_its_own_columns():
    from captcha_recognizer import ModelConfig, create_model
    from captcha_recognizer.model import character_windows

    model = create_model(ModelConfig(), learning_rate=1e-3)
    assert tuple(model.input.shape) == (None, 48, 128, 1)
    assert tuple(model.get_layer("columns").output.shape) == (None, 32, 64)
    assert tuple(model.get_layer("pool1").pool_size) == (2, 2)
    assert tuple(model.get_layer("pool2").pool_size) == (2, 2)
    assert tuple(model.get_layer("pool3").pool_size) == (2, 1)
    assert tuple(model.get_layer("pool4").pool_size) == (2, 1)
    assert model.get_layer("char_classifier").units == 62
    names = {layer.name for layer in model.layers}
    assert "dense1" not in names
    windows = character_windows(32, 5)
    assert windows == [(0, 8), (6, 8), (12, 8), (18, 8), (24, 8)]
    for index, (start, length) in enumerate(windows):
        span = model.get_layer(f"char_{index}_span")
        assert tuple(span.cropping) == (start, 32 - (start + length))
        head = model.get_layer(f"char_{index}_output")
        assert head.name == f"char_{index}_output"
    for index in range(1, 5):
        assert model.get_layer(f"conv{index}").filters == 64


def test_dropout_is_a_layer_only_when_the_rate_is_set():
    from captcha_recognizer import ModelConfig, create_model

    config = ModelConfig()
    with_dropout = create_model(config, dropout=0.5)
    assert with_dropout.get_layer("dropout").rate == 0.5

    without_dropout = create_model(config, dropout=0)
    names = {layer.name for layer in without_dropout.layers}
    assert "dropout" not in names
    assert "char_4_output" in names


def test_a_loaded_model_matches_the_one_just_saved(tmp_path):
    import tensorflow as tf

    from captcha_recognizer import ModelConfig, create_model
    from captcha_recognizer.model import load_model, save_model

    model = create_model(ModelConfig(), learning_rate=1e-3, dropout=0)
    sample = tf.zeros((1, 48, 128, 1))
    live = model(sample, training=False)
    path = tmp_path / "captcha_model.keras"
    save_model(model, str(path))
    loaded = load_model(str(path))
    again = loaded(sample, training=False)

    assert loaded.output_names == model.output_names
    for name in model.output_names:
        np.testing.assert_allclose(
            live[name].numpy(),
            again[name].numpy(),
            atol=1e-5,
        )
