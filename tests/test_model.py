"""The five-head network, dropout, and a saved round trip."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


def test_the_pooled_map_feeds_five_heads():
    from captcha_recognizer import ModelConfig, create_model

    model = create_model(ModelConfig(), learning_rate=1e-3)
    assert tuple(model.input.shape) == (None, 48, 128, 1)
    # Four 2x2 pools turn 128x48 into 8x3, still 64 filters.
    assert int(model.get_layer("flatten").output.shape[-1]) == 8 * 3 * 64
    assert model.get_layer("dense1").units == 1024
    for index in range(1, 5):
        assert model.get_layer(f"conv{index}").filters == 64
        assert tuple(model.get_layer(f"pool{index}").pool_size) == (2, 2)
    for index in range(5):
        head = model.get_layer(f"char_{index}_output")
        assert head.units == 62
        assert head.activation.__name__ == "softmax"


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
