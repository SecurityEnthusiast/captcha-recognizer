"""The package imports, and one training step finishes."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_package_imports():
    import captcha_recognizer

    assert captcha_recognizer.__version__ == "3.0.0"


def test_label_from_filename():
    from captcha_recognizer.data_loader import label_from_filename

    assert label_from_filename("1ab2s_num286.jpg") == "1ab2s"
    assert label_from_filename("CMQVA_num717_1.png") == "CMQVA"


def test_epoch_steps_covers_every_sample():
    from captcha_recognizer.data_loader import epoch_steps

    assert epoch_steps(4, 128) == 1
    assert epoch_steps(200, 128) == 2
    assert epoch_steps(32, 32) == 1


def test_learning_rate_comes_from_the_caller():
    from captcha_recognizer import ModelConfig, create_model

    model = create_model(ModelConfig(), learning_rate=0.0123)
    learning_rate = model.optimizer.learning_rate
    value = float(learning_rate.numpy()) if hasattr(learning_rate, "numpy") else float(learning_rate)
    assert value == pytest.approx(0.0123)
    assert model.output_names == [
        "char_0_output",
        "char_1_output",
        "char_2_output",
        "char_3_output",
        "char_4_output",
    ]


def _write_image(directory: Path, label: str, index: int) -> None:
    array = np.random.default_rng(index).integers(0, 256, size=(48, 128), dtype=np.uint8)
    Image.fromarray(array, mode="L").save(directory / f"{label}_num{index}.png")


def test_fit_one_step_on_four_images(tmp_path):
    from captcha_recognizer import ModelConfig, create_model
    from captcha_recognizer.data_loader import folder_dataset
    from captcha_recognizer.trainer import fit_model

    for index, label in enumerate(("aaaaa", "bbbbb", "ccccc", "ddddd")):
        _write_image(tmp_path, label, index)

    config = ModelConfig()
    model = create_model(config, learning_rate=1e-3)
    dataset, count = folder_dataset(tmp_path, config, batch_size=4, shuffle=False)
    history = fit_model(model, dataset, count, batch_size=4, epochs=1, verbose=0)

    assert count == 4
    assert len(history.history["loss"]) == 1
