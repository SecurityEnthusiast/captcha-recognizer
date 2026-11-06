"""Five-head CNN for a fixed-length captcha string.

Width is pooled only twice, so the columns stay in left-to-right order.
Each character is a softmax on its own slice of those columns, and the
softmax weights are shared across the slices.
"""

import logging

from tensorflow import keras
from tensorflow.keras import layers

from .config import ModelConfig

logger = logging.getLogger(__name__)


def character_windows(map_width: int, chars_num: int) -> list[tuple[int, int]]:
    """Return (start, length) for each character along the width axis.

    The windows cover every column. Neighbors overlap when the width is
    not a multiple of the character count.
    """
    if chars_num < 1:
        raise ValueError("chars_num must be positive")
    if map_width < chars_num:
        raise ValueError(
            f"feature width {map_width} is shorter than {chars_num} characters"
        )
    stride = map_width // chars_num
    length = map_width - (chars_num - 1) * stride
    return [(index * stride, length) for index in range(chars_num)]


def create_model(config: ModelConfig, learning_rate: float = 1e-4, dropout: float = 0.5) -> keras.Model:
    """Build the 4-block CNN and compile it.

    Each character position is its own softmax, named char_0_output through
    char_{n-1}_output. The learning rate comes from the caller, which is
    TrainingConfig during training.
    """
    if config.image_height % 16 or config.image_width % 4:
        raise ValueError("image height must be divisible by 16 and width by 4")

    inputs = layers.Input(shape=config.input_shape, name="image")
    x = inputs
    for index in range(1, 5):
        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name=f"conv{index}",
        )(x)
        # The last two pools shrink height only, so the characters stay side by side.
        pool_size = (2, 2) if index <= 2 else (2, 1)
        x = layers.MaxPooling2D(pool_size=pool_size, name=f"pool{index}")(x)

    map_height = config.image_height // 16
    map_width = config.image_width // 4
    x = layers.AveragePooling2D(pool_size=(map_height, 1), name="collapse_height")(x)
    x = layers.Reshape((map_width, 64), name="columns")(x)
    if dropout:
        x = layers.Dropout(dropout, name="dropout")(x)

    classifier = layers.Dense(
        config.classes_num,
        activation="softmax",
        name="char_classifier",
    )
    outputs = {}
    for index, (start, length) in enumerate(character_windows(map_width, config.chars_num)):
        right = map_width - (start + length)
        slot = layers.Cropping1D(
            cropping=(start, right),
            name=f"char_{index}_span",
        )(x)
        flat = layers.Flatten(name=f"char_{index}_flat")(slot)
        outputs[f"char_{index}_output"] = layers.Identity(name=f"char_{index}_output")(
            classifier(flat)
        )

    model = keras.Model(inputs=inputs, outputs=outputs, name="captcha_model")
    per_head = {
        name: "categorical_crossentropy" for name in outputs
    }
    per_head_accuracy = {name: "accuracy" for name in outputs}
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=per_head,
        metrics=per_head_accuracy,
    )
    logger.info("Created model with %s parameters", model.count_params())
    return model


def load_model(model_path: str) -> keras.Model:
    """Load a model saved by save_model."""
    model = keras.models.load_model(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


def save_model(model: keras.Model, model_path: str) -> None:
    """Save a trained model."""
    model.save(model_path)
    logger.info("Model saved to %s", model_path)
