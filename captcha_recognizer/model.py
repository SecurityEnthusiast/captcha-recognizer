"""Five-head CNN for a fixed-length captcha string."""

import logging

from tensorflow import keras
from tensorflow.keras import layers

from .config import ModelConfig

logger = logging.getLogger(__name__)


def create_model(config: ModelConfig, learning_rate: float = 1e-4) -> keras.Model:
    """Build the 4-block CNN and compile it.

    Each character position is its own softmax, named char_0_output through
    char_{n-1}_output. The learning rate comes from the caller, which is
    TrainingConfig during training.
    """
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
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{index}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(1024, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    outputs = {
        f"char_{index}_output": layers.Dense(
            config.classes_num,
            activation="softmax",
            name=f"char_{index}_output",
        )(x)
        for index in range(config.chars_num)
    }
    model = keras.Model(inputs=inputs, outputs=outputs, name="captcha_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
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
