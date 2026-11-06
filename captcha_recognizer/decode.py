"""Turn the five softmax heads into one string and one confidence."""

import numpy as np

from .config import Config
from .preprocess import load_image_batch


def probability_row(head) -> np.ndarray:
    """Return the class probabilities for one sample.

    model(batch) gives each head the shape (batch, classes). A batch of one
    is (1, classes). Argmax has to run on the class axis, not the flat index
    of that two-dimensional array.
    """
    if hasattr(head, "numpy"):
        row = np.asarray(head.numpy())
    else:
        row = np.asarray(head)
    if row.ndim == 2:
        if row.shape[0] != 1:
            raise ValueError("decode one sample at a time")
        row = row[0]
    if row.ndim != 1:
        raise ValueError(f"expected a class vector, got shape {row.shape}")
    return row


def ordered_heads(predictions, chars_num: int) -> list:
    """Character heads from index 0 to chars_num - 1."""
    if isinstance(predictions, dict):
        return [predictions[f"char_{index}_output"] for index in range(chars_num)]
    return list(predictions)


def decode_heads(heads, char_set: str) -> tuple:
    """Return (text, confidence) for one sample."""
    characters = []
    confidences = []
    for head in heads:
        row = probability_row(head)
        index = int(np.argmax(row))
        characters.append(char_set[index])
        confidences.append(float(row[index]))
    return "".join(characters), float(np.mean(confidences))


def predict(image, model, config: Config = None) -> tuple:
    """Read one image and return (text, confidence). Dropout is off."""
    config = config or Config()
    if isinstance(image, np.ndarray):
        batch = image if image.ndim == 4 else image[np.newaxis, ...]
    else:
        batch = load_image_batch(
            image,
            config.model.image_width,
            config.model.image_height,
        )
    outputs = model(batch, training=False)
    heads = ordered_heads(outputs, config.model.chars_num)
    return decode_heads(heads, config.model.char_set)
