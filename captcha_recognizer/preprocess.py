"""The only image preprocess path used by training and prediction.

Grayscale, resize to the model width and height (128×48 by default),
then scale pixels from ``[0, 255]`` to ``[-0.5, 0.5]``.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .config import ModelConfig


def preprocess_image(image, model_config=None) -> np.ndarray:
    """Return a float array of shape ``(height, width, 1)``."""
    config = model_config or ModelConfig()
    resized = _as_pil(image).convert("L").resize(
        (config.image_width, config.image_height),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(resized, dtype=np.float32)
    array = (array / 255.0) - 0.5
    return array[:, :, np.newaxis]


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.copy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)
