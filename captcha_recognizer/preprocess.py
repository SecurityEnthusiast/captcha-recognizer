"""One image scale for training and prediction."""

from pathlib import Path

import numpy as np
from PIL import Image

PIXEL_SCALE = 255.0
PIXEL_SHIFT = 0.5


def scale_pixels(array):
    """Map 0–255 pixels into [-0.5, 0.5]."""
    return (array / PIXEL_SCALE) - PIXEL_SHIFT


def preprocess_image(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Return one grayscale image as float32 shaped (height, width, 1)."""
    gray = image.convert("L")
    if gray.size != (width, height):
        gray = gray.resize((width, height))
    array = np.asarray(gray, dtype=np.float32)
    array = scale_pixels(array)
    return array[:, :, np.newaxis]


def load_image_batch(path, width: int, height: int) -> np.ndarray:
    """Load one image as a batch of size 1."""
    with Image.open(Path(path)) as image:
        array = preprocess_image(image, width, height)
    return array[np.newaxis, ...]
