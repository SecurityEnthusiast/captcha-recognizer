"""Load captcha images from a folder.

The label is the filename prefix before the first underscore.
Each sample's target is one one-hot vector per character head.
"""

import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import ModelConfig
from .preprocess import PIXEL_SCALE, PIXEL_SHIFT

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def label_from_filename(filename: str) -> str:
    """Return the captcha text stored in a filename."""
    prefix = Path(filename).name.split("_", 1)[0]
    return Path(prefix).stem


def indexes_for_label(label: str, char_set: str, chars_num: int) -> list:
    """Map a label to one class index per character position."""
    if len(label) != chars_num:
        raise ValueError(f"label {label!r} must be {chars_num} characters")
    indexes = []
    for char in label:
        try:
            indexes.append(char_set.index(char))
        except ValueError as exc:
            raise ValueError(f"{char!r} is not in the character set") from exc
    return indexes


def list_image_files(image_dir) -> list:
    """Return image files in a directory, sorted by name."""
    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files)


def epoch_steps(sample_count: int, batch_size: int) -> int:
    """How many batches make one pass over a finite set of samples."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return math.ceil(sample_count / batch_size)


def folder_dataset(image_dir, model_config: ModelConfig, batch_size: int, shuffle: bool, seed: int | None = None):
    """Build a finite batched dataset of (image, named one-hot heads).

    Returns the dataset and the number of images. The dataset is not repeated.
    """
    files = list_image_files(image_dir)
    if not files:
        raise ValueError(f"no images found in {image_dir}")

    paths = []
    indexes = []
    for file in files:
        label = label_from_filename(file.name)
        paths.append(str(file))
        indexes.append(
            indexes_for_label(label, model_config.char_set, model_config.chars_num)
        )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, np.asarray(indexes, dtype=np.int32))
    )
    if shuffle:
        dataset = dataset.shuffle(
            len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    height = model_config.image_height
    width = model_config.image_width
    classes_num = model_config.classes_num
    chars_num = model_config.chars_num

    def load(path, index_row):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=1, expand_animations=False)
        image.set_shape([None, None, 1])
        image = tf.cast(image, tf.float32)
        same_size = tf.logical_and(
            tf.equal(tf.shape(image)[0], height),
            tf.equal(tf.shape(image)[1], width),
        )
        image = tf.cond(
            same_size,
            lambda: image,
            lambda: tf.image.resize(image, [height, width]),
        )
        image.set_shape([height, width, 1])
        image = (image / PIXEL_SCALE) - PIXEL_SHIFT
        heads = {
            f"char_{index}_output": tf.one_hot(index_row[index], classes_num)
            for index in range(chars_num)
        }
        return image, heads

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, len(paths)
