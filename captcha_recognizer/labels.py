"""Filename labels and one-hot targets for the five character heads."""

from pathlib import Path

import numpy as np


def label_from_filename(name: str) -> str:
    """Return the label prefix before the first underscore.

    ``1ab2s_num286.jpg`` -> ``1ab2s``. ``CMQVA_num717_1.png`` -> ``CMQVA``.
    """
    return Path(name).stem.split("_", 1)[0]


def one_hot_heads(label: str, char_set: str, chars_num: int) -> tuple:
    """One one-hot vector per character, in position order."""
    if len(label) != chars_num:
        raise ValueError(
            f"label {label!r} has {len(label)} characters, expected {chars_num}"
        )
    heads = []
    for char in label:
        try:
            index = char_set.index(char)
        except ValueError as exc:
            raise ValueError(
                f"character {char!r} is not in the captcha alphabet"
            ) from exc
        head = np.zeros(len(char_set), dtype=np.float32)
        head[index] = 1.0
        heads.append(head)
    return tuple(heads)
