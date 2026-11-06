"""Draw captchas at the model size.

Characters are sampled with replacement, so a label can repeat a glyph.
The length argument is the number of characters actually drawn.
"""

import random
from pathlib import Path

from captcha.image import ImageCaptcha


def random_label(rng: random.Random, char_set: str, chars_num: int) -> str:
    """Pick chars_num characters, allowing repeats."""
    if chars_num < 1:
        raise ValueError("chars_num must be positive")
    if not char_set:
        raise ValueError("char_set must not be empty")
    return "".join(rng.choices(char_set, k=chars_num))


def generate(
    output_dir,
    count,
    *,
    chars_num,
    char_set,
    seed,
    width,
    height,
    font_size=28,
):
    """Write count images named {label}_num{i}.png.

    font_size defaults below the old size of 40 because the image is only
    48 pixels tall.
    """
    if count < 1:
        raise ValueError("count must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # ImageCaptcha uses the global random module for its lines and noise.
    random.seed(seed)
    drawer = ImageCaptcha(width=width, height=height, font_sizes=(font_size,))
    paths = []
    for index in range(count):
        label = random_label(rng, char_set, chars_num)
        path = destination / f"{label}_num{index}.png"
        drawer.write(label, str(path))
        paths.append(path)
    return paths
