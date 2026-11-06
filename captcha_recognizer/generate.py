"""Draw ImageCaptcha samples the folder loader can train on."""

import random
from pathlib import Path

from .config import ModelConfig

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def generate(output_dir, count, *, seed=0, model_config=None):
    """Write ``count`` images. The filename prefix is the text in the image.

    The random seed is fixed by default so a demo run can be repeated.
    Characters are drawn with replacement, so a label may repeat a glyph.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    config = model_config or ModelConfig()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for old in root.iterdir():
        if old.is_file() and old.suffix.lower() in _IMAGE_SUFFIXES:
            old.unlink()

    # Imported here so the rest of the package can load without the generator.
    from captcha.image import ImageCaptcha

    rng = random.Random(seed)
    captcha = ImageCaptcha(
        width=config.image_width,
        height=config.image_height,
        font_sizes=[40],
    )
    paths = []
    for index in range(count):
        label = "".join(rng.choice(config.char_set) for _ in range(config.chars_num))
        path = root / f"{label}_num{index}.png"
        captcha.write(label, str(path))
        paths.append(path)
    return paths
