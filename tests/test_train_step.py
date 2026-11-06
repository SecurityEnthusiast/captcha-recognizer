"""Import the package and run one fit step on four generated images.

Run with: python tests/test_train_step.py
"""

import tempfile
from pathlib import Path

from captcha.image import ImageCaptcha

import captcha_recognizer
from captcha_recognizer import ModelConfig, train
from captcha_recognizer.labels import label_from_filename


def main() -> int:
    assert captcha_recognizer.__version__
    config = ModelConfig()
    labels = ["ab1CD", "zz099", "QwErT", "12abZ"]
    captcha = ImageCaptcha(
        width=config.image_width,
        height=config.image_height,
        font_sizes=[40],
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index, label in enumerate(labels):
            path = root / f"{label}_num{index}.png"
            captcha.write(label, str(path))
            if label_from_filename(path.name) != label:
                raise AssertionError(path.name)
        model = train(
            str(root),
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            steps_per_epoch=1,
        )
        if model is None or len(model.outputs) != config.chars_num:
            raise AssertionError("expected five character heads")
    print("one training step completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
