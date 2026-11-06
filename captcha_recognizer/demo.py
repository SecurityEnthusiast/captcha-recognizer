"""Short train-and-score demo for captchas this repo generates."""

import argparse
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .data_loader import list_image_paths
from .generate import generate
from .labels import label_from_filename
from .preprocess import preprocess_image
from .trainer import train


def predict_image(model, image, char_set):
    """Return text and the mean of the five chosen class probabilities.

    Dropout stays off: the model is called with ``training=False``.
    """
    batch = np.expand_dims(preprocess_image(image), axis=0)
    outputs = model(batch, training=False)
    probabilities = [np.asarray(head)[0] for head in outputs]
    indexes = [int(np.argmax(probs)) for probs in probabilities]
    text = "".join(char_set[index] for index in indexes)
    confidence = float(
        np.mean([float(probs[index]) for probs, index in zip(probabilities, indexes)])
    )
    return text, confidence


def holdout_exact_match(model, holdout_dir, char_set):
    """Count holdout files whose full 5-character label matches."""
    correct = 0
    paths = list_image_paths(holdout_dir)
    for path in paths:
        text, _confidence = predict_image(model, path, char_set)
        if text == label_from_filename(path.name):
            correct += 1
    return correct, len(paths)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ImageCaptcha samples, train for a few epochs, "
            "and print holdout exact-match accuracy."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Training images to generate (default: 200).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs (default: 2).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default="data/demo")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.epochs < 1:
        parser.error("--samples and --epochs must be at least 1")

    work = Path(args.work_dir)
    train_dir = work / "train"
    holdout_dir = work / "holdout"
    holdout_count = max(1, args.samples // 5)
    generate(train_dir, args.samples, seed=args.seed)
    generate(holdout_dir, holdout_count, seed=args.seed + 1)

    model = train(
        str(train_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    config = ModelConfig()
    correct, total = holdout_exact_match(model, holdout_dir, config.char_set)
    accuracy = correct / total if total else 0.0
    print(f"holdout exact-match accuracy: {accuracy:.3f} ({correct}/{total})")
    return 0


def console_main():
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
