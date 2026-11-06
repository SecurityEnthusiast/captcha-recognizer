"""Train on captchas this process just drew, then score a holdout.

Exit status is 0 only when training loss fell and holdout per-character
accuracy beat chance. The checked-in test images are printed, not scored.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import CHAR_SET, Config, DataConfig, ModelConfig, TrainingConfig
from .data_loader import folder_dataset, label_from_filename, list_image_files
from .decode import predict
from .generate import generate
from .metrics import character_accuracy, confusion_counts, exact_matches
from .model import create_model
from .trainer import fit_model

DIGITS = "0123456789"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def split_files(files, holdout_ratio: float, seed: int):
    """Return (train_files, holdout_files). Holdout files are not in train."""
    if len(files) < 2:
        raise ValueError("need at least two images to hold one out")
    ordered = list(files)
    random.Random(seed).shuffle(ordered)
    holdout_count = int(round(len(ordered) * holdout_ratio))
    holdout_count = min(max(holdout_count, 1), len(ordered) - 1)
    return ordered[holdout_count:], ordered[:holdout_count]


def learning_bar_passed(first_loss, last_loss, per_character, charset_size: int) -> bool:
    """Loss must fall, and per-character accuracy must beat 1/charset."""
    if charset_size < 1:
        raise ValueError("charset_size must be positive")
    chance = 1.0 / charset_size
    return last_loss < first_loss and per_character > chance



def format_run_settings(samples: int, epochs: int, dropout: float, charset_size: int) -> str:
    """Name the run that produced the holdout score."""
    return (
        f"samples: {samples} epochs: {epochs} "
        f"dropout: {dropout:.1f} charset: {charset_size}"
    )


def _clear_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def run_demo(
    *,
    samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
    holdout_ratio: float,
    learning_rate: float,
    output_dir,
    test_dir,
    predictions_path,
    char_set: str,
    chars_num: int = 5,
    width: int = 128,
    height: int = 48,
    font_size: int = 28,
    dropout: float = 0.5,
) -> int:
    """Generate, train, print scores, and return a process status code."""
    set_seeds(seed)
    output_dir = Path(output_dir)
    generated_dir = output_dir / "generated"
    train_dir = output_dir / "train"
    holdout_dir = output_dir / "holdout"
    for directory in (generated_dir, train_dir, holdout_dir):
        _clear_files(directory)

    generated = generate(
        generated_dir,
        samples,
        chars_num=chars_num,
        char_set=char_set,
        seed=seed,
        width=width,
        height=height,
        font_size=font_size,
    )
    train_files, holdout_files = split_files(generated, holdout_ratio, seed)
    for path in train_files:
        path.rename(train_dir / path.name)
    for path in holdout_files:
        path.rename(holdout_dir / path.name)

    model_config = ModelConfig(
        image_height=height,
        image_width=width,
        char_set=char_set,
        chars_num=chars_num,
    )
    config = Config(
        model=model_config,
        training=TrainingConfig(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
        ),
        data=DataConfig(test_data_dir=str(test_dir)),
    )
    model = create_model(model_config, learning_rate=learning_rate, dropout=dropout)
    train_dataset, train_count = folder_dataset(
        train_dir,
        model_config,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    history = fit_model(
        model,
        train_dataset,
        train_count,
        batch_size,
        epochs,
        verbose=1,
    )
    losses = history.history["loss"]
    first_loss = float(losses[0])
    last_loss = float(losses[-1])

    pairs = []
    for path in list_image_files(holdout_dir):
        text, _confidence = predict(path, model, config)
        pairs.append((text, label_from_filename(path.name)))
    exact = exact_matches(pairs)
    per_character = character_accuracy(pairs)
    passed = learning_bar_passed(first_loss, last_loss, per_character, len(char_set))

    print(format_run_settings(samples, epochs, dropout, len(char_set)))
    print(f"holdout exact match: {exact:.4f}")
    print(f"holdout per-character: {per_character:.4f}")
    print(f"chance per-character: {1.0 / len(char_set):.4f}")
    print(f"first epoch loss: {first_loss:.4f}")
    print(f"last epoch loss: {last_loss:.4f}")
    print("learning bar: passed" if passed else "learning bar: failed")
    mistakes = confusion_counts(pairs)
    if mistakes:
        rendered = ", ".join(f"{truth}->{got} {count}" for truth, got, count in mistakes)
        print(f"holdout confusions: {rendered}")
    else:
        print("holdout confusions: none")

    rows = []
    test_path = Path(test_dir)
    if test_path.is_dir():
        for path in list_image_files(test_path):
            text, confidence = predict(path, model, config)
            print(f"{path.name} -> {text!r} (confidence {confidence:.4f})")
            rows.append((path.name, text, f"{confidence:.4f}"))

    predictions = Path(predictions_path)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "predicted_text", "confidence"))
        writer.writerows(rows)
    return 0 if passed else 1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate captchas, train on CPU, and score a holdout."
    )
    parser.add_argument("--samples", type=int, default=400, help="images to draw (default 400)")
    parser.add_argument("--epochs", type=int, default=15, help="training epochs (default 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size (default 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="./data/demo")
    parser.add_argument("--test-dir", default="./data/test_data")
    parser.add_argument("--predictions", default="./predictions.csv")
    parser.add_argument(
        "--digits-only",
        action="store_true",
        help="use 0-9 instead of the 62-character set",
    )
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="dropout rate (default 0.5). 0 memorizes the training images and the holdout gets worse",
    )
    args = parser.parse_args(argv)
    status = run_demo(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        holdout_ratio=args.holdout,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        test_dir=args.test_dir,
        predictions_path=args.predictions,
        char_set=DIGITS if args.digits_only else CHAR_SET,
        font_size=args.font_size,
        dropout=args.dropout,
    )
    sys.exit(status)


if __name__ == "__main__":
    main()
