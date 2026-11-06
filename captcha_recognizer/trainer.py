"""Train the five-head captcha model for a finite number of steps per epoch."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from tensorflow import keras

from .config import Config, ModelConfig
from .data_loader import folder_dataset
from .model import create_model

logger = logging.getLogger(__name__)


def train(
    data_dir: str,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    valid_dir: Optional[str] = None,
    steps_per_epoch: Optional[int] = None,
) -> keras.Model:
    """Fit on a folder of images and return the trained model.

    The training dataset is repeated so more than one epoch is possible.
    ``steps_per_epoch`` is always finite, so an epoch ends. It defaults to
    one full pass over the folder, including a partial last batch.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    model_config = ModelConfig()
    model = create_model(model_config, learning_rate=learning_rate)
    train_dataset, counted_steps = folder_dataset(
        data_dir,
        model_config,
        batch_size=batch_size,
        shuffle=True,
        repeat=True,
    )
    if steps_per_epoch is None:
        steps_per_epoch = counted_steps
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be at least 1")

    fit_kwargs = {
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "verbose": 1,
    }
    if valid_dir and Path(valid_dir).is_dir():
        try:
            valid_dataset, valid_steps = folder_dataset(
                valid_dir,
                model_config,
                batch_size=batch_size,
                shuffle=False,
                repeat=True,
            )
        except ValueError:
            valid_dataset = None
        else:
            fit_kwargs["validation_data"] = valid_dataset
            fit_kwargs["validation_steps"] = valid_steps

    logger.info(
        "Training for %s epoch(s), %s step(s) per epoch",
        epochs,
        steps_per_epoch,
    )
    model.fit(train_dataset, **fit_kwargs)
    return model


class CaptchaTrainer:
    """Small wrapper so existing imports can train from a Config."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def train(self) -> keras.Model:
        valid_dir = self.config.data.valid_dir
        self.model = train(
            self.config.data.train_dir,
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            valid_dir=valid_dir if Path(valid_dir).is_dir() else None,
        )
        return self.model


def main(argv=None) -> int:
    """Train from a folder of labeled images."""
    parser = argparse.ArgumentParser(description="Train the captcha recognizer")
    parser.add_argument("--data-dir", default="./data/train_data")
    parser.add_argument("--valid-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Batches per epoch. Defaults to one pass over the folder.",
    )
    args = parser.parse_args(argv)
    try:
        train(
            args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            valid_dir=args.valid_dir,
            steps_per_epoch=args.steps_per_epoch,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        return 1
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
