"""Train the captcha CNN on a folder of labeled images."""

import argparse
from pathlib import Path

from .config import Config, TrainingConfig
from .data_loader import epoch_steps, folder_dataset, list_image_files
from .model import create_model, save_model


def fit_model(
    model,
    train_dataset,
    train_count,
    batch_size,
    epochs,
    valid_dataset=None,
    valid_count=None,
    callbacks=None,
    verbose=1,
):
    """Fit for a fixed number of epochs.

    The training set is repeated, and steps_per_epoch is the number of
    batches in one pass. Without that step count, a repeated dataset never
    ends an epoch.
    """
    kwargs = {
        "epochs": epochs,
        "steps_per_epoch": epoch_steps(train_count, batch_size),
        "verbose": verbose,
    }
    if callbacks:
        kwargs["callbacks"] = callbacks
    if valid_dataset is not None:
        kwargs["validation_data"] = valid_dataset.repeat()
        kwargs["validation_steps"] = epoch_steps(valid_count, batch_size)
    return model.fit(train_dataset.repeat(), **kwargs)


class CaptchaTrainer:
    """Train, checkpoint, and save one model."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.history = None

    def prepare_model(self):
        self.model = create_model(
            self.config.model,
            learning_rate=self.config.training.learning_rate,
        )

    def train(self, train_dir=None, valid_dir=None):
        """Train on train_dir and, when that folder has images, validate."""
        train_dir = Path(train_dir or self.config.data.train_dir)
        valid_dir = Path(valid_dir or self.config.data.valid_dir)
        self.prepare_model()

        train_dataset, train_count = folder_dataset(
            train_dir,
            self.config.model,
            batch_size=self.config.training.batch_size,
            shuffle=True,
        )
        valid_dataset = None
        valid_count = None
        callbacks = []
        if valid_dir.is_dir() and list_image_files(valid_dir):
            valid_dataset, valid_count = folder_dataset(
                valid_dir,
                self.config.model,
                batch_size=self.config.training.batch_size,
                shuffle=False,
            )
            callbacks = self._callbacks()

        self.history = fit_model(
            self.model,
            train_dataset,
            train_count,
            self.config.training.batch_size,
            self.config.training.epochs,
            valid_dataset=valid_dataset,
            valid_count=valid_count,
            callbacks=callbacks,
        )
        destination = Path(self.config.training.model_save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_model(self.model, str(destination))
        return self.history

    def _callbacks(self):
        from tensorflow import keras

        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return [
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / "model_{epoch:02d}.keras"),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=0,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            ),
        ]


def train(train_dir, valid_dir=None, config: Config = None):
    """Train with Config defaults unless the caller passes its own."""
    trainer = CaptchaTrainer(config or Config())
    return trainer.train(train_dir, valid_dir)


def main():
    parser = argparse.ArgumentParser(description="Train on a folder of captcha images")
    parser.add_argument("--train-dir", default="./data/train_data")
    parser.add_argument("--valid-dir", default="./data/valid_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--model-path", default="./models/captcha_model.keras")
    args = parser.parse_args()
    config = Config(
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            model_save_path=args.model_path,
        )
    )
    train(args.train_dir, args.valid_dir, config)


if __name__ == "__main__":
    main()
