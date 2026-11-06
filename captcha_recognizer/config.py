"""
Configuration settings for the CAPTCHA recognition system.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    image_height: int = 48
    image_width: int = 128
    char_set: str = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chars_num: int = 5
    
    @property
    def classes_num(self) -> int:
        """Number of character classes."""
        return len(self.char_set)
    
    @property
    def input_shape(self) -> tuple:
        """Input image shape."""
        return (self.image_height, self.image_width, 1)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    learning_rate: float = 1e-4
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_dir: str = './checkpoints'
    model_save_path: str = './models/captcha_model.h5'


@dataclass
class DataConfig:
    """Data configuration."""
    record_dir: str = './data'
    train_file: str = 'train.tfrecords'
    valid_file: str = 'valid.tfrecords'
    test_data_dir: str = './data/test_data'
    
    @property
    def train_path(self) -> Path:
        """Full path to training data."""
        return Path(self.record_dir) / self.train_file
    
    @property
    def valid_path(self) -> Path:
        """Full path to validation data."""
        return Path(self.record_dir) / self.valid_file


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            model=ModelConfig(
                image_height=int(os.getenv('CAPTCHA_IMAGE_HEIGHT', 48)),
                image_width=int(os.getenv('CAPTCHA_IMAGE_WIDTH', 128)),
                chars_num=int(os.getenv('CAPTCHA_CHARS_NUM', 5))
            ),
            training=TrainingConfig(
                batch_size=int(os.getenv('CAPTCHA_BATCH_SIZE', 128)),
                learning_rate=float(os.getenv('CAPTCHA_LEARNING_RATE', 1e-4)),
                epochs=int(os.getenv('CAPTCHA_EPOCHS', 100))
            ),
            data=DataConfig(
                record_dir=os.getenv('CAPTCHA_RECORD_DIR', './data'),
                test_data_dir=os.getenv('CAPTCHA_TEST_DIR', './data/test_data')
            )
        )


# Backward compatibility
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 128
CHAR_SET = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CLASSES_NUM = len(CHAR_SET)
CHARS_NUM = 5
RECORD_DIR = './data'
TRAIN_FILE = 'train.tfrecords'
VALID_FILE = 'valid.tfrecords'

    
