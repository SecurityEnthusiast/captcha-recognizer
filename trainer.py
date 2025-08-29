"""
Modern training module for CAPTCHA recognition model.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import argparse
import sys
from datetime import datetime
import json

from .config import Config, TrainingConfig
from .model import create_model, save_model
from .data_loader import CaptchaDataLoader

logger = logging.getLogger(__name__)


class CaptchaTrainer:
    """
    Trainer class for CAPTCHA recognition model.
    
    This class handles the training process including model compilation,
    training, validation, and checkpointing.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = CaptchaDataLoader(config.data, config.model)
        self.model = None
        self.history = None
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        self._create_directories()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _create_directories(self):
        """Create necessary directories for training outputs."""
        Path(self.config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.training.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def prepare_model(self):
        """Prepare the model for training."""
        logger.info("Creating model...")
        self.model = create_model(self.config.model)
        
        # Print model summary
        self.model.summary()
        
        # Save model architecture
        model_arch_path = Path(self.config.training.checkpoint_dir) / "model_architecture.json"
        with open(model_arch_path, 'w') as f:
            json.dump(self.model.to_json(), f, indent=2)
        
        logger.info(f"Model architecture saved to {model_arch_path}")
    
    def prepare_data(self):
        """Prepare training and validation datasets."""
        logger.info("Preparing datasets...")
        
        try:
            self.train_dataset = self.data_loader.get_training_dataset(
                self.config.training.batch_size
            )
            self.val_dataset = self.data_loader.get_validation_dataset(
                self.config.training.batch_size
            )
            
            logger.info("Datasets prepared successfully")
            
        except Exception as e:
            logger.error(f"Failed to prepare datasets: {e}")
            raise
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=Path(self.config.training.checkpoint_dir) / "model_{epoch:02d}_{val_loss:.4f}.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # TensorBoard logging
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=Path(self.config.training.checkpoint_dir) / "logs",
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def train(self):
        """Execute the training process."""
        logger.info("Starting training...")
        
        # Prepare model and data
        self.prepare_model()
        self.prepare_data()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Training
        logger.info(f"Training for {self.config.training.epochs} epochs...")
        
        try:
            self.history = self.model.fit(
                self.train_dataset,
                epochs=self.config.training.epochs,
                validation_data=self.val_dataset,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed successfully!")
            
            # Save final model
            save_model(self.model, self.config.training.model_save_path)
            
            # Save training history
            history_path = Path(self.config.training.checkpoint_dir) / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f, indent=2)
            
            logger.info(f"Training history saved to {history_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, test_dataset: Optional[tf.data.Dataset] = None):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info("Evaluating model...")
        
        if test_dataset is None:
            test_dataset = self.val_dataset
        
        # Evaluate on test data
        evaluation_results = self.model.evaluate(test_dataset, verbose=1)
        
        # Log results
        metrics_names = self.model.metrics_names
        for name, value in zip(metrics_names, evaluation_results):
            logger.info(f"{name}: {value:.4f}")
        
        return dict(zip(metrics_names, evaluation_results))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CAPTCHA recognition model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory for saving checkpoints'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config and Path(args.config).exists():
        # Load from file (implement config file loading if needed)
        config = Config()
    else:
        # Use command line arguments
        config = Config(
            training=TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                checkpoint_dir=args.checkpoint_dir
            )
        )
    
    # Create trainer and start training
    trainer = CaptchaTrainer(config)
    
    try:
        trainer.train()
        
        # Evaluate the model
        evaluation_results = trainer.evaluate()
        logger.info("Training and evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

