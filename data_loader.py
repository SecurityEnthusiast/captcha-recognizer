"""
Modern data loading and preprocessing for CAPTCHA recognition.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging
from PIL import Image
import os

from .config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class CaptchaDataLoader:
    """
    Data loader for CAPTCHA images using TensorFlow 2.x data pipelines.
    
    This class handles loading, preprocessing, and batching of CAPTCHA images
    for training and validation.
    """
    
    def __init__(self, config: DataConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that data paths exist."""
        if not Path(self.config.record_dir).exists():
            logger.warning(f"Data directory {self.config.record_dir} does not exist")
        
        if not Path(self.config.test_data_dir).exists():
            logger.warning(f"Test data directory {self.config.test_data_dir} does not exist")
    
    def _parse_tfrecord(self, example_proto):
        """Parse a single TFRecord example."""
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label_raw': tf.io.FixedLenFeature([], tf.string),
        }
        
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode image
        image = tf.io.decode_raw(parsed_features['image_raw'], tf.int16)
        image = tf.reshape(image, [self.model_config.image_height, self.model_config.image_width])
        image = tf.cast(image, tf.float32) * (1.0 / 255.0) - 0.5
        image = tf.expand_dims(image, axis=-1)  # Add channel dimension
        
        # Decode label
        label = tf.io.decode_raw(parsed_features['label_raw'], tf.uint8)
        label = tf.reshape(label, [self.model_config.chars_num, self.model_config.classes_num])
        label = tf.cast(label, tf.float32)
        
        return image, label
    
    def create_dataset(self, 
                      file_path: str, 
                      batch_size: int, 
                      shuffle: bool = True,
                      repeat: bool = True) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from TFRecord files.
        
        Args:
            file_path: Path to TFRecord file
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            repeat: Whether to repeat the dataset
            
        Returns:
            TensorFlow dataset
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"TFRecord file not found: {file_path}")
        
        dataset = tf.data.TFRecordDataset(file_path)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if repeat:
            dataset = dataset.repeat()
        
        return dataset
    
    def get_training_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Get training dataset."""
        return self.create_dataset(
            self.config.train_path,
            batch_size=batch_size,
            shuffle=True,
            repeat=True
        )
    
    def get_validation_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Get validation dataset."""
        return self.create_dataset(
            self.config.valid_path,
            batch_size=batch_size,
            shuffle=False,
            repeat=False
        )
    
    def load_test_images(self, image_dir: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load test images from directory for prediction.
        
        Args:
            image_dir: Directory containing test images (uses config default if None)
            
        Returns:
            Tuple of (images, filenames)
        """
        if image_dir is None:
            image_dir = self.config.test_data_dir
        
        if not Path(image_dir).exists():
            raise FileNotFoundError(f"Test image directory not found: {image_dir}")
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        logger.info(f"Found {len(image_files)} test images")
        
        # Load and preprocess images
        images = []
        filenames = []
        
        for image_file in image_files:
            try:
                # Load image
                image = Image.open(image_file)
                image_gray = image.convert('L')  # Convert to grayscale
                image_resized = image_gray.resize(
                    (self.model_config.image_width, self.model_config.image_height)
                )
                image.close()
                
                # Convert to numpy array and normalize
                image_array = np.array(image_resized, dtype=np.float32)
                image_array = (image_array / 255.0) - 0.5
                image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
                
                images.append(image_array)
                filenames.append(image_file.name)
                
            except Exception as e:
                logger.warning(f"Failed to load image {image_file}: {e}")
                continue
        
        if not images:
            raise ValueError("No images could be loaded successfully")
        
        # Stack images into batch
        images_batch = np.stack(images, axis=0)
        
        return images_batch, filenames
    
    def create_synthetic_dataset(self, 
                               num_samples: int, 
                               save_dir: Optional[str] = None) -> tf.data.Dataset:
        """
        Create a synthetic dataset for testing purposes.
        
        Args:
            num_samples: Number of synthetic samples to create
            save_dir: Directory to save synthetic images (optional)
            
        Returns:
            TensorFlow dataset with synthetic data
        """
        # This is a placeholder for synthetic data generation
        # In a real implementation, you would generate CAPTCHA images here
        
        logger.info(f"Creating synthetic dataset with {num_samples} samples")
        
        # Create dummy data for now
        dummy_images = np.random.randn(
            num_samples, 
            self.model_config.image_height, 
            self.model_config.image_width, 
            1
        ).astype(np.float32)
        
        dummy_labels = np.random.randint(
            0, 
            self.model_config.classes_num, 
            (num_samples, self.model_config.chars_num)
        )
        
        # Convert labels to one-hot encoding
        dummy_labels_onehot = tf.one_hot(dummy_labels, self.model_config.classes_num)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels_onehot))
        dataset = dataset.batch(32)
        
        return dataset


# Backward compatibility functions
def inputs(train: bool, batch_size: int):
    """Backward compatibility function for old code."""
    config = DataConfig()
    model_config = ModelConfig()
    loader = CaptchaDataLoader(config, model_config)
    
    if train:
        return loader.get_training_dataset(batch_size)
    else:
        return loader.get_validation_dataset(batch_size)

