"""
Modern data loading and preprocessing for CAPTCHA recognition.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

from .config import DataConfig, ModelConfig
from .labels import label_from_filename, one_hot_heads
from .preprocess import preprocess_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def list_image_paths(image_dir: str) -> List[Path]:
    """Sorted image files in a folder."""
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    paths = [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"no images found in {image_dir}")
    return paths


def epoch_steps(num_samples: int, batch_size: int) -> int:
    """Batches in one pass over ``num_samples``, including a partial last batch."""
    if num_samples < 1:
        raise ValueError("need at least one image")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return (num_samples + batch_size - 1) // batch_size


def folder_dataset(
    image_dir: str,
    model_config: ModelConfig,
    batch_size: int,
    shuffle: bool = True,
    repeat: bool = False,
) -> Tuple[tf.data.Dataset, int]:
    """tf.data pipeline whose label is the filename prefix before ``_``.

    Returns the dataset and the number of batches in one epoch.
    """
    paths = list_image_paths(image_dir)
    images = np.stack([preprocess_image(path, model_config) for path in paths])
    columns: List[List[np.ndarray]] = [[] for _ in range(model_config.chars_num)]
    for path in paths:
        heads = one_hot_heads(
            label_from_filename(path.name),
            model_config.char_set,
            model_config.chars_num,
        )
        for index, head in enumerate(heads):
            columns[index].append(head)
    labels = tuple(np.stack(column) for column in columns)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    return dataset, epoch_steps(len(paths), batch_size)


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
        
        # Decode label into one one-hot vector per character head.
        label = tf.io.decode_raw(parsed_features['label_raw'], tf.uint8)
        label = tf.reshape(label, [self.model_config.chars_num, self.model_config.classes_num])
        label = tf.cast(label, tf.float32)
        heads = tuple(label[i] for i in range(self.model_config.chars_num))

        return image, heads
    
    def create_dataset(self, 
                      file_path: str, 
                      batch_size: int, 
                      shuffle: bool = True,
                      repeat: bool = False) -> tf.data.Dataset:
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
        """Folder of training images. The label is the filename prefix."""
        dataset, self.steps_per_epoch = folder_dataset(
            self.config.train_dir,
            self.model_config,
            batch_size=batch_size,
            shuffle=True,
            repeat=False,
        )
        return dataset

    def get_validation_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Folder of validation images. The label is the filename prefix."""
        dataset, self.validation_steps = folder_dataset(
            self.config.valid_dir,
            self.model_config,
            batch_size=batch_size,
            shuffle=False,
            repeat=False,
        )
        return dataset
    
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
        
        image_files = list_image_paths(image_dir)
        logger.info(f"Found {len(image_files)} test images")

        images = []
        filenames = []

        for image_file in image_files:
            try:
                images.append(preprocess_image(image_file, self.model_config))
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
            (num_samples, self.model_config.chars_num),
        )

        # One one-hot vector per character, matching the five softmax heads.
        dummy_labels_onehot = tf.one_hot(dummy_labels, self.model_config.classes_num)
        heads = tuple(
            dummy_labels_onehot[:, index, :]
            for index in range(self.model_config.chars_num)
        )

        dataset = tf.data.Dataset.from_tensor_slices((dummy_images, heads))
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
