"""
Modern CAPTCHA recognition model using TensorFlow 2.x and Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class CaptchaModel(keras.Model):
    """
    Modern CAPTCHA recognition model using CNN architecture.
    
    This model is designed to recognize CAPTCHA images with multiple characters.
    It uses a convolutional neural network with multiple convolutional layers
    followed by fully connected layers for character classification.
    """
    
    def __init__(self, config: ModelConfig, name: str = "captcha_model"):
        super().__init__(name=name)
        self.config = config
        self._build_model()
    
    def _build_model(self):
        """Build the model architecture."""
        # Input layer
        self.input_layer = layers.Input(shape=self.config.input_shape)
        
        # Convolutional layers
        self.conv1 = layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same',
            name='conv1'
        )
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')
        
        self.conv2 = layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same',
            name='conv2'
        )
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')
        
        self.conv3 = layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same',
            name='conv3'
        )
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')
        
        self.conv4 = layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same',
            name='conv4'
        )
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), name='pool4')
        
        # Flatten and dense layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu', name='dense1')
        self.dropout = layers.Dropout(0.5)
        
        # Output layer - one dense layer per character position
        self.output_layers = []
        for i in range(self.config.chars_num):
            output_layer = layers.Dense(
                self.config.classes_num, 
                activation='softmax',
                name=f'char_{i}_output'
            )
            self.output_layers.append(output_layer)
    
    def call(self, inputs, training=None):
        """Forward pass through the model."""
        x = inputs
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        
        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        
        # Output layers for each character
        outputs = []
        for output_layer in self.output_layers:
            char_output = output_layer(x)
            outputs.append(char_output)
        
        return outputs
    
    def build_model(self) -> keras.Model:
        """Build and return a functional model."""
        inputs = self.input_layer
        outputs = self.call(inputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


def create_model(config: ModelConfig) -> keras.Model:
    """
    Factory function to create a CAPTCHA recognition model.
    
    Args:
        config: Model configuration
        
    Returns:
        Compiled Keras model
    """
    model_instance = CaptchaModel(config)
    model = model_instance.build_model()
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Created model with {model.count_params()} parameters")
    return model


def load_model(model_path: str, config: ModelConfig) -> keras.Model:
    """
    Load a pre-trained model from disk.
    
    Args:
        model_path: Path to the saved model
        config: Model configuration
        
    Returns:
        Loaded Keras model
    """
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def save_model(model: keras.Model, model_path: str):
    """
    Save a trained model to disk.
    
    Args:
        model: Keras model to save
        model_path: Path where to save the model
    """
    try:
        model.save(model_path)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        raise
