"""
Modern prediction module for CAPTCHA recognition.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import argparse
import sys
from datetime import datetime
import json
from PIL import Image

from .config import Config, ModelConfig
from .model import load_model
from .data_loader import CaptchaDataLoader

logger = logging.getLogger(__name__)


class CaptchaPredictor:
    """
    Predictor class for CAPTCHA recognition.
    
    This class handles loading trained models and making predictions
    on new CAPTCHA images.
    """
    
    def __init__(self, config: Config, model_path: Optional[str] = None):
        self.config = config
        self.data_loader = CaptchaDataLoader(config.data, config.model)
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = load_model(model_path, self.config.model)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = Image.open(image_path)
            image_gray = image.convert('L')  # Convert to grayscale
            image_resized = image_gray.resize(
                (self.config.model.image_width, self.config.model.image_height)
            )
            image.close()
            
            # Convert to numpy array and normalize
            image_array = np.array(image_resized, dtype=np.float32)
            image_array = (image_array / 255.0) - 0.5
            image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
            image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
            
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def predict_single(self, image_path: str) -> Tuple[str, float]:
        """
        Predict CAPTCHA text for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_text, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        image_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        
        # Process predictions
        predicted_text = self._decode_predictions(predictions)
        confidence = self._calculate_confidence(predictions)
        
        return predicted_text, confidence
    
    def predict_batch(self, image_dir: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """
        Predict CAPTCHA text for multiple images.
        
        Args:
            image_dir: Directory containing images (uses config default if None)
            
        Returns:
            List of tuples: (filename, predicted_text, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load test images
        images, filenames = self.data_loader.load_test_images(image_dir)
        
        # Make predictions
        predictions = self.model.predict(images, verbose=1)
        
        # Process results
        results = []
        for i, filename in enumerate(filenames):
            predicted_text = self._decode_predictions([pred[i] for pred in predictions])
            confidence = self._calculate_confidence([pred[i] for pred in predictions])
            results.append((filename, predicted_text, confidence))
        
        return results
    
    def _decode_predictions(self, predictions: List[np.ndarray]) -> str:
        """
        Decode model predictions to text.
        
        Args:
            predictions: List of prediction arrays for each character position
            
        Returns:
            Decoded text string
        """
        decoded_text = ""
        
        for char_pred in predictions:
            # Get the character with highest probability
            char_index = np.argmax(char_pred)
            decoded_text += self.config.model.char_set[char_index]
        
        return decoded_text
    
    def _calculate_confidence(self, predictions: List[np.ndarray]) -> float:
        """
        Calculate confidence score for predictions.
        
        Args:
            predictions: List of prediction arrays for each character position
            
        Returns:
            Average confidence score
        """
        confidences = []
        
        for char_pred in predictions:
            # Get the probability of the predicted character
            char_index = np.argmax(char_pred)
            confidence = char_pred[char_index]
            confidences.append(confidence)
        
        return np.mean(confidences)
    
    def evaluate_accuracy(self, 
                         test_dir: str, 
                         ground_truth: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Evaluate prediction accuracy on test data.
        
        Args:
            test_dir: Directory containing test images
            ground_truth: Dictionary mapping filenames to true labels (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get predictions
        results = self.predict_batch(test_dir)
        
        # Calculate accuracy
        total_count = len(results)
        correct_count = 0
        
        for filename, predicted_text, confidence in results:
            if ground_truth and filename in ground_truth:
                true_text = ground_truth[filename]
                if predicted_text == true_text:
                    correct_count += 1
            else:
                # Try to extract true text from filename
                # This assumes filename contains the true CAPTCHA text
                filename_without_ext = Path(filename).stem
                if predicted_text.lower() in filename_without_ext.lower():
                    correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # Calculate average confidence
        avg_confidence = np.mean([conf for _, _, conf in results])
        
        evaluation_results = {
            'total_samples': total_count,
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'predictions': results
        }
        
        logger.info(f"Evaluation results: {correct_count}/{total_count} correct ({accuracy:.3f})")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        return evaluation_results
    
    def save_predictions(self, 
                        results: List[Tuple[str, str, float]], 
                        output_path: str):
        """
        Save prediction results to a file.
        
        Args:
            results: List of prediction results
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                f.write("filename,predicted_text,confidence\n")
                for filename, predicted_text, confidence in results:
                    f.write(f"{filename},{predicted_text},{confidence:.4f}\n")
            
            logger.info(f"Predictions saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict CAPTCHA text from images')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./data/test_data',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./predictions.csv',
        help='Output file for predictions'
    )
    parser.add_argument(
        '--single_image',
        type=str,
        default=None,
        help='Path to single image for prediction (optional)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    
    # Create predictor
    predictor = CaptchaPredictor(config, args.model_path)
    
    try:
        if args.single_image:
            # Single image prediction
            predicted_text, confidence = predictor.predict_single(args.single_image)
            print(f"Image: {args.single_image}")
            print(f"Predicted text: {predicted_text}")
            print(f"Confidence: {confidence:.4f}")
        else:
            # Batch prediction
            results = predictor.predict_batch(args.image_dir)
            
            # Print results
            for filename, predicted_text, confidence in results:
                print(f"{filename}: {predicted_text} (confidence: {confidence:.4f})")
            
            # Save results
            predictor.save_predictions(results, args.output_file)
            
            # Evaluate accuracy
            evaluation = predictor.evaluate_accuracy(args.image_dir)
            print(f"\nAccuracy: {evaluation['accuracy']:.3f}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
