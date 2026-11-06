"""
Modern prediction module for CAPTCHA recognition.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import argparse
import sys
from datetime import datetime
import json
from PIL import Image

from .config import Config, ModelConfig
from .data_loader import label_from_filename, list_image_files
from .decode import decode_heads, ordered_heads, predict
from .metrics import character_accuracy, exact_matches
from .model import load_model
from .preprocess import load_image_batch

logger = logging.getLogger(__name__)


class CaptchaPredictor:
    """
    Predictor class for CAPTCHA recognition.
    
    This class handles loading trained models and making predictions
    on new CAPTCHA images.
    """
    
    def __init__(self, config: Config, model_path: Optional[str] = None):
        self.config = config
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
            return load_image_batch(
                image_path,
                self.config.model.image_width,
                self.config.model.image_height,
            )
            
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
        
        predictions = ordered_heads(
            self.model(image_array, training=False),
            self.config.model.chars_num,
        )
        
        predicted_text, confidence = decode_heads(
            predictions,
            self.config.model.char_set,
        )
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
        
        files = list_image_files(image_dir or self.config.data.test_data_dir)
        images = np.stack([
            load_image_batch(
                file,
                self.config.model.image_width,
                self.config.model.image_height,
            )[0]
            for file in files
        ])
        filenames = [file.name for file in files]

        predictions = ordered_heads(
            self.model(images, training=False),
            self.config.model.chars_num,
        )
        
        # Process results
        results = []
        for i, filename in enumerate(filenames):
            sample_heads = [head[i] for head in predictions]
            predicted_text, confidence = decode_heads(
                sample_heads,
                self.config.model.char_set,
            )
            results.append((filename, predicted_text, confidence))
        
        return results

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
        
        results = self.predict_batch(test_dir)
        pairs = []
        for filename, predicted_text, _confidence in results:
            if ground_truth and filename in ground_truth:
                expected = ground_truth[filename]
            else:
                expected = label_from_filename(filename)
            pairs.append((predicted_text, expected))

        exact = exact_matches(pairs)
        per_character = character_accuracy(pairs)
        average_confidence = (
            float(np.mean([confidence for _, _, confidence in results]))
            if results
            else 0.0
        )
        correct_count = sum(predicted == expected for predicted, expected in pairs)
        logger.info(
            "Evaluation results: %s/%s exact (%.3f), per-character %.3f",
            correct_count,
            len(results),
            exact,
            per_character,
        )
        return {
            "total_samples": len(results),
            "correct_predictions": correct_count,
            "accuracy": exact,
            "character_accuracy": per_character,
            "average_confidence": average_confidence,
            "predictions": results,
        }
    
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
