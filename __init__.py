"""
CAPTCHA Recognition System

A TensorFlow-based system for recognizing and solving CAPTCHA images.
"""

__version__ = "2.0.0"
__author__ = "CAPTCHA Recognizer Team"

from .config import Config
from .model import CaptchaModel
from .data_loader import CaptchaDataLoader
from .trainer import CaptchaTrainer
from .predictor import CaptchaPredictor

__all__ = [
    "Config",
    "CaptchaModel", 
    "CaptchaDataLoader",
    "CaptchaTrainer",
    "CaptchaPredictor"
]

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
