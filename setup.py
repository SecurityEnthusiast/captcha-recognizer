#!/usr/bin/env python3
"""
Setup script for CAPTCHA Recognizer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="captcha-recognizer",
    version="3.0.0",
    author="CAPTCHA Recognizer Team",
    author_email="team@captcha-recognizer.com",
    description="A modern CAPTCHA recognition system built with TensorFlow 2.x",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/captcha-recognizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.15.0",
        ],
        "full": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "opencv-python>=4.8.0",
            "albumentations>=1.3.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "captcha-train=captcha_recognizer.trainer:main",
            "captcha-predict=captcha_recognizer.predictor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "captcha_recognizer": ["*.md", "*.txt"],
    },
    keywords="captcha recognition tensorflow machine learning computer vision",
    project_urls={
        "Bug Reports": "https://github.com/your-username/captcha-recognizer/issues",
        "Source": "https://github.com/your-username/captcha-recognizer",
        "Documentation": "https://github.com/your-username/captcha-recognizer#readme",
    },
)
