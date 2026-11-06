# Captcha Recognizer

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A powerful and efficient CAPTCHA recognition system built with TensorFlow that can recognize text-based CAPTCHAs without requiring image segmentation. This project provides a complete pipeline from CAPTCHA generation to neural network training and recognition.

## 🚀 Features

- **No Image Segmentation Required**: Direct end-to-end recognition using deep learning
- **High Accuracy**: Achieves excellent recognition rates with proper training data
- **Multi-GPU Support**: Train on multiple GPUs for faster training
- **Flexible Input**: Supports both JPG and PNG image formats
- **Easy Training**: Simple pipeline from data preparation to model deployment
- **Production Ready**: Includes evaluation and recognition scripts for real-world use

## 📋 Requirements

### System Requirements
- **OS**: Ubuntu 18.04+ (tested on Ubuntu 18.04)
- **Python**: 3.10+
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large datasets)

### Dependencies
- **Python 3.10**
- **TensorFlow 2.10.0**
- **NumPy 1.23.4**
- **captcha** package

## 🛠️ Installation

### Option 1: Using pip
```bash
pip install -r requirements.txt
```

### Option 2: Manual Installation
```bash
# Install TensorFlow
pip install tensorflow==2.10.0

# Install NumPy
pip install numpy==1.23.4

# Install captcha package
pip install captcha==0.1.1
```

## Project structure

```
captcha-recognizer/
├── captcha_recognizer/   # Importable trainer
├── data/test_data/       # Checked-in sample images
├── legacy/               # Old TensorFlow 1 scripts (not the quick start)
└── requirements.txt
```

## Quick Start

```bash
pip install -e .
python -c "import captcha_recognizer"
```

The scripts in `legacy/` use removed TensorFlow 1 queue runners. They are kept for reference and are not part of this quick start.

## 🔧 Configuration

### Model Parameters
The neural network architecture and training parameters can be customized in `config.py`:

- **Input dimensions**: Image width and height
- **Character set**: Supported characters for recognition
- **Network architecture**: Layer sizes and activation functions
- **Training parameters**: Learning rate, batch size, epochs

### Training Configuration
Adjust training parameters in the training scripts:
- **Batch size**: Adjust based on available GPU memory
- **Learning rate**: Start with default and tune as needed
- **Epochs**: More epochs for better accuracy (with proper validation)

## 📊 Performance

### Accuracy Factors
- **Training Data Size**: Larger datasets improve accuracy
- **Data Quality**: Clean, diverse CAPTCHAs work better
- **Training Steps**: More iterations generally help
- **Model Architecture**: Optimized for CAPTCHA recognition

### Optimization Tips
- Use data augmentation for better generalization
- Implement early stopping to prevent overfitting
- Experiment with different learning rates
- Consider ensemble methods for production use

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/captcha-recognizer.git
cd captcha-recognizer

# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_basic.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CAPTCHA Generator**: [Gregwar/CaptchaBundle](https://github.com/Gregwar/CaptchaBundle) for sample CAPTCHA generation
- **TensorFlow**: For the deep learning framework
- **Community**: All contributors and users of this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/captcha-recognizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/captcha-recognizer/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/captcha-recognizer/wiki)

---

**Made with ❤️ for the open-source community**

*If you find this project useful, please consider giving it a ⭐ star!*

