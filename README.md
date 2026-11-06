# Captcha Recognizer

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

A local trainer for 5-character text captchas generated in this repository. Images are grayscale and 128×48. The label is the filename prefix before the first `_`.

This project does not call reCAPTCHA, hCaptcha, or Turnstile.

## Quick Start

```bash
captcha-demo --samples 200 --epochs 2
```

## Install

```bash
pip install -e .
```

That installs TensorFlow, NumPy, Pillow, and captcha. The demo command generates a few hundred ImageCaptcha images, trains for two epochs, prints holdout exact-match accuracy, and prints one prediction line for each file in `data/test_data/`.

## Layout

- `captcha_recognizer/` is the importable trainer.
- `data/test_data/` holds the checked-in sample images.
- `legacy/` holds the old TensorFlow 1 scripts. They are not the quick start.

## Contributing

Clone [SecurityEnthusiast/captcha-recognizer](https://github.com/SecurityEnthusiast/captcha-recognizer) and open a pull request there.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Support

- [Issues](https://github.com/SecurityEnthusiast/captcha-recognizer/issues)
- [Source](https://github.com/SecurityEnthusiast/captcha-recognizer)
