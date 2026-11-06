# Captcha Recognizer

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Train a fixed-length text captcha model on images this repo generates, then read an exact-match score and a per-character score. It is a local trainer. It does not call reCAPTCHA, hCaptcha, or Turnstile.

Images are grayscale, 128×48, and exactly 5 characters. The default alphabet is 62 symbols (lower, digits, upper). The label is the filename prefix before the first `_`, as in `1ab2s_num286.jpg` and `CMQVA_num717_1.png`.

## Install

Python 3.12. On a slim Linux image, install a font first (`fonts-dejavu-core` on Ubuntu). `ImageCaptcha` cannot draw without one.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

```bash
captcha-demo
```

The defaults are 2000 generated images, batch size 32, and 15 epochs on CPU. The seed is fixed. The command holds out 20% of the images it just drew, trains on the rest, and exits 0 only when both of these are true:

- last-epoch training loss is below the first epoch
- holdout per-character accuracy is above `1 / 62`

It prints that per-character accuracy, the holdout exact-match rate beside it, the common character confusions, and one prediction line for each file in `data/test_data/`. Those 12 files are a listing, not the score. A CSV of the listing is written to `predictions.csv`.

Each character is a slice of the feature map, and the slice keeps its rows. One shared softmax scores every slice. A run on CPU (seed 42, dropout 0.5, 15 epochs, 2000 images) printed:

- holdout exact match `0.0625`
- holdout per-character `0.5430` (chance is about `0.0161`)
- training loss from `20.6388` down to `8.1171`

The usual mistakes are still lookalikes (`D`/`0`, `O`/`0`, `X`/`x`, `V`/`v`, `8`/`B`, `I`/`l`, `S`/`5`). Pooling those rows away, at the same settings, reached per-character `0.3070` and exact match `0.0000`. The score log names the run with `samples:`, `epochs:`, `dropout:`, and `charset:`.

The previous head was one 1024-unit layer on an 8×3 map. At these settings its holdout stayed near chance: 400 images per-character `0.0250`, 2000 images `0.0145`, and 4000 images `0.0175`. At 4000 images it read every file in `data/test_data/` as `fGLNi`. On a 400-image set, dropout off for 40 epochs on that head drove training loss to `0.20` and holdout per-character accuracy to `0.0075`, below chance. `--dropout 0` is there to see that memorization. `--digits-only` switches the alphabet to `0-9`.

## Train on your own folder

Put images in a directory named `{label}_anything.png`. The label is five characters from the alphabet above.

```bash
python -m captcha_recognizer.trainer --train-dir ./data/train_data --valid-dir ./data/valid_data
```

There is no `captcha-train` console script. The only installed command is `captcha-demo`.

## Legacy scripts

`legacy/` is the old TensorFlow 1 queue-runner path. It is not maintained and it is not the quick start. See [legacy/README.md](legacy/README.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache License 2.0. See [LICENSE](LICENSE). The copyright line in that file is unset until the owner of the 2017 merge from `PatrickLib/captcha_recognition` is confirmed. This repository does not switch the license to MIT.

Source: <https://github.com/SecurityEnthusiast/captcha-recognizer>
