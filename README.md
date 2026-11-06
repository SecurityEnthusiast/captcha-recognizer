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

The defaults are 400 generated images, batch size 32, and 15 epochs on CPU. The seed is fixed. The command holds out 20% of the images it just drew, trains on the rest, and exits 0 only when both of these are true:

- last-epoch training loss is below the first epoch
- holdout per-character accuracy is above `1 / 62`

It prints that per-character accuracy, the holdout exact-match rate beside it, the common character confusions, and one prediction line for each file in `data/test_data/`. Those 12 files are a listing, not the score. A CSV of the listing is written to `predictions.csv`.

A default run on CPU (seed 42, dropout 0.5) printed:

- holdout exact match `0.0000`
- holdout per-character `0.0250` (chance is about `0.0161`)
- training loss from `20.6465` down to `20.1597`

The same settings with more generated images, still seed 42, dropout 0.5, and 15 epochs:

- 2000 images: exact match `0.0000`, per-character `0.0145` (below chance), loss `20.6380` to `20.5579`
- 4000 images: exact match `0.0000`, per-character `0.0175`, loss `20.6388` to `20.6003`

The holdout stays near chance. The 4000-image run passes the learning bar because `0.0175` is just above `0.0161`. Its common mistakes pile onto a few letters (`N`, `f`, `L`), and all 12 files in `data/test_data/` were read as `fGLNi`. The commands are `captcha-demo --samples 2000` and `captcha-demo --samples 4000`. The score log names the run with `samples:`, `epochs:`, `dropout:`, and `charset:`.

On that 400-image set, dropout off for 40 epochs drove training loss to `0.20` and holdout per-character accuracy to `0.0075`, below chance. The network memorizes a few hundred captchas. It does not carry that to new ones from the same generator. `--dropout 0` is there to see the memorization. `--digits-only` switches the alphabet to `0-9`.

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
