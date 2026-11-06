"""Generate, train, predict, and score. A few images, the real network."""

import csv
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

SAMPLES = 16
EPOCHS = 2


def _printed(captured: str, label: str) -> float:
    prefix = f"{label}: "
    for line in captured.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    raise AssertionError(label)


def test_demo_generates_trains_predicts_and_scores(tmp_path, capsys):
    from PIL import Image

    from captcha_recognizer.config import CHAR_SET
    from captcha_recognizer.data_loader import label_from_filename, list_image_files
    from captcha_recognizer.demo import run_demo

    repo = Path(__file__).resolve().parents[1]
    test_dir = repo / "data" / "test_data"
    predictions = tmp_path / "predictions.csv"
    output_dir = tmp_path / "demo"
    status = run_demo(
        samples=SAMPLES,
        epochs=EPOCHS,
        batch_size=8,
        seed=1,
        holdout_ratio=0.2,
        learning_rate=1e-3,
        output_dir=output_dir,
        test_dir=test_dir,
        predictions_path=predictions,
        char_set=CHAR_SET,
        font_size=28,
        dropout=0.5,
    )
    captured = capsys.readouterr().out

    assert status in (0, 1)
    if status == 0:
        assert "learning bar: passed" in captured
    else:
        assert "learning bar: failed" in captured
    assert "holdout confusions:" in captured
    assert "chance per-character: 0.0161" in captured
    for label in (
        "holdout exact match",
        "holdout per-character",
        "first epoch loss",
        "last epoch loss",
    ):
        assert math.isfinite(_printed(captured, label))

    exact = _printed(captured, "holdout exact match")
    per_character = _printed(captured, "holdout per-character")
    assert 0.0 <= exact <= 1.0
    assert 0.0 <= per_character <= 1.0

    train_files = list_image_files(output_dir / "train")
    holdout_files = list_image_files(output_dir / "holdout")
    assert list_image_files(output_dir / "generated") == []
    assert train_files
    assert holdout_files
    assert {path.name for path in train_files}.isdisjoint(
        path.name for path in holdout_files
    )
    assert len(train_files) + len(holdout_files) == SAMPLES
    for path in train_files + holdout_files:
        assert len(label_from_filename(path.name)) == 5
        with Image.open(path) as image:
            assert image.size == (128, 48)

    test_files = list_image_files(test_dir)
    with predictions.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["filename", "predicted_text", "confidence"]
    assert len(rows) == 1 + len(test_files)
    for filename, predicted, confidence in rows[1:]:
        assert filename
        assert len(predicted) == 5
        assert set(predicted) <= set(CHAR_SET)
        value = float(confidence)
        assert 0.0 <= value <= 1.0
