"""Short train-and-score demo for captchas this repo generates."""

import argparse
from pathlib import Path

from .data_loader import list_image_paths
from .generate import generate
from .labels import label_from_filename
from .metrics import score_pairs
from .predictor import predict
from .trainer import train


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ImageCaptcha samples, train for a few epochs, "
            "print holdout exact-match accuracy, and print one "
            "prediction line per file in data/test_data."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Training images to generate (default: 200).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs (default: 2).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default="data/demo")
    parser.add_argument(
        "--test-dir",
        default="data/test_data",
        help="Folder of images to print one prediction line per file.",
    )
    args = parser.parse_args(argv)
    if args.samples < 1 or args.epochs < 1:
        parser.error("--samples and --epochs must be at least 1")

    work = Path(args.work_dir)
    train_dir = work / "train"
    holdout_dir = work / "holdout"
    holdout_count = max(1, args.samples // 5)
    generate(train_dir, args.samples, seed=args.seed)
    generate(holdout_dir, holdout_count, seed=args.seed + 1)

    model = train(
        str(train_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    holdout_pairs = []
    for path in list_image_paths(holdout_dir):
        text, _confidence = predict(path, model)
        holdout_pairs.append((text, label_from_filename(path.name)))
    scores = score_pairs(holdout_pairs)
    print(
        "holdout exact-match accuracy: "
        f"{scores['exact_match']:.3f} ({scores['exact_correct']}/{scores['total']})"
    )
    print(f"holdout per-character accuracy: {scores['per_character']:.3f}")

    rows = []
    for path in list_image_paths(args.test_dir):
        text, confidence = predict(path, model)
        print(f"{path.name}: {text} (confidence {confidence:.3f})")
        rows.append((path.name, text, confidence))
    _write_predictions(work / "predictions.csv", rows)
    return 0


def _write_predictions(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("filename,predicted_text,confidence\n")
        for filename, text, confidence in rows:
            handle.write(f"{filename},{text},{confidence:.4f}\n")


def console_main():
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
