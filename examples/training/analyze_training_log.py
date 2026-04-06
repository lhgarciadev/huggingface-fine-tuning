"""Analyze local Trainer JSONL logs and summarize model performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local training log JSONL file.")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to JSONL log file.")
    return parser.parse_args()


def load_records(log_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _max_metric(records: list[dict[str, Any]], metric_name: str) -> tuple[float | None, int | None]:
    best_value: float | None = None
    best_step: int | None = None
    for item in records:
        value = item.get(metric_name)
        if isinstance(value, (int, float)) and (best_value is None or value > best_value):
            best_value = float(value)
            best_step = int(item.get("step", -1))
    return best_value, best_step


def _min_metric(records: list[dict[str, Any]], metric_name: str) -> tuple[float | None, int | None]:
    best_value: float | None = None
    best_step: int | None = None
    for item in records:
        value = item.get(metric_name)
        if isinstance(value, (int, float)) and (best_value is None or value < best_value):
            best_value = float(value)
            best_step = int(item.get("step", -1))
    return best_value, best_step


def _extract_metric_series(records: list[dict[str, Any]], metric_name: str) -> list[float]:
    series: list[float] = []
    for item in records:
        value = item.get(metric_name)
        if isinstance(value, (int, float)):
            series.append(float(value))
    return series


def _overfitting_signal(eval_loss_series: list[float], train_loss_series: list[float]) -> str:
    if len(eval_loss_series) < 3 or len(train_loss_series) < 3:
        return "insufficient data"

    eval_trend_up = eval_loss_series[-1] > eval_loss_series[-2] > eval_loss_series[-3]
    train_trend_down = train_loss_series[-1] < train_loss_series[-2] < train_loss_series[-3]

    if eval_trend_up and train_trend_down:
        return "possible overfitting"
    return "no clear overfitting pattern"


def main() -> None:
    args = parse_args()
    if not args.log_file.exists():
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    records = load_records(args.log_file)
    if not records:
        raise ValueError("No log records found in file.")

    best_accuracy, best_accuracy_step = _max_metric(records, "eval_accuracy")
    best_precision, best_precision_step = _max_metric(records, "eval_precision")
    best_recall, best_recall_step = _max_metric(records, "eval_recall")
    best_f1, best_f1_step = _max_metric(records, "eval_f1")
    lowest_train_loss, lowest_train_loss_step = _min_metric(records, "loss")
    lowest_eval_loss, lowest_eval_loss_step = _min_metric(records, "eval_loss")

    eval_loss_series = _extract_metric_series(records, "eval_loss")
    train_loss_series = _extract_metric_series(records, "loss")

    print("=== Training Summary ===")
    print(f"Log records: {len(records)}")

    if best_accuracy is not None:
        print(f"Best eval_accuracy: {best_accuracy:.4f} (step {best_accuracy_step})")
    if best_precision is not None:
        print(f"Best eval_precision: {best_precision:.4f} (step {best_precision_step})")
    if best_recall is not None:
        print(f"Best eval_recall: {best_recall:.4f} (step {best_recall_step})")
    if best_f1 is not None:
        print(f"Best eval_f1: {best_f1:.4f} (step {best_f1_step})")
    if lowest_train_loss is not None:
        print(f"Lowest train loss: {lowest_train_loss:.4f} (step {lowest_train_loss_step})")
    if lowest_eval_loss is not None:
        print(f"Lowest eval_loss: {lowest_eval_loss:.4f} (step {lowest_eval_loss_step})")

    print(f"Generalization signal: {_overfitting_signal(eval_loss_series, train_loss_series)}")

    print("\nOptimization opportunities:")
    print("- If eval metrics plateau early, reduce epochs or enable early stopping.")
    print("- If loss is noisy, reduce learning rate (for example: 2e-5 -> 1e-5).")
    print("- If training is slow and GPU is available, try fp16 configuration.")


if __name__ == "__main__":
    main()
