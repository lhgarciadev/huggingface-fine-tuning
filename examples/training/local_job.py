"""Run a local Hugging Face training job with metric logging.

This script is the local equivalent of a SageMaker training job workflow:
- Local CSV input instead of S3 channels
- Trainer + TrainingArguments instead of managed job config
- JSONL metric logs instead of CloudWatch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class JsonlMetricLogger(TrainerCallback):
    """Write Trainer log events to a JSONL file for later analysis."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **logs,
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Trainer job with JSONL metrics.")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV dataset.")
    parser.add_argument("--text-column", default="status", help="Text feature column name.")
    parser.add_argument("--label-column", default="Blankets_Creek", help="Label column name.")
    parser.add_argument("--model-name", default="bert-base-uncased", help="Pretrained model to fine-tune.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/training/trail_classifier_local"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("examples/training/trail_training_log.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_dataset_columns(df: pd.DataFrame, text_column: str, label_column: str) -> None:
    missing = [col for col in (text_column, label_column) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_labels(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    label_values = df[label_column]
    if label_values.dtype == bool:
        df[label_column] = label_values.astype(int)
        return df

    unique = sorted(set(label_values.dropna().tolist()))
    if unique == [0, 1]:
        return df

    # Preserve deterministic mapping for non-numeric binary labels.
    label_map = {value: idx for idx, value in enumerate(unique)}
    if len(label_map) != 2:
        raise ValueError(
            "This script expects binary labels with exactly 2 classes. "
            f"Found classes: {unique}"
        )
    df[label_column] = df[label_column].map(label_map)
    return df


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    predictions, labels = eval_pred
    predicted_labels = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predicted_labels),
        "precision": precision_score(labels, predicted_labels, zero_division=0),
        "recall": recall_score(labels, predicted_labels, zero_division=0),
        "f1": f1_score(labels, predicted_labels, zero_division=0),
    }


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = pd.read_csv(args.data)
    _validate_dataset_columns(df, args.text_column, args.label_column)
    df = _normalize_labels(df, args.label_column)

    dataset = Dataset.from_pandas(df)
    split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def tokenize_and_label(examples: dict[str, Any]) -> dict[str, Any]:
        tokenized = tokenizer(examples[args.text_column], truncation=True, padding=True)
        tokenized["labels"] = examples[args.label_column]
        return tokenized

    tokenized_datasets = split_dataset.map(tokenize_and_label, batched=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to="none",
    )

    logger_callback = JsonlMetricLogger(args.log_file)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[logger_callback],
    )

    print(f"Train examples: {len(tokenized_datasets['train'])}")
    print(f"Eval examples: {len(tokenized_datasets['test'])}")
    print(f"Metrics log: {args.log_file}")

    trainer.train()

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Saved model artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
