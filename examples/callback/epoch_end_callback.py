"""Custom epoch-end callback with plateau-based early stopping.

This example shows how to:
- implement `on_epoch_end` in a custom callback,
- log epoch metrics to a JSONL file, and
- stop training when eval loss plateaus.
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


class EpochEndMetricsCallback(TrainerCallback):
    """Log epoch metrics and apply plateau-based early stopping on eval_loss."""

    def __init__(self, log_file: Path, patience: int = 2, min_delta: float = 0.0) -> None:
        self.log_file = log_file
        self.patience = patience
        self.min_delta = min_delta
        self.best_eval_loss = float("inf")
        self.epochs_without_improvement = 0
        self._last_processed_step = -1

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text("", encoding="utf-8")

    def _latest_eval_metrics(self, state: Any) -> dict[str, Any] | None:
        for event in reversed(state.log_history):
            if "eval_loss" in event:
                return event
        return None

    def on_epoch_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step == self._last_processed_step:
            return

        metrics = self._latest_eval_metrics(state)
        self._last_processed_step = int(state.global_step)

        eval_loss = float(metrics["eval_loss"]) if metrics and "eval_loss" in metrics else None
        if eval_loss is not None:
            improved = (self.best_eval_loss - eval_loss) > self.min_delta
            if improved:
                self.best_eval_loss = eval_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        should_stop = eval_loss is not None and self.epochs_without_improvement >= self.patience
        if should_stop:
            control.should_training_stop = True

        payload = {
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "step": int(state.global_step),
            "eval_loss": eval_loss,
            "eval_accuracy": metrics.get("eval_accuracy") if metrics else None,
            "eval_precision": metrics.get("eval_precision") if metrics else None,
            "eval_recall": metrics.get("eval_recall") if metrics else None,
            "eval_f1": metrics.get("eval_f1") if metrics else None,
            "best_eval_loss": None if self.best_eval_loss == float("inf") else self.best_eval_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "should_stop": should_stop,
        }

        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

        print(
            f"[epoch_end] epoch={payload['epoch']:.1f} "
            f"eval_loss={payload['eval_loss']} best_eval_loss={payload['best_eval_loss']} "
            f"wait={self.epochs_without_improvement}/{self.patience} "
            f"stop={should_stop}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with a custom epoch-end callback.")
    parser.add_argument("--data", type=Path, default=Path("examples/loading/status.csv"))
    parser.add_argument("--text-column", default="status")
    parser.add_argument("--label-column", default="Blankets_Creek")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/callback/trail_classifier_epoch_cb"))
    parser.add_argument("--log-file", type=Path, default=Path("examples/callback/epoch_metrics.jsonl"))
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, text_column: str, label_column: str) -> None:
    missing = [column for column in (text_column, label_column) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_binary_labels(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    label_values = df[label_column]
    if label_values.dtype == bool:
        df[label_column] = label_values.astype(int)
        return df

    unique = sorted(set(label_values.dropna().tolist()))
    if unique == [0, 1]:
        return df

    label_map = {value: index for index, value in enumerate(unique)}
    if len(label_map) != 2:
        raise ValueError(f"Expected binary labels. Found classes: {unique}")

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
    _validate_columns(df, args.text_column, args.label_column)
    df = _normalize_binary_labels(df, args.label_column)

    dataset = Dataset.from_pandas(df)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    if args.max_train_samples > 0:
        max_train = min(args.max_train_samples, len(split_dataset["train"]))
        split_dataset["train"] = split_dataset["train"].select(range(max_train))
    if args.max_eval_samples > 0:
        max_eval = min(args.max_eval_samples, len(split_dataset["test"]))
        split_dataset["test"] = split_dataset["test"].select(range(max_eval))

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
        report_to="none",
        disable_tqdm=True,
    )

    epoch_callback = EpochEndMetricsCallback(
        log_file=args.log_file,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[epoch_callback],
    )

    print(f"Train examples: {len(tokenized_datasets['train'])}")
    print(f"Eval examples: {len(tokenized_datasets['test'])}")
    print(f"Epoch log file: {args.log_file}")
    print(f"Early stopping: patience={args.patience}, min_delta={args.min_delta}")

    trainer.train()

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Saved model artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
