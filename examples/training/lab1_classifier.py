"""End-to-end fine-tuning example built from the Lab 1 dataset.

Run from the repository root:
uv run python examples/training/lab1_classifier.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def load_status_dataset() -> Dataset:
    """Load the local trail status CSV as a Hugging Face dataset."""
    csv_path = Path(__file__).resolve().parent.parent / "loading" / "status.csv"
    dataframe = pd.read_csv(csv_path)
    return Dataset.from_pandas(dataframe)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Calculate accuracy for the validation split."""
    predictions, labels = eval_pred
    predicted_labels = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predicted_labels)}


def main() -> None:
    """Train and evaluate a binary classifier from the Lab 1 dataset."""
    dataset = load_status_dataset()
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(f"Train examples: {len(split_dataset['train'])}")
    print(f"Validation examples: {len(split_dataset['test'])}")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_and_label(examples: dict[str, list[str | int]]) -> dict[str, list[int]]:
        tokenized = tokenizer(examples["status"], truncation=True)
        tokenized["labels"] = examples["Blankets_Creek"]
        return tokenized

    tokenized_datasets = split_dataset.map(
        tokenize_and_label,
        batched=True,
        remove_columns=split_dataset["train"].column_names,
    )

    sample = tokenized_datasets["train"][0]
    print(f"Tokenized fields: {list(sample.keys())}")
    print(f"Input length: {len(sample['input_ids'])}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    output_dir = Path(__file__).resolve().parent / "lab1_classifier"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
