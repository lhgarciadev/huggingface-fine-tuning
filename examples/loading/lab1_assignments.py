"""Lab 1 walkthrough: loading and exploring datasets with Hugging Face Datasets.

Run from repository root:
uv run python examples/loading/lab1_assignments.py
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


def print_section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "status.csv"
    json_file = base_dir / "status.csv.json"
    parquet_file = base_dir / "status.parquet"

    print_section("Exercise 1: Load CSV")
    dataset = load_dataset("csv", data_files=str(csv_file))
    print(dataset)

    train_data = dataset["train"]
    print(f"Rows: {len(train_data)}")
    print(f"Columns: {train_data.column_names}")
    print(f"First row: {train_data[0]}")

    print_section("Exercise 2: Load JSON")
    json_dataset = load_dataset("json", data_files=str(json_file))
    print(json_dataset)
    print("CSV columns == JSON columns:", train_data.column_names == json_dataset["train"].column_names)

    print_section("Exercise 3: Load Parquet")
    parquet_dataset = load_dataset("parquet", data_files=str(parquet_file))
    print(parquet_dataset)
    print("Parquet is useful for analytics pipelines because columnar reads are faster and smaller on disk.")

    print_section("Exercise 4: Load from Hub")
    imdb = load_dataset("imdb")
    print(imdb)
    print(f"Train size: {len(imdb['train'])}")
    print(f"Test size: {len(imdb['test'])}")
    print("IMDB features:", imdb["train"].features)

    print_section("Exercise 5: Dataset Exploration")
    for i, example in enumerate(dataset["train"]):
        if i >= 3:
            break
        print(f"Example {i}: {example}")

    statuses = dataset["train"]["status"]
    print(f"First 3 statuses: {statuses[:3]}")
    print("Dataset info:", dataset["train"].info)

    print_section("Challenge Suggestion")
    print("Try: ag_news")
    print("Code: custom_ds = load_dataset('ag_news')")


if __name__ == "__main__":
    main()
