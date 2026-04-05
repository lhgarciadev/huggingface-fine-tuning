"""CSV cleaning and transformation pipeline using Hugging Face Datasets only.

Run from repository root:
uv run python examples/transform/csv_processing_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import datasets


def head_rows(dataset: datasets.Dataset, n: int = 5) -> list[dict[str, object]]:
    """Return the first n rows, similar to pandas head()."""
    limit = min(n, len(dataset))
    subset = dataset.select(range(limit))
    return [subset[i] for i in range(limit)]


def main() -> None:
    """Load, inspect, clean, transform, and save a CSV dataset."""
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir.parent / "loading" / "status.csv"
    output_csv = base_dir / "status.cleaned.csv"

    print("1) Loading CSV dataset")
    dataset_dict = datasets.load_dataset("csv", data_files=str(input_csv))
    dataset = dataset_dict["train"]
    print(f"Rows: {len(dataset)}")

    print("\n2) Inspecting structure")
    print("Features:", dataset.features)
    print("Head rows:")
    for row in head_rows(dataset, n=5):
        print(row)

    print("\n3) Cleaning rows with missing values")
    rows_before = len(dataset)
    cleaned_dataset = dataset.filter(
        lambda example: example.get("status") is not None
        and example.get("Blankets_Creek") is not None
        and example.get("Rope_Mill") is not None
    )
    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning: {len(cleaned_dataset)}")

    print("\n4) Transforming status to lowercase")
    transformed_dataset = cleaned_dataset.map(
        lambda example: {"status": example["status"].lower()}
    )
    print("Sample transformed rows:")
    for row in head_rows(transformed_dataset, n=3):
        print(row)

    print("\n5) Saving transformed dataset")
    transformed_dataset.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved transformed CSV to: {output_csv}")


if __name__ == "__main__":
    main()
