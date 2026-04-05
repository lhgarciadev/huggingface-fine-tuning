"""CSV processing pipeline using load_dataset + pandas operations.

Run from repository root:
uv run python examples/transform/csv_processing_mixed.py
"""

from __future__ import annotations

from pathlib import Path

import datasets


def main() -> None:
    """Load with datasets, then clean and transform with pandas."""
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir.parent / "loading" / "status.csv"
    output_csv = base_dir / "status.cleaned.mixed.csv"

    print("1) Loading CSV with datasets.load_dataset")
    dataset_dict = datasets.load_dataset("csv", data_files=str(input_csv))
    dataset = dataset_dict["train"]

    print("\n2) Converting to pandas for head/dropna/apply/to_csv")
    dataframe = dataset.to_pandas()
    print("Columns:", list(dataframe.columns))
    print("Rows:", len(dataframe))

    print("\n3) Inspecting with features and head()")
    print("Features:", dataset.features)
    print(dataframe.head(5))

    print("\n4) Cleaning missing values with dropna()")
    rows_before = len(dataframe)
    cleaned_df = dataframe.dropna(subset=["status", "Blankets_Creek", "Rope_Mill"]).copy()
    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning: {len(cleaned_df)}")

    print("\n5) Transforming status to lowercase with apply()")
    cleaned_df["status"] = cleaned_df["status"].apply(lambda value: str(value).lower())
    print(cleaned_df.head(3))

    print("\n6) Saving transformed CSV with to_csv()")
    cleaned_df.to_csv(output_csv, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Saved transformed CSV to: {output_csv}")


if __name__ == "__main__":
    main()
