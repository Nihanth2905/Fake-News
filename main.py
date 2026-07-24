"""Command-line entry point for fake-news detection."""
from pathlib import Path

from src.preprocess import load_dataset


if __name__ == "__main__":
    dataset = Path("data/train.csv")
    if dataset.exists():
        print(f"Loaded {len(load_dataset(dataset))} records from {dataset}.")
    else:
        print("Dataset not found. Add data/train.csv, then run the training pipeline.")
