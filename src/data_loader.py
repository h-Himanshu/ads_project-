"""Functions for reading and preparing the review dataset."""

import re
from pathlib import Path

import pandas as pd


LABEL_ORDER = ["negative", "neutral", "positive"]
DEFAULT_TEXT_COLUMN = "Review"
DEFAULT_LABEL_COLUMN = "sentiment"


def clean_text(value):
    """Clean one review before sending it to a model."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_dataset(path: Path, text_column: str, label_column: str):
    """Read the CSV and return clean text with its sentiment labels."""
    data = pd.read_csv(path)

    if text_column not in data.columns:
        raise ValueError(
            f"Text column '{text_column}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    if label_column not in data.columns:
        raise ValueError(
            f"Sentiment column '{label_column}' was not found. "
            "Add a sentiment column containing positive, neutral, or negative."
        )

    labels = data[label_column].astype("string").str.strip().str.lower()
    invalid_labels = ~labels.isin(LABEL_ORDER)
    if invalid_labels.any():
        raise ValueError(
            "The sentiment column contains missing or invalid values. "
            "Only positive, neutral, and negative are allowed."
        )

    text = data[text_column].fillna("").map(clean_text)

    clean_data = pd.DataFrame({"text": text, "label": labels})
    clean_data = clean_data[clean_data["text"] != ""]
    clean_data = clean_data[clean_data["label"].isin(LABEL_ORDER)]
    clean_data = clean_data.drop_duplicates(subset=["text"])
    clean_data = clean_data.reset_index(drop=True)

    if len(clean_data) == 0:
        raise ValueError("No usable review text and labels were found.")

    if clean_data["label"].nunique() < 2:
        raise ValueError("At least two sentiment classes are needed for training.")

    return clean_data, "provided_labels"
