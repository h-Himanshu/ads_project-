"""Run the complete sentiment classification project."""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from classical_models import train_classical_models
from data_loader import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TEXT_COLUMN,
    LABEL_ORDER,
    load_dataset,
)
from transformer_model import train_transformer


def read_arguments():
    parser = argparse.ArgumentParser(
        description="Train classical and transformer sentiment models."
    )
    parser.add_argument(
        "--data",
        default="data/daraz_review_filtered2.csv",
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sentiment_results",
        help="Folder for metrics and images",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Column containing the review text",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Column containing sentiment labels",
    )
    parser.add_argument(
        "--transformer-model",
        default="xlm-roberta-base",
        help="Hugging Face model name",
    )
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main():
    arguments = read_arguments()
    output_folder = Path(arguments.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    data, label_source = load_dataset(
        Path(arguments.data),
        arguments.text_column,
        arguments.label_column,
    )

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    results = {}
    results["dataset"] = {
        "rows_used": len(data),
        "class_distribution": data["label"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
        .to_dict(),
        "label_source": label_source,
    }

    results["classical_models"] = train_classical_models(
        train_data["text"],
        test_data["text"],
        train_data["label"],
        test_data["label"],
        output_folder,
    )

    results["transformer"] = train_transformer(
        train_data,
        test_data,
        output_folder,
        arguments.transformer_model,
        arguments.epochs,
    )

    metrics_file = output_folder / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
