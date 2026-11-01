"""Functions for measuring model performance."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data_loader import LABEL_ORDER


def metric_summary(actual_labels, predicted_labels):
    """Return the main classification scores."""
    precision, recall, f1, unused_value = precision_recall_fscore_support(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(actual_labels, predicted_labels)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            actual_labels,
            predicted_labels,
            labels=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_confusion_matrix(actual_labels, predicted_labels, model_name, output_folder):
    """Save a picture showing actual and predicted labels."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=LABEL_ORDER,
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(model_name + " confusion matrix")
    plt.tight_layout()

    file_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(Path(output_folder) / file_name, dpi=160)
    plt.close()
