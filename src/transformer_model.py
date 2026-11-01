"""XLM-R transformer model for sentiment classification."""

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_loader import LABEL_ORDER
from evaluation import save_confusion_matrix


def train_transformer(train_data, test_data, output_folder, model_name, epochs):
    """Train XLM-R and return its test results."""
    load_dotenv()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install torch, transformers, and datasets before running the transformer."
        ) from error

    label_to_number = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    number_to_label = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        label2id=label_to_number,
        id2label=number_to_label,
    )

    train_for_model = train_data.copy()
    test_for_model = test_data.copy()
    train_for_model["label"] = train_for_model["label"].map(label_to_number)
    test_for_model["label"] = test_for_model["label"].map(label_to_number)

    train_dataset = Dataset.from_pandas(train_for_model, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_for_model, preserve_index=False)

    def tokenize_reviews(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize_reviews, batched=True)
    test_dataset = test_dataset.map(tokenize_reviews, batched=True)

    def calculate_metrics(result):
        predictions = np.argmax(result.predictions, axis=1)
        actual = result.label_ids

        precision, recall, f1, unused_value = precision_recall_fscore_support(
            actual,
            predictions,
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": float(accuracy_score(actual, predictions)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }

    training_settings = TrainingArguments(
        output_dir=str(Path(output_folder) / "xlm_roberta"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_settings,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=calculate_metrics,
    )

    trainer.train()
    trainer.save_model(str(Path(output_folder) / "xlm_roberta"))
    tokenizer.save_pretrained(str(Path(output_folder) / "xlm_roberta"))
    test_results = trainer.evaluate()

    prediction_result = trainer.predict(test_dataset)
    predicted_numbers = np.argmax(prediction_result.predictions, axis=1)
    predicted_labels = pd.Series(predicted_numbers).map(number_to_label)

    save_confusion_matrix(
        test_data["label"],
        predicted_labels,
        "xlm_roberta",
        output_folder,
    )

    simple_results = {}
    for key, value in test_results.items():
        if isinstance(value, (int, float)):
            simple_results[key] = float(value)

    return simple_results
