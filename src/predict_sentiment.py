"""Predict the sentiment of a new review using the saved XLM-R model."""

import argparse
import pickle
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_review(review_text, model_folder):
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_folder)

    encoded_review = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        result = model(**encoded_review)

    predicted_number = torch.argmax(result.logits, dim=1).item()
    predicted_label = model.config.id2label[predicted_number]
    confidence = torch.softmax(result.logits, dim=1)[0][predicted_number].item()

    return predicted_label, confidence


def predict_classical_models(review_text, model_folder):
    with (model_folder / "tfidf_vectorizer.pkl").open("rb") as file:
        vectorizer = pickle.load(file)

    review_features = vectorizer.transform([review_text])
    predictions = {}

    with (model_folder / "logistic_regression.pkl").open("rb") as file:
        logistic_model = pickle.load(file)
    predictions["Logistic Regression"] = logistic_model.predict(review_features)[0]

    with (model_folder / "linear_svm.pkl").open("rb") as file:
        svm_model = pickle.load(file)
    predictions["Linear SVM"] = svm_model.predict(review_features)[0]

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a new review.")
    parser.add_argument("review", help="Review text to classify")
    parser.add_argument(
        "--model-folder",
        default="results/sentiment_results/xlm_roberta",
        help="Folder containing the saved transformer model",
    )
    arguments = parser.parse_args()

    model_folder = Path(arguments.model_folder)
    if not model_folder.exists():
        raise FileNotFoundError(
            "Saved model was not found. Run sentiment_pipeline.py first."
        )

    classical_folder = model_folder.parent / "classical_models"
    classical_predictions = predict_classical_models(
        arguments.review,
        classical_folder,
    )
    transformer_label, confidence = predict_review(arguments.review, model_folder)

    for model_name, label in classical_predictions.items():
        print(f"{model_name}: {label}")
    print(f"XLM-R Transformer: {transformer_label}")
    print(f"Transformer confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
