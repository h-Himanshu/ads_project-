"""Classical machine-learning models for sentiment classification."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from evaluation import metric_summary, save_confusion_matrix


def train_classical_models(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, dict[str, object]]:
    # First convert the review text into numbers.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(x_train)
    test_features = vectorizer.transform(x_test)

    model_folder = Path(output_dir) / "classical_models"
    model_folder.mkdir(parents=True, exist_ok=True)
    with (model_folder / "tfidf_vectorizer.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)

    results = {}

    # Train and evaluate Logistic Regression.
    logistic_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    logistic_model.fit(train_features, y_train)
    with (model_folder / "logistic_regression.pkl").open("wb") as file:
        pickle.dump(logistic_model, file)
    logistic_predictions = logistic_model.predict(test_features)
    logistic_name = "tfidf_logistic_regression"
    results[logistic_name] = metric_summary(y_test, logistic_predictions)
    save_confusion_matrix(
        y_test,
        logistic_predictions,
        logistic_name,
        output_dir,
    )

    # Train and evaluate Linear SVM.
    svm_model = LinearSVC(class_weight="balanced")
    svm_model.fit(train_features, y_train)
    with (model_folder / "linear_svm.pkl").open("wb") as file:
        pickle.dump(svm_model, file)
    svm_predictions = svm_model.predict(test_features)
    svm_name = "tfidf_linear_svm"
    results[svm_name] = metric_summary(y_test, svm_predictions)
    save_confusion_matrix(
        y_test,
        svm_predictions,
        svm_name,
        output_dir,
    )

    return results
