"""Train the email/SMS message model without opening links or attachments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.text_features import TEXT_FEATURE_COLUMNS, text_feature_frame, validate_text


def read_message_dataset(path: Path) -> tuple[list[str], pd.Series, pd.Series | None]:
    data = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(required)}; missing {sorted(missing)}.")
    data = data.dropna(subset=["text", "label"]).copy()
    try:
        data["text"] = data.text.map(validate_text)
    except ValueError as exc:
        raise ValueError(f"Dataset contains invalid message text: {exc}") from exc
    data["label"] = pd.to_numeric(data.label, errors="raise").astype(int)
    if not set(data.label.unique()).issubset({0, 1}) or data.label.nunique() != 2:
        raise ValueError("label must contain both 0 (legitimate) and 1 (phishing).")
    conflict = data.groupby("text").label.nunique()
    if (conflict > 1).any():
        raise ValueError("The same message text has conflicting labels; clean the dataset before training.")
    data = data.drop_duplicates(subset="text").reset_index(drop=True)
    groups = data.source_group.astype(str) if "source_group" in data.columns else None
    return data.text.tolist(), data.label, groups


def make_message_matrix(messages: list[str], vectorizer: TfidfVectorizer | None = None, *, fit: bool = False):
    numeric = csr_matrix(text_feature_frame(messages).astype(float).to_numpy())
    if fit:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=20_000, sublinear_tf=True)
        return hstack([numeric, vectorizer.fit_transform(messages)], format="csr"), vectorizer
    if vectorizer is None:
        raise ValueError("A fitted vectorizer is required for prediction.")
    return hstack([numeric, vectorizer.transform(messages)], format="csr")


def predict_message_bundle(bundle: dict, messages: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    normalized = [validate_text(message) for message in messages]
    features = text_feature_frame(normalized)
    matrix = hstack([csr_matrix(features.astype(float).to_numpy()), bundle["vectorizer"].transform(normalized)], format="csr")
    return bundle["model"].predict_proba(matrix)[:, 1], features


def make_split(labels: pd.Series, groups: pd.Series | None, test_size: float) -> tuple[np.ndarray, np.ndarray, str]:
    folds = max(2, min(10, round(1 / test_size)))
    if groups is not None and groups.nunique() >= folds:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        try:
            for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels, groups):
                if labels.iloc[train_idx].nunique() == 2 and labels.iloc[test_idx].nunique() == 2:
                    return train_idx, test_idx, "Stratified, source-group-disjoint held-out split"
        except ValueError:
            pass
        raise ValueError("Not enough varied source groups per label for a group-disjoint split.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return train_idx, test_idx, "Stratified row-wise held-out split (add source_group for stronger evaluation)"


def train_messages(dataset: Path, model_path: Path, metrics_path: Path, test_size: float = 0.25) -> dict:
    messages, labels, groups = read_message_dataset(dataset)
    train_idx, test_idx, evaluation = make_split(labels, groups, test_size)
    train_messages_list = [messages[index] for index in train_idx]
    test_messages_list = [messages[index] for index in test_idx]
    x_train, vectorizer = make_message_matrix(train_messages_list, fit=True)
    x_test = make_message_matrix(test_messages_list, vectorizer)
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=1)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    metrics = {
        "evaluation": evaluation, "accuracy": accuracy_score(y_test, predictions), "precision_phishing": precision,
        "recall_phishing": recall, "f1_phishing": f1, "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities), "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)), "source_group_overlap": int(len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))) if groups is not None else None,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, predictions, target_names=["legitimate", "phishing"], output_dict=True, zero_division=0),
        "model": "RandomForest (safe message features + character TF-IDF n-grams)", "feature_count": int(x_train.shape[1]),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer, "feature_columns": TEXT_FEATURE_COLUMNS, "version": 1}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhishGuard AI's email/SMS text model. Links and attachments are never opened.")
    parser.add_argument("--dataset", type=Path, default=Path("data/sample_messages.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/phishguard_message_rf.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("models/message_metrics.json"))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")
    print(json.dumps(train_messages(args.dataset, args.model, args.metrics, args.test_size), indent=2))


if __name__ == "__main__":
    main()
