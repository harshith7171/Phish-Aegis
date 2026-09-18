"""Leakage-aware training for PhishGuard AI; submitted URLs are never opened."""
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
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.features import FEATURE_COLUMNS, feature_frame, normalize_url, registered_domain


def read_dataset(path: Path) -> tuple[list[str], pd.Series, pd.Series]:
    data = pd.read_csv(path)
    required = {"url", "label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(required)}; missing {sorted(missing)}.")
    data = data.dropna(subset=["url", "label"]).copy()
    try:
        data["url"] = data["url"].map(normalize_url)
    except ValueError as exc:
        raise ValueError(f"Dataset contains an invalid URL: {exc}") from exc
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    if not set(data.label.unique()).issubset({0, 1}) or data.label.nunique() != 2:
        raise ValueError("label must contain both 0 (legitimate) and 1 (phishing).")
    duplicates = data.groupby("url")["label"].nunique()
    if (duplicates > 1).any():
        raise ValueError("The same URL has conflicting labels; clean the dataset before training.")
    data = data.drop_duplicates(subset="url").reset_index(drop=True)
    groups = data.url.map(registered_domain)
    if (groups == "invalid").any():
        raise ValueError("Dataset contains an invalid URL.")
    if groups.nunique() < 4:
        raise ValueError("Use at least four distinct domain groups for a domain-disjoint split.")
    return data.url.tolist(), data.label, groups


def make_matrix(urls: list[str], vectorizer: TfidfVectorizer | None = None, *, fit: bool = False):
    """Join safe handcrafted features and character n-grams without network access."""
    numeric = csr_matrix(feature_frame(urls).astype(float).to_numpy())
    if fit:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=15_000, sublinear_tf=True)
        text = vectorizer.fit_transform(urls)
        return hstack([numeric, text], format="csr"), vectorizer
    if vectorizer is None:
        raise ValueError("A fitted vectorizer is required for prediction.")
    return hstack([numeric, vectorizer.transform(urls)], format="csr")


def predict_bundle(bundle: dict, urls: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    normalized = [normalize_url(url) for url in urls]
    features = feature_frame(normalized)
    numeric = csr_matrix(features.astype(float).to_numpy())
    text = bundle["vectorizer"].transform(normalized)
    matrix = hstack([numeric, text], format="csr")
    return bundle["model"].predict_proba(matrix)[:, 1], features


def domain_disjoint_split(labels: pd.Series, groups: pd.Series, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    folds = max(2, min(10, round(1 / test_size)))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    try:
        for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels, groups):
            if labels.iloc[train_idx].nunique() == 2 and labels.iloc[test_idx].nunique() == 2:
                return train_idx, test_idx
    except ValueError as exc:
        raise ValueError("Not enough varied domains per label for a stratified, domain-disjoint split.") from exc
    raise ValueError("Domain-disjoint split produced one class; add more varied domains or adjust --test-size.")


def train(dataset: Path, model_path: Path, metrics_path: Path, test_size: float = 0.25) -> dict:
    urls, labels, groups = read_dataset(dataset)
    train_idx, test_idx = domain_disjoint_split(labels, groups, test_size)
    train_urls = [urls[index] for index in train_idx]
    test_urls = [urls[index] for index in test_idx]
    x_train, vectorizer = make_matrix(train_urls, fit=True)
    x_test = make_matrix(test_urls, vectorizer)
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    model = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    metrics = {
        "evaluation": "Stratified, domain-group-disjoint held-out split",
        "accuracy": accuracy_score(y_test, predictions), "precision_phishing": precision,
        "recall_phishing": recall, "f1_phishing": f1, "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities), "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)), "train_domains": int(groups.iloc[train_idx].nunique()),
        "test_domains": int(groups.iloc[test_idx].nunique()),
        "domain_overlap": int(len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, predictions, target_names=["legitimate", "phishing"], output_dict=True, zero_division=0),
        "model": "RandomForest (handcrafted URL features + character TF-IDF n-grams)",
        "feature_count": int(x_train.shape[1]),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "vectorizer": vectorizer, "feature_columns": FEATURE_COLUMNS, "version": 2}
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhishGuard AI on a local labeled CSV. No URLs are visited.")
    parser.add_argument("--dataset", type=Path, default=Path("data/sample_urls.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/phishguard_rf.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("models/metrics.json"))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")
    print(json.dumps(train(args.dataset, args.model, args.metrics, args.test_size), indent=2))


if __name__ == "__main__":
    main()
