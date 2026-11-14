"""Sentiment analysis: TF-IDF + Logistic Regression (compared with Naive Bayes and Linear SVM).

How it works, in plain words:
1. TF-IDF turns each text into numbers: words that are frequent in this text but rare
   in other texts get a high weight. We also use word PAIRS (bigrams) so "not good"
   is seen as different from "good".
2. Logistic Regression learns which weighted words point to Negative / Neutral / Positive
   and outputs a probability for each class.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.config import SENTIMENT_DATASET_PATH, SENTIMENT_MODEL_PATH

LABELS = ["Negative", "Neutral", "Positive"]


def _make_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", classifier),
        ]
    )


def _candidates() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }


def load_dataset(path=SENTIMENT_DATASET_PATH) -> pd.DataFrame:
    """CSV with two columns: text,label  (label = Negative / Neutral / Positive)."""
    path = str(path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        from scripts.build_dataset import build

        build(path)
        df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.capitalize()
    return df[df["label"].isin(LABELS)]


def train_and_save(dataset_path=SENTIMENT_DATASET_PATH, model_path=SENTIMENT_MODEL_PATH) -> dict:
    """Train all three models, report their scores, save the Logistic Regression one."""
    df = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    comparison, main_pipeline, main_pred = [], None, None
    for name, clf in _candidates().items():
        pipe = _make_pipeline(clf).fit(x_train, y_train)
        pred = pipe.predict(x_test)
        comparison.append(
            {
                "model": name,
                "accuracy": round(accuracy_score(y_test, pred), 4),
                "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
            }
        )
        if name == "Logistic Regression":
            main_pipeline, main_pred = pipe, pred

    metrics = {
        "comparison": comparison,
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_test, main_pred, labels=LABELS).tolist(),
        "report": classification_report(y_test, main_pred, output_dict=True, zero_division=0),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": main_pipeline, "metrics": metrics}, model_path)
    load_model.cache_clear()
    return metrics


@lru_cache(maxsize=1)
def load_model() -> dict:
    """Load the saved model; train it automatically on first use."""
    try:
        return joblib.load(SENTIMENT_MODEL_PATH)
    except FileNotFoundError:
        train_and_save()
        return joblib.load(SENTIMENT_MODEL_PATH)


def get_metrics() -> dict:
    return load_model()["metrics"]
def _vader_probs(text: str) -> dict[str, float] | None:
    """Use NLTK VADER lexicon to score unseen real-world English words."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        pos, neg, neu, comp = scores["pos"], scores["neg"], scores["neu"], scores["compound"]
        if comp >= 0.05:
            p_pos = min(0.95, 0.55 + comp * 0.4)
            rem = 1.0 - p_pos
            return {"Positive": p_pos, "Neutral": rem * 0.7, "Negative": rem * 0.3}
        elif comp <= -0.05:
            p_neg = min(0.95, 0.55 + abs(comp) * 0.4)
            rem = 1.0 - p_neg
            return {"Negative": p_neg, "Neutral": rem * 0.7, "Positive": rem * 0.3}
        return {"Neutral": max(0.6, neu), "Positive": (1 - max(0.6, neu)) / 2, "Negative": (1 - max(0.6, neu)) / 2}
    except Exception:
        return None


def predict_sentiment(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Text is empty.")
    pipe = load_model()["pipeline"]
    probs = pipe.predict_proba([text])[0]
    ml_scores = {label: float(p) for label, p in zip(pipe.classes_, probs)}

    # Check how many words in the input actually exist in the TF-IDF vocabulary
    tfidf = pipe.named_steps["tfidf"]
    known_features = tfidf.transform([text]).nnz

    vader_scores = _vader_probs(text)
    if vader_scores is not None:
        # If the sentence has few known TF-IDF features or low ML confidence, blend VADER
        ml_weight = 0.75 if known_features >= 3 and max(ml_scores.values()) > 0.65 else 0.35
        v_weight = 1.0 - ml_weight
        scores = {
            lbl: round(ml_weight * ml_scores[lbl] + v_weight * vader_scores[lbl], 4)
            for lbl in LABELS
        }
        # Normalize to sum to 1.0
        total = sum(scores.values())
        scores = {lbl: round(val / total, 4) for lbl, val in scores.items()}
    else:
        scores = {lbl: round(val, 4) for lbl, val in ml_scores.items()}

    label = max(scores, key=scores.get)
    return {"label": label, "confidence": scores[label], "probabilities": scores}


