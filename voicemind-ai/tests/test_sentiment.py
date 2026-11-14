import pytest

from core.sentiment import LABELS, predict_sentiment, train_and_save


def test_train_returns_metrics(tmp_path):
    m = train_and_save(model_path=tmp_path / "m.joblib")
    assert {r["model"] for r in m["comparison"]} == {"Logistic Regression", "Naive Bayes", "Linear SVM"}
    assert m["labels"] == LABELS


@pytest.mark.parametrize("text,expected", [
    ("I love this, it is absolutely amazing!", "Positive"),
    ("This was terrible and a complete waste of money.", "Negative"),
    ("The meeting is scheduled for Monday at noon.", "Neutral"),
    ("Not good at all.", "Negative"),
])
def test_predictions(text, expected):
    assert predict_sentiment(text)["label"] == expected


def test_probabilities_sum_to_one():
    r = predict_sentiment("Great phone")
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3


def test_empty_text_raises():
    with pytest.raises(ValueError):
        predict_sentiment("   ")
