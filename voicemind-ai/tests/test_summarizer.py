import pytest

pytest.importorskip("sumy")
pytest.importorskip("nltk")

from core.summarizer import summarize  # noqa: E402

ARTICLE = (
    "Machine learning is a field of artificial intelligence. It lets computers learn from data. "
    "Many companies use machine learning to improve products. Cats are popular pets around the world. "
    "Deep learning is a subset of machine learning that uses neural networks. "
    "Neural networks are inspired by the human brain. Data quality matters more than model size. "
    "Good data leads to better machine learning models. Researchers publish new methods every year. "
    "Students can learn machine learning with free online courses."
)


def test_summary_is_shorter():
    r = summarize(ARTICLE, "Short")
    assert r["words_summary"] < r["words_original"]
    assert 0 < r["reduction_percent"] < 100 and r["keywords"]


def test_longer_setting_gives_more_sentences():
    assert summarize(ARTICLE, "Long")["sentences_summary"] > summarize(ARTICLE, "Short")["sentences_summary"]


def test_bad_input():
    with pytest.raises(ValueError):
        summarize("   ")
