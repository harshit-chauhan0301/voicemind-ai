import pytest

pytest.importorskip("nltk")
pytest.importorskip("spacy")

from core import nlp_analyzer as nlp  # noqa: E402

TEXT = ("Apple opened a new office in Bangalore in March. Customers loved the launch! "
        "The running dogs were barking loudly.")


def test_full_pipeline_shapes():
    r = nlp.analyze(TEXT)
    assert r["stats"]["sentences"] == 3
    assert r["stats"]["words"] > 10
    assert any(e["label"] in {"ORG", "GPE"} for e in r["entities"])
    assert r["pos_tags"] and r["tfidf"] and r["word_frequency"]


def test_stopwords_removed():
    kept, removed = nlp.remove_stopwords(["the", "dogs", "are", "running"])
    assert kept == ["dogs", "running"] and removed == ["the", "are"]


def test_stemming_and_lemma():
    assert nlp.stem_words(["running"])[0]["stem"] == "run"
    assert {l["lemma"] for l in nlp.lemmatize_text("The dogs were barking")} >= {"dog", "bark"}


def test_ngrams():
    assert nlp.generate_ngrams(["a", "b", "c"], 2)[0]["count"] == 1
    assert nlp.generate_ngrams(["a"], 2) == []
