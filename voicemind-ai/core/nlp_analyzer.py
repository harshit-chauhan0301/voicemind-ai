"""NLP Analyzer: classic NLP pipeline built on NLTK + spaCy + scikit-learn.

Every step is its own small function so you can explain (and test) each one.
NLTK  -> tokenization, stopwords, stemming, POS tagging, n-grams
spaCy -> lemmatization, named entity recognition
sklearn -> TF-IDF
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from utils.config import SPACY_MODEL
from utils.helpers import clean_text, ensure_nltk_data

# Friendly descriptions for the most common Penn Treebank POS tags
POS_DESCRIPTIONS = {
    "NN": "Noun", "NNS": "Noun (plural)", "NNP": "Proper noun", "NNPS": "Proper noun (plural)",
    "VB": "Verb", "VBD": "Verb (past)", "VBG": "Verb (-ing)", "VBN": "Verb (past participle)",
    "VBP": "Verb (present)", "VBZ": "Verb (3rd person)", "JJ": "Adjective",
    "JJR": "Adjective (comparative)", "JJS": "Adjective (superlative)", "RB": "Adverb",
    "RBR": "Adverb (comparative)", "RBS": "Adverb (superlative)", "PRP": "Pronoun",
    "PRP$": "Possessive pronoun", "DT": "Determiner", "IN": "Preposition", "CC": "Conjunction",
    "CD": "Number", "MD": "Modal verb", "TO": "'to'", "WDT": "Wh-determiner", "WP": "Wh-pronoun",
    "WRB": "Wh-adverb", "EX": "Existential 'there'", "UH": "Interjection", "POS": "Possessive 's",
    "RP": "Particle", "FW": "Foreign word",
}


# ---------------------------------------------------------------- model loading
@lru_cache(maxsize=1)
def get_spacy():
    """Load the spaCy model once. Gives a clear error if it is not installed."""
    import spacy

    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' is not installed. Run:  "
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc


# ---------------------------------------------------------------- individual steps
def sentence_tokenize(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize

    ensure_nltk_data()
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def word_tokenize_text(text: str) -> list[str]:
    from nltk.tokenize import word_tokenize

    ensure_nltk_data()
    return word_tokenize(text)


def clean_words(tokens: list[str]) -> list[str]:
    """Lower-case alphabetic tokens only (drops punctuation and numbers)."""
    return [t.lower() for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", t)]


def remove_stopwords(words: list[str]) -> tuple[list[str], list[str]]:
    """Return (kept_words, removed_stopwords)."""
    from nltk.corpus import stopwords

    ensure_nltk_data()
    stop = set(stopwords.words("english"))
    kept = [w for w in words if w not in stop]
    removed = [w for w in words if w in stop]
    return kept, removed


def stem_words(words: list[str]) -> list[dict]:
    """Porter stemming: crude chopping of word endings (running -> run, studies -> studi)."""
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append({"word": w, "stem": stemmer.stem(w)})
    return out


def lemmatize_text(text: str) -> list[dict]:
    """spaCy lemmatization: dictionary base form (better -> good, mice -> mouse)."""
    doc = get_spacy()(text)
    seen, out = set(), []
    for tok in doc:
        if tok.is_alpha and not tok.is_stop:
            key = tok.text.lower()
            if key not in seen:
                seen.add(key)
                out.append({"word": tok.text, "lemma": tok.lemma_.lower(), "pos": tok.pos_})
    return out


def pos_tag_text(tokens: list[str]) -> tuple[list[dict], dict]:
    """NLTK POS tagging. Returns (tagged tokens, tag counts)."""
    import nltk

    ensure_nltk_data()
    tagged = nltk.pos_tag([t for t in tokens if re.search(r"\w", t)])
    rows = [
        {"word": w, "tag": t, "meaning": POS_DESCRIPTIONS.get(t, t)} for w, t in tagged
    ]
    counts = Counter(r["meaning"] for r in rows)
    return rows, dict(counts.most_common())


def named_entities(text: str) -> list[dict]:
    """spaCy NER: people, places, organisations, dates, money, ..."""
    import spacy

    doc = get_spacy()(text)
    return [
        {"text": ent.text, "label": ent.label_, "meaning": spacy.explain(ent.label_) or ent.label_}
        for ent in doc.ents
    ]


def generate_ngrams(words: list[str], n: int = 2, top_n: int = 15) -> list[dict]:
    from nltk.util import ngrams

    if n < 1 or len(words) < n:
        return []
    counts = Counter(" ".join(g) for g in ngrams(words, n))
    return [{"ngram": g, "count": c} for g, c in counts.most_common(top_n)]


def word_frequency(words: list[str], top_n: int = 15) -> list[dict]:
    return [{"word": w, "count": c} for w, c in Counter(words).most_common(top_n)]


def tfidf_terms(sentences: list[str], top_n: int = 15) -> list[dict]:
    """TF-IDF treating every sentence as a tiny document.

    A term scores high when it is frequent in some sentences but not in all of them.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vec = TfidfVectorizer(stop_words="english", lowercase=True)
        matrix = vec.fit_transform(sentences)
    except ValueError:  # nothing left after stop-word removal
        return []
    scores = matrix.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"term": t, "score": round(float(s), 4)} for t, s in ranked]


def text_statistics(text: str, sentences: list[str], words: list[str]) -> dict:
    unique = set(words)
    return {
        "characters": len(text),
        "characters_no_spaces": len(re.sub(r"\s", "", text)),
        "words": len(words),
        "sentences": len(sentences),
        "unique_words": len(unique),
        "avg_word_length": round(sum(map(len, words)) / len(words), 2) if words else 0,
        "avg_sentence_length": round(len(words) / len(sentences), 2) if sentences else 0,
        "lexical_diversity": round(len(unique) / len(words), 3) if words else 0,
        "reading_time_min": round(len(words) / 200, 1),  # ~200 words per minute
    }


# ---------------------------------------------------------------- full pipeline
def analyze(text: str, ngram_n: int = 2, top_n: int = 15) -> dict:
    """Run the complete pipeline and return one JSON-friendly dictionary."""
    text = clean_text(text)
    sentences = sentence_tokenize(text)
    tokens = word_tokenize_text(text)
    words = clean_words(tokens)
    kept, removed = remove_stopwords(words)
    pos_rows, pos_counts = pos_tag_text(tokens)

    return {
        "stats": text_statistics(text, sentences, words),
        "sentences": sentences,
        "tokens": tokens,
        "words": words,
        "without_stopwords": kept,
        "removed_stopwords": removed,
        "stems": stem_words(kept),
        "lemmas": lemmatize_text(text),
        "pos_tags": pos_rows,
        "pos_counts": pos_counts,
        "entities": named_entities(text),
        "ngram_n": ngram_n,
        "ngrams": generate_ngrams(kept, ngram_n, top_n),
        "word_frequency": word_frequency(kept, top_n),
        "tfidf": tfidf_terms(sentences, top_n),
    }
