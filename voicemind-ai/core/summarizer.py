"""Text Summarizer: extractive summarization with Sumy.

Extractive = we PICK the most important existing sentences (no new text is written).
LexRank / TextRank rank sentences like Google PageRank ranks web pages.
LSA uses linear algebra to find sentences covering the main topics.
"""
from __future__ import annotations

from utils.config import SUMMARY_ALGORITHMS, SUMMARY_LENGTHS
from utils.helpers import clean_text, ensure_nltk_data, word_count


def _build_summarizer(name: str):
    from sumy.nlp.stemmers import Stemmer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.utils import get_stop_words

    classes = {"lexrank": LexRankSummarizer, "textrank": TextRankSummarizer, "lsa": LsaSummarizer}
    summarizer = classes[name](Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")
    return summarizer


def extract_keywords(text: str, top_n: int = 10) -> list[dict]:
    """Top keywords by TF-IDF score (each sentence = one document)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences = [s for s in text.replace("\n", " ").split(".") if s.strip()] or [text]
    try:
        vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b")
        matrix = vec.fit_transform(sentences)
    except ValueError:
        return []
    scores = matrix.sum(axis=0).A1
    ranked = sorted(zip(vec.get_feature_names_out(), scores), key=lambda x: x[1], reverse=True)
    return [{"keyword": k, "score": round(float(s), 3)} for k, s in ranked[:top_n]]


def summarize(text: str, length: str = "Medium", algorithm: str = "lexrank", num_keywords: int = 10) -> dict:
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser

    text = clean_text(text)
    if length not in SUMMARY_LENGTHS:
        raise ValueError(f"length must be one of {list(SUMMARY_LENGTHS)}")
    algorithm = SUMMARY_ALGORITHMS.get(algorithm, algorithm)
    if algorithm not in SUMMARY_ALGORITHMS.values():
        raise ValueError(f"algorithm must be one of {list(SUMMARY_ALGORITHMS.values())}")

    ensure_nltk_data()  # Sumy's tokenizer uses NLTK's punkt data
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    total = len(parser.document.sentences)
    if total == 0:
        raise ValueError("No sentences found in the text.")

    target = min(total, max(1, round(total * SUMMARY_LENGTHS[length])))
    chosen = _build_summarizer(algorithm)(parser.document, target)  # returned in document order
    summary = " ".join(str(s) for s in chosen)

    original_words, summary_words = word_count(text), word_count(summary)
    reduction = round((1 - summary_words / original_words) * 100, 1) if original_words else 0.0
    return {
        "summary": summary,
        "sentences_original": total,
        "sentences_summary": len(chosen),
        "words_original": original_words,
        "words_summary": summary_words,
        "reduction_percent": reduction,  # 70 means the summary is 70% shorter
        "compression_ratio": round(original_words / summary_words, 2) if summary_words else 0.0,  # 3.5 means 3.5:1
        "keywords": extract_keywords(text, num_keywords),
        "length": length,
        "algorithm": algorithm,
    }
