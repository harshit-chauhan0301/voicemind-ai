"""The UI talks to the Flask API over HTTP.

If the API is not running, it falls back to calling core/ directly, so the app
still works with a single command. The sidebar shows which mode is active.
"""
from __future__ import annotations

import requests

from utils.config import API_URL
from utils.helpers import save_temp_audio

TIMEOUT = 900  # Whisper can be slow on CPU


def api_online() -> bool:
    try:
        return requests.get(f"{API_URL}/api/health", timeout=1.5).ok
    except requests.RequestException:
        return False


def _post(path: str, **kwargs) -> dict | None:
    """POST to the API. Returns None if the API is unreachable, raises on API errors."""
    try:
        resp = requests.post(f"{API_URL}/api{path}", timeout=TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        return None
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"API returned an unexpected response ({resp.status_code}).")
    if resp.status_code >= 400:
        raise RuntimeError(payload.get("error", f"API error {resp.status_code}"))
    return payload


def analyze(text: str, ngram_n: int = 2, top_n: int = 15) -> dict:
    result = _post("/analyze", json={"text": text, "ngram_n": ngram_n, "top_n": top_n})
    if result is None:
        from core import nlp_analyzer

        result = nlp_analyzer.analyze(text, ngram_n, top_n)
    return result


def summarize(text: str, length: str, algorithm: str) -> dict:
    result = _post("/summarize", json={"text": text, "length": length, "algorithm": algorithm})
    if result is None:
        from core import summarizer

        result = summarizer.summarize(text, length, algorithm)
    return result


def sentiment(text: str) -> dict:
    result = _post("/sentiment", json={"text": text})
    if result is None:
        from core import sentiment as sent

        result = sent.predict_sentiment(text)
    return result


def transcribe(filename: str, data: bytes, model: str) -> dict:
    result = _post("/transcribe", files={"file": (filename, data)}, data={"model": model})
    if result is None:
        import os

        from core import speech

        path = save_temp_audio(filename, data)
        try:
            result = speech.transcribe(path, model)
        finally:
            os.remove(path)
    return result
