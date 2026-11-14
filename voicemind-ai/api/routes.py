"""REST API routes. Each route is a thin wrapper: validate input -> call core/ -> return JSON.

POST /api/analyze     {"text": "...", "ngram_n": 2, "top_n": 15}
POST /api/summarize   {"text": "...", "length": "Short|Medium|Long", "algorithm": "lexrank"}
POST /api/sentiment   {"text": "..."}
POST /api/transcribe  multipart form: file=<audio>, model=<tiny|base|small>
GET  /api/health
"""
import os

from flask import Blueprint, jsonify, request

from utils.config import WHISPER_CHOICES, WHISPER_MODEL
from utils.helpers import save_temp_audio

api_bp = Blueprint("api", __name__)


def _json_text() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "text" not in data:
        raise ValueError("Send JSON with a 'text' field.")
    return data


@api_bp.errorhandler(ValueError)
def bad_request(err):
    return jsonify({"error": str(err)}), 400


@api_bp.errorhandler(RuntimeError)
def server_error(err):
    return jsonify({"error": str(err)}), 500


@api_bp.errorhandler(413)
def too_large(_err):
    return jsonify({"error": "Uploaded file is too large."}), 413


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.post("/analyze")
def analyze_route():
    from core import nlp_analyzer

    data = _json_text()
    return jsonify(nlp_analyzer.analyze(data["text"], int(data.get("ngram_n", 2)), int(data.get("top_n", 15))))


@api_bp.post("/summarize")
def summarize_route():
    from core import summarizer

    data = _json_text()
    return jsonify(
        summarizer.summarize(data["text"], data.get("length", "Medium"), data.get("algorithm", "lexrank"))
    )


@api_bp.post("/sentiment")
def sentiment_route():
    from core import sentiment

    return jsonify(sentiment.predict_sentiment(_json_text()["text"]))


@api_bp.post("/transcribe")
def transcribe_route():
    from core import speech

    file = request.files.get("file")
    if file is None or not file.filename:
        raise ValueError("Upload an audio file in the 'file' field.")
    model = request.form.get("model", WHISPER_MODEL)
    if model not in WHISPER_CHOICES:
        raise ValueError(f"model must be one of {WHISPER_CHOICES}")
    path = save_temp_audio(file.filename, file.read())
    try:
        return jsonify(speech.transcribe(path, model))
    finally:
        os.remove(path)
