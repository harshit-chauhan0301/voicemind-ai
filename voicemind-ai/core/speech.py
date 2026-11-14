"""Voice Studio logic: speech-to-text with the PRETRAINED OpenAI Whisper model.

We never train Whisper. We download a ready-made model once and use it for inference.
Whisper needs ffmpeg installed on your computer to read MP3 / M4A files.
"""
from __future__ import annotations

import shutil

from utils.config import WHISPER_MODEL

_models: dict = {}  # cache: loading a model takes seconds, so do it once per model size


def load_whisper(model_name: str = WHISPER_MODEL):
    if model_name not in _models:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("openai-whisper is not installed. Run: pip install openai-whisper") from exc
        _models[model_name] = whisper.load_model(model_name)  # downloads on first use
    return _models[model_name]


def transcribe(audio_path: str, model_name: str = WHISPER_MODEL) -> dict:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Install it and make sure it is on your PATH.")
    model = load_whisper(model_name)
    result = model.transcribe(audio_path, fp16=False)  # fp16=False avoids a CPU warning
    segments = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    text = result["text"].strip()
    return {
        "text": text,
        "language": result.get("language", "unknown"),
        "duration": segments[-1]["end"] if segments else 0.0,
        "segments": segments,
        "model": model_name,
        "words": len(text.split()),
    }
