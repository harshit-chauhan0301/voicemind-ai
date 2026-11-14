"""Central configuration. Change values here (or via environment variables)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SENTIMENT_DATASET_PATH = DATA_DIR / "sentiment_dataset.csv"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"

# Flask API
API_HOST = os.getenv("VOICEMIND_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("VOICEMIND_API_PORT", "5000"))
API_URL = os.getenv("VOICEMIND_API_URL", f"http://{API_HOST}:{API_PORT}")

# NLP
SPACY_MODEL = "en_core_web_sm"

# Speech recognition
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny | base | small | medium
WHISPER_CHOICES = ["tiny", "base", "small"]
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a"}
MAX_AUDIO_MB = 50

# Summarizer: fraction of the original sentences kept
SUMMARY_LENGTHS = {"Short": 0.20, "Medium": 0.35, "Long": 0.50}
SUMMARY_ALGORITHMS = {"LexRank": "lexrank", "TextRank": "textrank", "LSA": "lsa"}

MAX_TEXT_CHARS = 100_000
