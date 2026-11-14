"""One-time setup: python -m scripts.setup_nlp_data
Downloads NLTK data, the spaCy model, and trains the sentiment model."""
import subprocess
import sys

from utils.config import SPACY_MODEL
from utils.helpers import ensure_nltk_data

if __name__ == "__main__":
    print("1/3 Downloading NLTK data...")
    ensure_nltk_data()
    print(f"2/3 Downloading spaCy model {SPACY_MODEL}...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
    print("3/3 Training sentiment model...")
    from core.sentiment import train_and_save

    train_and_save()
    print("Setup complete.")
