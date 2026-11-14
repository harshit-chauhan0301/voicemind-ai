"""Small shared helpers used by the API, UI and core modules."""
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from utils.config import ALLOWED_AUDIO_EXTENSIONS, MAX_AUDIO_MB, MAX_TEXT_CHARS

# NLTK resources: {download name: path used to check if it is already installed}
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


@lru_cache(maxsize=1)
def ensure_nltk_data() -> bool:
    """Download any missing NLTK data (only runs once per process)."""
    import nltk

    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
    return True


def clean_text(text: str) -> str:
    """Normalise whitespace and validate length."""
    if text is None:
        raise ValueError("No text provided.")
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        raise ValueError("Text is empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Text is too long (max {MAX_TEXT_CHARS:,} characters).")
    return text


def decode_text_file(raw: bytes) -> str:
    """Decode an uploaded .txt file, trying common encodings."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the text file.")


def validate_audio(filename: str, size_bytes: int) -> str:
    """Return the lower-case extension or raise ValueError."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Allowed: "
            + ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        )
    if size_bytes > MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_AUDIO_MB} MB.")
    return ext


def save_temp_audio(filename: str, data: bytes) -> str:
    """Write audio bytes to a temp file (keeps the extension so ffmpeg can detect it)."""
    ext = validate_audio(filename, len(data))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    tmp.write(data)
    tmp.close()
    return tmp.name


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))
