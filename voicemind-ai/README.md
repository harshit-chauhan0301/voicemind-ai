# VoiceMind AI

AI-powered speech recognition, text summarization and NLP analysis, in one dark-themed Streamlit dashboard backed by a Flask REST API.

| Module | What it does | Main tools |
|---|---|---|
| Voice Studio | WAV/MP3/M4A to text, download transcript, send to other modules | OpenAI Whisper (pretrained) |
| Text Summarizer | Extractive summary, length control, keywords, compression ratio | Sumy (LexRank/TextRank/LSA), scikit-learn |
| NLP Analyzer | Tokens, stopwords, stem/lemma, POS, NER, n-grams, frequency, TF-IDF, stats, sentiment | NLTK, spaCy, scikit-learn |

## Setup (about 10 minutes)

**Use Python 3.11 or 3.12.** Python 3.13 is too new for some of the PyTorch/Whisper dependencies.

```bash
# 1. virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. dependencies
pip install -r requirements.txt

# 3. ffmpeg (needed for MP3/M4A)
#    Windows: winget install ffmpeg   |  macOS: brew install ffmpeg  |  Linux: sudo apt install ffmpeg

# 4. one-time data setup: NLTK data + spaCy model + trains the sentiment model
python -m scripts.setup_nlp_data
```

## Run

Two terminals (both with the virtual environment activated):

```bash
python run_api.py          # terminal 1: Flask API on http://127.0.0.1:5000
streamlit run app.py       # terminal 2: dashboard on http://localhost:8501
```

The dashboard also works without the API (the sidebar shows "direct mode"), but running both shows the full architecture.

Run the tests: `python -m pytest -q`

## Project structure

```
app.py            Streamlit entry point (sidebar + page routing)
run_api.py        Flask entry point
api/              Flask app factory + REST routes (thin wrappers)
core/             All logic, no UI/Flask code
  nlp_analyzer.py   speech.py   summarizer.py   sentiment.py
ui/               styles.py (theme), components.py (cards/charts), client.py (API client), pages/
utils/            config.py, helpers.py
scripts/          build_dataset.py, train_sentiment.py, setup_nlp_data.py
data/ models/ tests/
```

## REST API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /api/health` | | `{"status":"ok"}` |
| `POST /api/analyze` | `{"text","ngram_n","top_n"}` | full NLP analysis |
| `POST /api/summarize` | `{"text","length":"Short/Medium/Long","algorithm":"lexrank"}` | summary, keywords, compression |
| `POST /api/sentiment` | `{"text"}` | label, confidence, probabilities |
| `POST /api/transcribe` | multipart `file`, `model` | transcript, language, segments |

```bash
curl -X POST http://127.0.0.1:5000/api/sentiment -H "Content-Type: application/json" -d '{"text":"I love this!"}'
```

## How the main parts work (interview notes)

- **Whisper**: a pretrained speech-to-text transformer. We only run inference, so no training. Model sizes trade speed for accuracy (tiny < base < small).
- **Extractive summarization**: LexRank builds a graph where sentences are nodes and similarity is the edge weight, then ranks sentences like PageRank. We keep the top N in original order. Nothing is rewritten.
- **TF-IDF**: term frequency × inverse document frequency. It gives high weight to words that are frequent in one text but rare across others.
- **Sentiment**: TF-IDF (unigrams + bigrams) feeds Logistic Regression, which outputs a probability for each of Negative/Neutral/Positive. Naive Bayes and Linear SVM are trained alongside it for comparison, and the results appear on the Project Information page.
- **Stemming vs lemmatization**: stemming chops endings (`studies` → `studi`); lemmatization returns a real dictionary form (`studies` → `study`).
- **Design**: `core/` has no UI code, so the UI, the API and the tests all reuse the same functions.

## Known limitations (be upfront about these)

- The bundled sentiment dataset is **synthetic**. Scores near 100% on it do not predict real-world accuracy. For a real project, swap `data/sentiment_dataset.csv` (columns `text,label`) for a real labelled dataset and run `python -m scripts.train_sentiment`.
- Whisper on CPU is slow for long audio; use the `tiny` or `base` model.
- Extractive summaries can feel choppy because sentences are copied unchanged.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg was not found` | Install ffmpeg and restart the terminal so PATH updates |
| `spaCy model 'en_core_web_sm' is not installed` | `python -m spacy download en_core_web_sm` |
| NLTK `LookupError` | `python -m scripts.setup_nlp_data` |
| Whisper install fails | Use Python 3.11/3.12; install CPU PyTorch first from pytorch.org |
| Port 5000 busy (macOS AirPlay) | `set VOICEMIND_API_PORT=5001` (Windows) or `export VOICEMIND_API_PORT=5001` (other), in both terminals |
