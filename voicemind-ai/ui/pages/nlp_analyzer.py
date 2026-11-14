import pandas as pd
import streamlit as st

from ui import client
from ui.components import (bar_chart, donut_chart, metric_row, page_header, pills, sentiment_banner,
                           sentiment_chart)
from utils.helpers import decode_text_file


def _table(rows: list[dict], height: int = 300) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=height)
    else:
        st.info("Nothing found.")


def render() -> None:
    page_header("🧠 NLP Analyzer", "Break text down: tokens, grammar, entities, statistics and sentiment.")

    upload = st.file_uploader("Upload a .txt file (optional)", type=["txt"], key="nlp_upload")
    if upload is not None and st.session_state.get("_nlp_file") != upload.file_id:
        st.session_state["_nlp_file"] = upload.file_id
        try:
            st.session_state["nlp_text"] = decode_text_file(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc))

    text = st.text_area("Text to analyze", height=200, key="nlp_text",
                        placeholder="Apple opened a new office in Bangalore in March. Customers loved the launch!")
    n = st.radio("N-gram size", [2, 3], horizontal=True, format_func=lambda x: {2: "Bigrams", 3: "Trigrams"}[x])

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter or upload some text first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    st.session_state["nlp_result"] = client.analyze(text, ngram_n=n)
                    st.session_state["nlp_sentiment"] = client.sentiment(text)
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

    res, sent = st.session_state.get("nlp_result"), st.session_state.get("nlp_sentiment")
    if not res:
        return

    s = res["stats"]
    st.markdown("---")
    metric_row([("Words", s["words"]), ("Sentences", s["sentences"]), ("Unique words", s["unique_words"]),
                ("Read time", f"{s['reading_time_min']} min")])

    tabs = st.tabs(["Sentiment", "Statistics", "Tokens", "Stopwords", "Stem & Lemma", "POS Tags",
                    "Entities", "N-grams", "Frequency", "TF-IDF"])

    with tabs[0]:
        a, b = st.columns([1, 1])
        with a:
            sentiment_banner(sent)
            st.caption("Model: TF-IDF + Logistic Regression (see Project Information for accuracy details).")
        with b:
            sentiment_chart(sent["probabilities"])

    with tabs[1]:
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame({"Metric": [k.replace("_", " ").title() for k in s], "Value": list(s.values())}),
                    use_container_width=True, hide_index=True)
        with b:
            donut_chart(["Content words", "Stopwords"],
                        [len(res["without_stopwords"]), len(res["removed_stopwords"])])

    with tabs[2]:
        st.markdown("**Sentences**")
        for i, sentence in enumerate(res["sentences"], 1):
            st.markdown(f"{i}. {sentence}")
        st.markdown("**Word tokens**")
        pills(res["tokens"][:200])
        if len(res["tokens"]) > 200:
            st.caption(f"Showing 200 of {len(res['tokens'])} tokens.")

    with tabs[3]:
        a, b = st.columns(2)
        a.markdown("**After removal**")
        with a:
            pills(res["without_stopwords"][:150])
        b.markdown("**Removed stopwords**")
        with b:
            pills(sorted(set(res["removed_stopwords"])))

    with tabs[4]:
        a, b = st.columns(2)
        a.markdown("**Stemming (NLTK Porter)**")
        with a:
            _table(res["stems"])
        b.markdown("**Lemmatization (spaCy)**")
        with b:
            _table(res["lemmas"])

    with tabs[5]:
        a, b = st.columns([1, 1])
        with a:
            _table(res["pos_tags"], 340)
        with b:
            bar_chart(list(res["pos_counts"]), list(res["pos_counts"].values()), "Part-of-speech distribution")

    with tabs[6]:
        _table(res["entities"])

    with tabs[7]:
        a, b = st.columns([1, 1])
        with a:
            _table(res["ngrams"], 340)
        with b:
            bar_chart([g["ngram"] for g in res["ngrams"]], [g["count"] for g in res["ngrams"]])

    with tabs[8]:
        bar_chart([w["word"] for w in res["word_frequency"]], [w["count"] for w in res["word_frequency"]],
                  "Most frequent words (stopwords removed)", height=420)

    with tabs[9]:
        st.caption("Each sentence is treated as a document. A higher score means the term is distinctive.")
        bar_chart([t["term"] for t in res["tfidf"]], [t["score"] for t in res["tfidf"]], "Top TF-IDF terms",
                  height=420)
