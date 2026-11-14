import pandas as pd
import streamlit as st

from ui.components import confusion_matrix_chart, feature_card, page_header, pills


def render() -> None:
    page_header("ℹ️ Project Information", "How VoiceMind AI is built.")

    st.markdown("##### Architecture")
    st.code(
        "Streamlit UI  ──HTTP──▶  Flask REST API  ──▶  core/ (speech · summarizer · nlp · sentiment)\n"
        "      └── falls back to calling core/ directly if the API is offline",
        language="text",
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(feature_card("🎙️", "Speech", "Pretrained OpenAI Whisper (inference only, no training)."),
                unsafe_allow_html=True)
    c2.markdown(feature_card("📝", "Summarization", "Sumy LexRank / TextRank / LSA: extractive, no generation."),
                unsafe_allow_html=True)
    c3.markdown(feature_card("🧠", "NLP + ML", "NLTK, spaCy, TF-IDF and Logistic Regression sentiment."),
                unsafe_allow_html=True)

    st.markdown("<br>##### Sentiment model", unsafe_allow_html=True)
    try:
        from core.sentiment import get_metrics

        m = get_metrics()
    except Exception as exc:  # model could not be loaded/trained
        st.warning(f"Sentiment metrics unavailable: {exc}")
    else:
        st.caption(f"Trained on {m['train_size']} samples, evaluated on {m['test_size']} held-out samples.")
        a, b = st.columns(2)
        with a:
            st.dataframe(pd.DataFrame(m["comparison"]), use_container_width=True, hide_index=True)
        with b:
            confusion_matrix_chart(m["confusion_matrix"], m["labels"])
        st.info(
            "Honest note: the bundled starter dataset is synthetic (template-generated), so scores near 100% do "
            "NOT mean the model is perfect on real-world text. Replace data/sentiment_dataset.csv with a real "
            "labelled dataset and retrain for realistic numbers."
        )

    st.markdown("##### Tech stack")
    pills(["Python", "Streamlit", "Flask", "OpenAI Whisper", "NLTK", "spaCy", "scikit-learn", "Sumy", "Plotly"])
