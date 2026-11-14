import streamlit as st

from ui.components import feature_card, pills


def render() -> None:
    st.markdown(
        '<div class="hero"><h1>Turn speech into <span class="grad-text">insight</span>.</h1>'
        "<p>VoiceMind AI transcribes audio with Whisper, condenses long text into clear summaries, "
        "and breaks language down with a full NLP toolkit, all in one workspace.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(feature_card("🎙️", "Voice Studio",
                             "Upload WAV, MP3 or M4A and get an accurate transcript from pretrained Whisper."),
                unsafe_allow_html=True)
    c2.markdown(feature_card("📝", "Text Summarizer",
                             "Extractive summaries with Sumy, adjustable length, keywords and compression stats."),
                unsafe_allow_html=True)
    c3.markdown(feature_card("🧠", "NLP Analyzer",
                             "Tokens, POS tags, entities, n-grams, TF-IDF, statistics and sentiment in one view."),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### How it flows")
    st.markdown(
        '<div class="card"><p>1. Upload audio in <b>Voice Studio</b> &rarr; 2. Send the transcript to the '
        "<b>Summarizer</b> or <b>NLP Analyzer</b> with one click &rarr; 3. Download results as text files. "
        "Each module also works on its own with pasted text.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Built with")
    pills(["Python", "Streamlit", "Flask", "OpenAI Whisper", "NLTK", "spaCy", "Scikit-learn", "Sumy", "Plotly"])
