"""VoiceMind AI: Streamlit entry point.   Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="VoiceMind AI", page_icon="🎙️", layout="wide")

from ui import client  # noqa: E402
from ui.pages import home, nlp_analyzer, project_info, text_summarizer, voice_studio  # noqa: E402
from ui.styles import inject_css  # noqa: E402

PAGES = {
    "🏠  Home": home.render,
    "🎙️  Voice Studio": voice_studio.render,
    "📝  Text Summarizer": text_summarizer.render,
    "🧠  NLP Analyzer": nlp_analyzer.render,
    "ℹ️  Project Information": project_info.render,
}

inject_css()
st.session_state["_page_labels"] = list(PAGES)
st.session_state.setdefault("page", list(PAGES)[0])

with st.sidebar:
    st.markdown('<div class="brand">Voice<span>Mind</span> AI</div>'
                '<div class="tagline">Speech · Summaries · Language intelligence</div>', unsafe_allow_html=True)
    st.radio("Navigation", list(PAGES), key="page", label_visibility="collapsed")
    st.markdown("---")
    if client.api_online():
        st.markdown('<span class="pill ok">● Flask API connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill off">● API offline (direct mode)</span>', unsafe_allow_html=True)
        st.caption("Start it with `python run_api.py` to use the REST backend.")

if flash := st.session_state.pop("_flash", None):
    st.toast(flash, icon="✅")

PAGES[st.session_state["page"]]()
