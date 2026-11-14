import streamlit as st

from ui import client
from ui.components import metric_row, page_header, send_to
from utils.config import ALLOWED_AUDIO_EXTENSIONS, MAX_AUDIO_MB, WHISPER_CHOICES, WHISPER_MODEL


def _fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def render() -> None:
    page_header("🎙️ Voice Studio", "Convert speech to text with pretrained OpenAI Whisper.")

    left, right = st.columns([2, 1])
    with left:
        audio = st.file_uploader(f"Upload audio (max {MAX_AUDIO_MB} MB)",
                                 type=sorted(ALLOWED_AUDIO_EXTENSIONS))
    with right:
        model = st.selectbox("Whisper model", WHISPER_CHOICES, index=WHISPER_CHOICES.index(WHISPER_MODEL),
                             help="tiny = fastest, small = most accurate. Models download once on first use.")

    if audio:
        st.audio(audio)
        if st.button("Transcribe", type="primary"):
            with st.spinner(f"Transcribing with Whisper '{model}'... the first run downloads the model."):
                try:
                    st.session_state["transcript"] = client.transcribe(audio.name, audio.getvalue(), model)
                    st.session_state["transcript"]["filename"] = audio.name
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

    result = st.session_state.get("transcript")
    if not result:
        st.info("Upload an audio file and click **Transcribe**.")
        return

    st.markdown("---")
    metric_row([("Language", result["language"].upper()), ("Duration", _fmt_time(result["duration"])),
                ("Words", result["words"]), ("Model", result["model"])])
    st.markdown("<br>", unsafe_allow_html=True)

    st.text_area("Transcript", value=result["text"], height=240, key="transcript_view")
    stem = result.get("filename", "transcript").rsplit(".", 1)[0]
    a, b, c = st.columns(3)
    a.download_button("⬇ Download transcript", result["text"], file_name=f"{stem}_transcript.txt",
                      mime="text/plain", use_container_width=True)
    b.button("➜ Send to Text Summarizer", on_click=send_to, args=(result["text"], "Text Summarizer"),
             use_container_width=True)
    c.button("➜ Send to NLP Analyzer", on_click=send_to, args=(result["text"], "NLP Analyzer"),
             use_container_width=True)

    with st.expander("Timestamped segments"):
        for seg in result["segments"]:
            st.markdown(f"`{_fmt_time(seg['start'])} → {_fmt_time(seg['end'])}` &nbsp; {seg['text']}")
