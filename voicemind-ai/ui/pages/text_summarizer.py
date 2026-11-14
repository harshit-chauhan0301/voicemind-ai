import streamlit as st

from ui import client
from ui.components import bar_chart, donut_chart, metric_row, page_header, pills
from utils.config import SUMMARY_ALGORITHMS, SUMMARY_LENGTHS
from utils.helpers import decode_text_file


def render() -> None:
    page_header("📝 Text Summarizer", "Extractive summarization: keeps the most important sentences, word for word.")

    upload = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
    if upload is not None and st.session_state.get("_summ_file") != upload.file_id:
        st.session_state["_summ_file"] = upload.file_id
        try:
            st.session_state["summ_text"] = decode_text_file(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc))

    text = st.text_area("Or paste your text here", height=260, key="summ_text",
                        placeholder="Paste an article, essay or transcript...")

    c1, c2 = st.columns(2)
    length = c1.select_slider("Summary length", options=list(SUMMARY_LENGTHS), value="Medium")
    algo_label = c2.selectbox("Algorithm", list(SUMMARY_ALGORITHMS),
                              help="LexRank/TextRank rank sentences like PageRank. LSA finds topic-covering sentences.")

    if st.button("Summarize", type="primary"):
        if not text.strip():
            st.warning("Please enter or upload some text first.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    st.session_state["summary"] = client.summarize(text, length, SUMMARY_ALGORITHMS[algo_label])
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

    res = st.session_state.get("summary")
    if not res:
        return

    st.markdown("---")
    metric_row([("Original words", res["words_original"]), ("Summary words", res["words_summary"]),
                ("Reduction", f"{res['reduction_percent']}%"), ("Compression", f"{res['compression_ratio']}:1")])
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("##### Summary")
    st.markdown(f'<div class="summary-box">{res["summary"]}</div>', unsafe_allow_html=True)
    st.download_button("⬇ Download summary", res["summary"], file_name="summary.txt", mime="text/plain")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("##### Keywords")
        pills([k["keyword"] for k in res["keywords"]])
        bar_chart([k["keyword"] for k in res["keywords"]], [k["score"] for k in res["keywords"]], height=300)
    with right:
        st.markdown("##### Original vs summary")
        donut_chart(["Kept in summary", "Removed"],
                    [res["words_summary"], res["words_original"] - res["words_summary"]])
