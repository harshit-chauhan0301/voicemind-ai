"""Reusable UI pieces: cards, metric tiles, charts, shared-state helpers."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

PURPLE, BLUE = "#8b5cf6", "#3b82f6"
SENTIMENT_COLORS = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}


# ------------------------------------------------------------------ layout bits
def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def feature_card(icon: str, title: str, text: str) -> str:
    return f'<div class="card"><div class="icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>'


def metric_row(items: list[tuple[str, object]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>',
                     unsafe_allow_html=True)


def pills(values: list[str]) -> None:
    st.markdown("".join(f'<span class="pill">{v}</span>' for v in values), unsafe_allow_html=True)


def sentiment_banner(result: dict) -> None:
    emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}[result["label"]]
    st.markdown(f'<div class="sent {result["label"]}">{emoji} {result["label"]} '
                f'<span style="font-weight:500;font-size:.95rem">&nbsp;({result["confidence"]:.0%} confidence)</span></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ charts
def _style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8f5", family="Inter"),
        xaxis=dict(gridcolor="#26264a"), yaxis=dict(gridcolor="#26264a"), showlegend=False,
    )
    return fig


def bar_chart(labels: list, values: list, title: str = "", horizontal: bool = True, height: int = 340) -> None:
    if not labels:
        st.info("Nothing to chart yet.")
        return
    if horizontal:  # biggest bar on top
        labels, values = labels[::-1], values[::-1]
    fig = go.Figure(go.Bar(
        x=values if horizontal else labels, y=labels if horizontal else values,
        orientation="h" if horizontal else "v",
        marker=dict(color=values, colorscale=[[0, BLUE], [1, PURPLE]]),
    ))
    fig.update_layout(title=title)
    st.plotly_chart(_style(fig, height), use_container_width=True, config={"displayModeBar": False})


def donut_chart(labels: list, values: list, title: str = "", colors: list | None = None) -> None:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, sort=False,
                           marker=dict(colors=colors or [PURPLE, BLUE, "#6366f1", "#a78bfa", "#60a5fa", "#818cf8"],
                                       line=dict(color="#0b0b14", width=2)),
                           textinfo="percent", textfont=dict(color="#fff")))
    fig.update_layout(title=title)
    fig = _style(fig, 340)
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def sentiment_chart(probabilities: dict) -> None:
    labels = ["Positive", "Neutral", "Negative"]
    values = [probabilities.get(l, 0) for l in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=[SENTIMENT_COLORS[l] for l in labels]),
                           text=[f"{v:.0%}" for v in values], textposition="outside"))
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
    st.plotly_chart(_style(fig, 300), use_container_width=True, config={"displayModeBar": False})


def confusion_matrix_chart(matrix: list, labels: list) -> None:
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, colorscale=[[0, "#15152a"], [1, PURPLE]],
                               text=matrix, texttemplate="%{text}", showscale=False))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(_style(fig, 320), use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------------ cross-module state
def send_to(text: str, page_label: str) -> None:
    """Callback for 'send transcript to ...' buttons. Runs BEFORE the next rerun renders widgets."""
    if page_label == "Text Summarizer":
        st.session_state["summ_text"] = text
    else:
        st.session_state["nlp_text"] = text
    st.session_state["page"] = next(p for p in st.session_state["_page_labels"] if page_label in p)
    st.session_state["_flash"] = f"Text sent to {page_label}."
