"""Global dark theme (purple + blue accents). Injected once from app.py."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0b0b14; --panel: #13132299; --card: #15152a; --border: #26264a;
  --text: #e8e8f5; --muted: #9a9ab8;
  --purple: #8b5cf6; --blue: #3b82f6; --grad: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
}
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp { background: radial-gradient(1200px 600px at 80% -10%, #2a1a5e55, transparent),
                     radial-gradient(900px 500px at -10% 10%, #12306b44, transparent), var(--bg); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0e0e1c; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  background: transparent; border: 1px solid transparent; border-radius: 12px;
  padding: 10px 14px; transition: all .2s ease; width: 100%; cursor: pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  background: #1b1b36; border-color: var(--border); transform: translateX(3px); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(135deg, #8b5cf633, #3b82f633); border-color: #8b5cf6aa; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
.brand { font-size: 1.5rem; font-weight: 800; letter-spacing: -.5px; margin-bottom: .2rem; }
.brand span, .grad-text { background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.tagline { color: var(--muted); font-size: .8rem; margin-bottom: 1.2rem; }

/* hero + cards */
.hero { background: linear-gradient(135deg, #1d1540, #121a3d); border: 1px solid var(--border);
        border-radius: 24px; padding: 2.6rem 2.4rem; margin-bottom: 1.6rem; animation: fadeUp .6s ease both; }
.hero h1 { font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; margin: 0 0 .6rem; }
.hero p { color: var(--muted); font-size: 1.05rem; max-width: 680px; margin: 0; }
.page-title { font-size: 2rem; font-weight: 800; letter-spacing: -.5px; margin-bottom: .1rem; }
.page-sub { color: var(--muted); margin-bottom: 1.4rem; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 1.3rem 1.4rem;
        height: 100%; transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
        animation: fadeUp .6s ease both; }
.card:hover { transform: translateY(-4px); border-color: #8b5cf6; box-shadow: 0 12px 30px #8b5cf622; }
.card h3 { margin: .3rem 0 .4rem; font-size: 1.1rem; }
.card p { color: var(--muted); font-size: .92rem; margin: 0; }
.icon { width: 44px; height: 44px; border-radius: 12px; background: var(--grad); display: flex;
        align-items: center; justify-content: center; font-size: 1.3rem; }
.metric { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.2rem;
          animation: fadeUp .5s ease both; }
.metric .label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric .value { font-size: 1.7rem; font-weight: 800; margin-top: .15rem; }
.pill { display: inline-block; padding: .25rem .75rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
        background: #8b5cf622; border: 1px solid #8b5cf666; margin: 0 .35rem .35rem 0; }
.pill.ok { background: #22c55e22; border-color: #22c55e66; color: #86efac; }
.pill.off { background: #ef444422; border-color: #ef444466; color: #fca5a5; }
.sent { border-radius: 16px; padding: 1.1rem 1.4rem; font-size: 1.3rem; font-weight: 700; border: 1px solid; }
.sent.Positive { background: #22c55e1a; border-color: #22c55e66; color: #86efac; }
.sent.Negative { background: #ef44441a; border-color: #ef444466; color: #fca5a5; }
.sent.Neutral { background: #94a3b81a; border-color: #94a3b866; color: #cbd5e1; }
.summary-box { background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--purple);
               border-radius: 14px; padding: 1.2rem 1.4rem; line-height: 1.7; }

/* widgets */
.stButton > button, .stDownloadButton > button { border-radius: 12px; border: 1px solid var(--border);
  background: #1a1a33; color: var(--text); font-weight: 600; padding: .55rem 1.2rem; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--purple); transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: var(--grad); border: none; color: white; }
.stButton > button[kind="primary"]:hover { box-shadow: 0 8px 24px #8b5cf655; }
.stTextArea textarea, .stTextInput input { background: #101022 !important; border-radius: 14px !important;
  border: 1px solid var(--border) !important; color: var(--text) !important; }
.stTextArea textarea:focus { border-color: var(--purple) !important; box-shadow: 0 0 0 1px var(--purple) !important; }
[data-testid="stFileUploader"] section { background: #101022; border: 1px dashed #8b5cf688; border-radius: 16px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #15152a; border-radius: 10px; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #8b5cf633, #3b82f633); }
.stTabs [data-baseweb="tab-highlight"] { display: none; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
hr { border-color: var(--border); }

@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@media (max-width: 768px) { .hero h1 { font-size: 1.8rem; } .hero { padding: 1.6rem 1.2rem; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
