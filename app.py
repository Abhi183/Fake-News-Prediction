"""
app.py
------
Fake News Detection — Streamlit Prediction Interface

Run with:  streamlit run app.py
Requires:  python train.py  (generates models/)
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "best_model.pkl")
VECT_PATH    = os.path.join("models", "tfidf_vectorizer.pkl")
NAME_PATH    = os.path.join("models", "best_model_name.pkl")
RESULTS_PATH = os.path.join("models", "model_results.csv")
DATA_PATH    = os.path.join("data",   "fake_news_dataset.csv")

port_stem  = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .big-title { font-size:2.2rem; font-weight:700; color:#1a1a2e; margin-bottom:0; }
    .subtitle  { font-size:1.05rem; color:#555; margin-top:0; margin-bottom:1.5rem; }
    .result-real {
        background:#d4edda; border-left:6px solid #28a745; border-radius:10px;
        padding:1.2rem 1.5rem; font-size:1.3rem; font-weight:700; color:#155724;
    }
    .result-fake {
        background:#f8d7da; border-left:6px solid #dc3545; border-radius:10px;
        padding:1.2rem 1.5rem; font-size:1.3rem; font-weight:700; color:#721c24;
    }
    .section-hdr {
        font-size:1.15rem; font-weight:600; border-bottom:2px solid #dc3545;
        padding-bottom:4px; margin:1.2rem 0 0.8rem; color:#1a1a2e;
    }
    .sample-card {
        background:#f1f3f4; border-radius:8px; padding:0.8rem 1rem;
        margin-bottom:0.5rem; cursor:pointer; font-size:0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, "Unknown"
    model     = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    name      = joblib.load(NAME_PATH) if os.path.exists(NAME_PATH) else "Model"
    return model, vectorizer, name

model, vectorizer, model_name = load_artifacts()

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(title: str, text: str = "") -> str:
    content = title + " " + text
    content = re.sub(r"[^a-zA-Z]", " ", content)
    tokens  = content.lower().split()
    tokens  = [port_stem.stem(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

def predict(title: str, text: str = ""):
    cleaned = preprocess(title, text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    try:
        prob = model.predict_proba(vec)[0]
    except AttributeError:
        prob = np.array([1 - pred, pred], dtype=float)
    return int(pred), prob   # 0=FAKE, 1=REAL

# ── Sample articles ────────────────────────────────────────────────────────────
SAMPLES = [
    {
        "label": "REAL",
        "title": "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns",
        "text":  "The Federal Reserve raised its benchmark interest rate by a quarter percentage point on Wednesday, continuing its campaign to bring inflation down to its 2 percent target.",
    },
    {
        "label": "FAKE",
        "title": "Scientists confirm chemtrails contain mind-control nanobots planted by governments",
        "text":  "A leaked document from a secret government agency has revealed that the white trails left by aircraft contain microscopic robots designed to make the population more docile and obedient.",
    },
    {
        "label": "REAL",
        "title": "NASA's James Webb Space Telescope captures deepest infrared image of universe",
        "text":  "NASA released the deepest and sharpest infrared image of the distant universe ever taken, showing thousands of galaxies that have never been seen before.",
    },
    {
        "label": "FAKE",
        "title": "Doctors baffled as local man cures cancer with essential oils and prayer alone",
        "text":  "A man from rural Ohio claims he cured his stage-4 cancer by diffusing lavender oil and reciting Bible verses, prompting doctors across the country to investigate.",
    },
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="big-title">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">NLP-powered misinformation classifier using TF-IDF and ensemble machine learning. '
            'Trained on 40,587 labeled news articles.</p>', unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Info")
    st.markdown(f"**Algorithm**: {model_name}")
    st.markdown("**Vectorizer**: TF-IDF (50k features, bigrams)")
    st.markdown("**Dataset**: 40,587 articles (FAKE + REAL)")
    st.markdown("**Train/Test Split**: 80 / 20")

    if os.path.exists(RESULTS_PATH):
        st.markdown("---")
        st.markdown("### Model Leaderboard")
        df_res = pd.read_csv(RESULTS_PATH)
        st.dataframe(
            df_res[["Model", "Test Accuracy", "F1 Score"]].set_index("Model"),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### Sample Articles")
    st.caption("Click a sample to auto-fill the form.")
    for i, s in enumerate(SAMPLES):
        badge = "🟢" if s["label"] == "REAL" else "🔴"
        if st.button(f"{badge} {s['title'][:55]}...", key=f"sample_{i}", use_container_width=True):
            st.session_state["title_input"] = s["title"]
            st.session_state["text_input"]  = s["text"]

    st.markdown("---")
    st.caption("Chicco & Jurman (2020) | GonzaloA/fake_news dataset (HuggingFace)")

# ── Main columns ───────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<p class="section-hdr">Article Input</p>', unsafe_allow_html=True)

    title_val = st.session_state.get("title_input", "")
    text_val  = st.session_state.get("text_input",  "")

    title = st.text_input(
        "Headline / Title *",
        value=title_val,
        placeholder="e.g. Scientists discover breakthrough cancer treatment",
    )
    text = st.text_area(
        "Article Body (optional — improves accuracy)",
        value=text_val,
        height=180,
        placeholder="Paste the article body here for a more confident prediction...",
    )

    col_a, col_b = st.columns(2)
    run = col_a.button("Analyze", use_container_width=True, type="primary")
    if col_b.button("Clear", use_container_width=True):
        st.session_state["title_input"] = ""
        st.session_state["text_input"]  = ""
        st.rerun()

with right:
    st.markdown('<p class="section-hdr">Prediction</p>', unsafe_allow_html=True)

    if run or "last_title" in st.session_state:
        if run:
            if not title.strip():
                st.warning("Please enter at least a headline.")
                st.stop()
            st.session_state["last_title"] = title
            st.session_state["last_text"]  = text

        t = st.session_state.get("last_title", title)
        x = st.session_state.get("last_text",  text)

        pred, prob = predict(t, x)
        p_fake = prob[0]
        p_real = prob[1]

        if pred == 1:
            st.markdown(
                f'<div class="result-real">✅ REAL NEWS<br>'
                f'<span style="font-size:0.95rem;font-weight:400;">Confidence: {p_real*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-fake">🚨 FAKE NEWS<br>'
                f'<span style="font-size:0.95rem;font-weight:400;">Confidence: {p_fake*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(5, 1.6))
        ax.barh(["REAL", "FAKE"], [p_real, p_fake],
                color=["#28a745", "#dc3545"], edgecolor="white", height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False, labelsize=9)
        ax.set_xlabel("Probability", fontsize=9)
        for i, v in enumerate([p_real, p_fake]):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown('<p class="section-hdr">Input Summary</p>', unsafe_allow_html=True)
        st.markdown(f"**Headline**: {t}")
        word_count = len(t.split()) + len(x.split()) if x.strip() else len(t.split())
        st.caption(f"{word_count} words analyzed | Model: {model_name}")

    else:
        st.info("Enter a headline (and optionally the article body) then click **Analyze**.")
        st.markdown("""
**Tips for best results:**
- Include both headline and article body when possible
- The model uses TF-IDF bigrams on stemmed text
- Use the sidebar samples to quickly test the model
        """)

# ── Dataset stats section ──────────────────────────────────────────────────────
with st.expander("Dataset Overview", expanded=False):
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, usecols=["label", "label_name"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Articles", f"{len(df):,}")
        c2.metric("FAKE Articles",  f"{(df.label==0).sum():,}")
        c3.metric("REAL Articles",  f"{(df.label==1).sum():,}")

        fig2, ax2 = plt.subplots(figsize=(4, 2.5))
        counts = df["label_name"].value_counts()
        ax2.bar(counts.index, counts.values, color=["#dc3545", "#28a745"], edgecolor="white")
        ax2.set_title("Class Distribution", fontsize=11)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_ylabel("Count")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=False)
    else:
        st.info("Dataset not available. Run `python train.py` to generate it.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Disclaimer**: This tool is for educational purposes only and may produce incorrect predictions. "
    "Always verify news from authoritative sources.  "
    f"Model: {model_name} | Dataset: GonzaloA/fake_news (HuggingFace)"
)
