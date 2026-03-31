"""
app.py
------
Fake News Detection — Streamlit Prediction Interface

Loads the pre-trained Logistic Regression model and TF-IDF vectorizer,
accepts user input (author + headline), and predicts whether the article
is real or fake.

Run with:
    streamlit run app.py
"""

import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ── Setup ─────────────────────────────────────────────────────────────────────

nltk.download("stopwords", quiet=True)

MODEL_PATH = "model/fake_news_model.pkl"
VECT_PATH  = "model/tfidf_vectorizer.pkl"

port_stem  = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


# ── Load Artifacts ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_vectorizer():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(author: str, title: str) -> str:
    """Apply the same stemming pipeline used during training."""
    content = author + " " + title
    content = re.sub(r"[^a-zA-Z]", " ", content)
    tokens  = content.lower().split()
    tokens  = [port_stem.stem(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title = "Fake News Detector",
        page_icon  = "📰",
        layout     = "centered",
    )

    # Header
    st.title("📰 Fake News Detector")
    st.markdown(
        "Enter the **author** and **headline** of a news article. "
        "The model will predict whether it is likely real or fake."
    )
    st.divider()

    # Load model
    try:
        model, vectorizer = load_model_and_vectorizer()
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found. "
            "Please run `python train.py` first to generate `model/fake_news_model.pkl` "
            "and `model/tfidf_vectorizer.pkl`."
        )
        st.stop()

    # Input fields
    author = st.text_input(
        "Author Name",
        placeholder="e.g. Jane Doe",
        help="Full name of the article's author.",
    )
    title = st.text_area(
        "Article Headline",
        placeholder="e.g. Scientists discover new treatment for common cold",
        height=100,
        help="The headline or title of the news article.",
    )

    st.divider()

    # Prediction
    if st.button("🔍 Analyze Article", use_container_width=True):
        if not title.strip():
            st.warning("Please enter at least the article headline.")
        else:
            content_clean = preprocess(author, title)
            X_input       = vectorizer.transform([content_clean])
            prediction    = model.predict(X_input)[0]
            confidence    = model.predict_proba(X_input)[0]

            real_conf = confidence[0] * 100
            fake_conf = confidence[1] * 100

            st.subheader("Prediction Result")

            if prediction == 0:
                st.success("✅ This article appears to be **REAL**.")
            else:
                st.error("🚨 This article appears to be **FAKE**.")

            # Confidence breakdown
            col1, col2 = st.columns(2)
            col1.metric("Real confidence",  f"{real_conf:.1f}%")
            col2.metric("Fake confidence",  f"{fake_conf:.1f}%")

            st.progress(int(fake_conf), text=f"Fake likelihood: {fake_conf:.1f}%")

    # Footer
    st.divider()
    st.caption(
        "Model: Logistic Regression on TF-IDF features | "
        "Dataset: [Kaggle Fake News](https://www.kaggle.com/competitions/fake-news/data) | "
        "Test accuracy: 97.9%"
    )


if __name__ == "__main__":
    main()
