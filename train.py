"""
train.py
--------
Fake News Detection — Training Pipeline

Trains a Logistic Regression classifier on TF-IDF features derived from
news article author names and titles. Exports the fitted model and
vectorizer for use in the prediction UI (app.py).

Dataset: https://www.kaggle.com/competitions/fake-news/data
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH  = "data/train.csv"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECT_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

RANDOM_STATE = 2
TEST_SIZE    = 0.2

# ── Setup ────────────────────────────────────────────────────────────────────

nltk.download("stopwords", quiet=True)
os.makedirs(MODEL_DIR, exist_ok=True)

port_stem   = PorterStemmer()
STOP_WORDS  = set(stopwords.words("english"))


# ── Preprocessing ─────────────────────────────────────────────────────────────

def stem_text(content: str) -> str:
    """
    Clean and stem a raw text string.

    Steps:
        1. Remove non-alphabetic characters.
        2. Lowercase.
        3. Tokenize and remove stopwords.
        4. Apply Porter stemming.
    """
    content = re.sub(r"[^a-zA-Z]", " ", content)
    tokens  = content.lower().split()
    tokens  = [port_stem.stem(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


def load_and_preprocess(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the CSV dataset, build the content feature, apply stemming,
    and return feature array X and label array Y.
    """
    df = pd.read_csv(path)

    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}\n")

    # Impute missing values with empty strings
    df.fillna("", inplace=True)

    # Combine author and title as the input feature
    df["content"] = df["author"] + " " + df["title"]
    df["content"] = df["content"].apply(stem_text)

    X = df["content"].values
    Y = df["label"].values

    return X, Y


# ── Training ──────────────────────────────────────────────────────────────────

def train(X: np.ndarray, Y: np.ndarray):
    """
    Vectorize text, split data, train Logistic Regression, and report metrics.
    Returns the fitted model and vectorizer.
    """
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_tfidf    = vectorizer.fit_transform(X)

    # Stratified train/test split to preserve class balance
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_tfidf, Y,
        test_size    = TEST_SIZE,
        stratify     = Y,
        random_state = RANDOM_STATE,
    )

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Evaluation
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc  = accuracy_score(Y_test,  model.predict(X_test))

    print(f"[RESULT] Training accuracy : {train_acc:.4f}")
    print(f"[RESULT] Test accuracy     : {test_acc:.4f}\n")
    print("[RESULT] Classification Report (Test Set):")
    print(classification_report(Y_test, model.predict(X_test),
                                target_names=["Real", "Fake"]))

    return model, vectorizer


# ── Export ────────────────────────────────────────────────────────────────────

def save_artifacts(model, vectorizer):
    """Persist the trained model and vectorizer to disk."""
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECT_PATH)
    print(f"[SAVED] Model      → {MODEL_PATH}")
    print(f"[SAVED] Vectorizer → {VECT_PATH}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, Y             = load_and_preprocess(DATA_PATH)
    model, vectorizer = train(X, Y)
    save_artifacts(model, vectorizer)
