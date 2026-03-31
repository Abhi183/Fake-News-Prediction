"""
train.py
--------
Trains and evaluates multiple classifiers for fake news detection.
Saves the best model and TF-IDF vectorizer to disk.

Usage:
    python train.py

Outputs:
    models/best_model.pkl
    models/tfidf_vectorizer.pkl
    models/model_results.csv
"""

import os
import re
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score
)

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "fake_news_dataset.csv")
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pkl")
VECT_PATH    = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
NAME_PATH    = os.path.join(MODEL_DIR, "best_model_name.pkl")
RESULTS_PATH = os.path.join(MODEL_DIR, "model_results.csv")

RANDOM_STATE = 42
TEST_SIZE    = 0.20
MAX_FEATURES = 50000

os.makedirs(MODEL_DIR, exist_ok=True)
nltk.download("stopwords", quiet=True)

port_stem  = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text   = re.sub(r"[^a-zA-Z]", " ", str(text))
    tokens = text.lower().split()
    tokens = [port_stem.stem(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def load_data(path: str):
    df = pd.read_csv(path)
    print(f"Dataset: {df.shape[0]:,} articles | "
          f"FAKE={df[df.label==0].shape[0]:,} | REAL={df[df.label==1].shape[0]:,}")

    df.fillna("", inplace=True)
    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].apply(preprocess)

    return df["content"].values, df["label"].values   # 0=FAKE, 1=REAL


# ── Model Zoo ──────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=5.0, solver="lbfgs", random_state=RANDOM_STATE
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000, C=0.5, random_state=RANDOM_STATE
    ),
    "Linear SVM": LinearSVC(
        max_iter=2000, C=1.0, random_state=RANDOM_STATE
    ),
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    ),
}


# ── Evaluate one model ─────────────────────────────────────────────────────────
def evaluate(name, model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred     = model.predict(X_test)
    train_acc  = accuracy_score(y_train, model.predict(X_train))
    test_acc   = accuracy_score(y_test,  y_pred)
    f1         = f1_score(y_test, y_pred, average="weighted")

    try:
        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except AttributeError:
        roc = float("nan")

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  F1 (weighted)  : {f1:.4f}")
    if not np.isnan(roc):
        print(f"  ROC-AUC        : {roc:.4f}")
    print(f"  Train time     : {elapsed:.1f}s")
    print(f"\n{classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])}")

    return {
        "Model":          name,
        "Train Accuracy": round(train_acc, 4),
        "Test Accuracy":  round(test_acc, 4),
        "F1 Score":       round(f1, 4),
        "ROC-AUC":        round(roc, 4) if not np.isnan(roc) else "N/A",
        "Train Time (s)": round(elapsed, 1),
        "_model":         model,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X, y = load_data(DATA_PATH)

    print(f"\nVectorizing (max_features={MAX_FEATURES}, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    results   = []
    best_acc  = 0
    best_model = None
    best_name  = ""

    for name, model in MODELS.items():
        row = evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(row)
        if row["Test Accuracy"] > best_acc:
            best_acc   = row["Test Accuracy"]
            best_model = row["_model"]
            best_name  = name

    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != "_model"} for r in results])
    df_results = df_results.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

    print(f"\n{'='*55}")
    print("  FINAL LEADERBOARD")
    print(f"{'='*55}")
    print(df_results.to_string(index=False))
    print(f"\nBest model: {best_name}  ({best_acc:.4f} test accuracy)")

    df_results.to_csv(RESULTS_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer,  VECT_PATH)
    joblib.dump(best_name,   NAME_PATH)

    print(f"\nSaved → {MODEL_PATH}")
    print(f"Saved → {VECT_PATH}")
    print(f"Saved → {RESULTS_PATH}")
