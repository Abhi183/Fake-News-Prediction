# Fake News Detection Using NLP and Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project implements a binary text classification system to detect fake news articles using Natural Language Processing (NLP) and a Logistic Regression classifier. Given a news article's author and title, the model predicts whether the article is **real (0)** or **fake (1)**.

The project includes a complete ML pipeline — from raw text preprocessing to a deployable Streamlit web interface — achieving **~97.9% accuracy** on held-out test data.

---

## Dataset

**Source:** [Kaggle — Fake News Competition](https://www.kaggle.com/competitions/fake-news/data)

| Property | Value |
|---|---|
| Total samples | 20,800 |
| Features used | `author`, `title` |
| Target | `label` (0 = Real, 1 = Fake) |
| Train / Test split | 80% / 20% (stratified) |

Missing values in `author` (1,957) and `title` (558) were imputed with empty strings prior to feature construction.

---

## Methodology

### 1. Feature Engineering
Author name and article title were concatenated into a single `content` field to capture authorship patterns alongside headline semantics.

### 2. Text Preprocessing
Each content string was passed through a custom stemming pipeline:
- Remove non-alphabetic characters using regex
- Lowercase normalization
- Tokenization
- Stopword removal (NLTK English corpus)
- Porter Stemming to reduce tokens to their root form

### 3. Vectorization
Preprocessed text was transformed into numerical feature vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**, which weights tokens by their importance relative to the corpus.

### 4. Classification
A **Logistic Regression** model was trained on the TF-IDF feature matrix. Despite its simplicity, logistic regression is well-suited to high-dimensional sparse text data.

---

## Results

| Split | Accuracy |
|---|---|
| Training | 98.66% |
| Test | 97.91% |

The small gap between training and test accuracy indicates the model generalizes well without significant overfitting.

---

## Project Structure

```
fake-news-detection/
│
├── Fake_News_Prediction.ipynb   # Jupyter notebook (EDA + training pipeline)
├── train.py                     # Standalone training script with model export
├── app.py                       # Streamlit prediction UI
│
├── model/
│   ├── fake_news_model.pkl      # Saved Logistic Regression model
│   └── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
│
├── data/
│   └── train.csv                # Kaggle dataset (not included, see link above)
│
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install dependencies

```bash
pip install numpy pandas scikit-learn nltk streamlit
```

### Download NLTK stopwords

```python
import nltk
nltk.download('stopwords')
```

---

## Usage

### Step 1: Train the model

```bash
python train.py
```

This reads `data/train.csv`, trains the model, and saves `model/fake_news_model.pkl` and `model/tfidf_vectorizer.pkl`.

### Step 2: Launch the UI

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and enter a news article's author and title to get a prediction.

---

## Limitations & Future Work

- The model uses only `author` and `title`, not the full article body. Incorporating article text would likely improve performance.
- The training data is from a single Kaggle competition and may not generalize to modern news sources.
- Future iterations could explore transformer-based models (e.g., BERT) for richer contextual representations.

---

## Technologies Used

- **Python** — core language
- **pandas / NumPy** — data handling
- **NLTK** — stopword removal, stemming
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, evaluation
- **Streamlit** — interactive prediction UI
- **joblib** — model serialization

---

## License

MIT License. See `LICENSE` for details.
