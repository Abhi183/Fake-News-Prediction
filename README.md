# Fake News Detection using NLP and Ensemble Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Dataset-40%2C587%20articles-green" />
  <img src="https://img.shields.io/badge/Best%20Accuracy-98.23%25-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## Abstract

The proliferation of misinformation across digital media platforms poses significant societal risks, influencing public opinion, electoral processes, and public health outcomes. This project develops an automated fake news detection system using natural language processing (NLP) and classical machine learning classifiers applied to a corpus of 40,587 labeled news articles. Five classifiers are systematically trained and evaluated: Logistic Regression, Passive Aggressive Classifier, Linear Support Vector Machine (SVM), Multinomial Naïve Bayes, and Random Forest. All models use TF-IDF vectorization with bigram features on Porter-stemmed text. The best-performing model — a Linear SVM — achieves **98.23% test accuracy** and is deployed in an interactive Streamlit web application for real-time classification.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Interactive Demo](#interactive-demo)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Background

Fake news, defined as fabricated or deliberately misleading information presented as factual reporting, has become a significant challenge in the era of social media. Studies have shown that false stories spread approximately six times faster than true ones on platforms like Twitter (Vosoughi et al., 2018). Automated detection systems represent a scalable first-line defense against such content.

Classical NLP approaches — combining bag-of-words or TF-IDF representations with discriminative classifiers — have demonstrated strong performance on benchmark fake news detection tasks (Ahmed et al., 2017; Shu et al., 2017). This project benchmarks five such approaches on a balanced, real-world dataset to determine the most effective combination of vectorization strategy and classifier for production deployment.

---

## Dataset

**Source**: GonzaloA/fake_news (HuggingFace Datasets Hub)
**Original basis**: Kaggle Fake News Detection Competition corpus

| Property | Value |
|---|---|
| Total articles | 40,587 |
| FAKE articles | 18,663 (46.0%) |
| REAL articles | 21,924 (54.0%) |
| Features used | Title + Article body |
| Missing values | Imputed with empty string |
| Train / Test split | 80% / 20% (stratified) |

### Column Description

| Column | Type | Description |
|---|---|---|
| `id` | Integer | Unique article identifier |
| `title` | String | Article headline |
| `text` | String | Full article body |
| `label` | Binary | 0 = FAKE, 1 = REAL |
| `label_name` | String | Human-readable label |

The dataset is stored at `data/fake_news_dataset.csv` and is included in this repository for full reproducibility.

---

## Methodology

### 1. Text Preprocessing

All article text (title + body concatenated) is processed through the following pipeline:

1. **Non-alphabetic removal**: `re.sub(r"[^a-zA-Z]", " ", text)`
2. **Lowercasing**
3. **Stopword removal**: NLTK English stopword list
4. **Token length filter**: Tokens shorter than 3 characters discarded
5. **Porter Stemming**: Reduces inflected forms to root (e.g., *running* → *run*)

### 2. Feature Extraction — TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) is computed with the following configuration:

| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 50,000 | Vocabulary cap to control dimensionality |
| `ngram_range` | (1, 2) | Unigrams + bigrams capture phrasal patterns |
| `sublinear_tf` | True | Log-scaled TF dampens high-frequency term dominance |

### 3. Classifiers

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | C=5.0, solver=lbfgs, max_iter=1000 |
| Passive Aggressive | C=0.5, max_iter=1000 |
| Linear SVM (LinearSVC) | C=1.0, max_iter=2000 |
| Multinomial Naïve Bayes | alpha=0.1 (Laplace smoothing) |
| Random Forest | 200 trees, max_depth=None |

### 4. Evaluation Metrics

- **Accuracy**: Overall fraction of correct predictions
- **F1 Score (weighted)**: Harmonic mean of precision and recall, weighted by class support
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve (where available)
- **Classification Report**: Per-class precision, recall, and F1

---

## Results

### Model Leaderboard

| Rank | Model | Train Acc | Test Acc | F1 Score | ROC-AUC | Train Time |
|---|---|---|---|---|---|---|
| 1 | **Linear SVM** | 99.96% | **98.23%** | **0.9823** | N/A | 0.2s |
| 2 | Passive Aggressive | 100.00% | 98.15% | 0.9815 | N/A | 0.2s |
| 3 | Logistic Regression | 99.26% | 97.82% | 0.9782 | **0.9972** | 0.2s |
| 4 | Random Forest | 100.00% | 97.68% | 0.9769 | 0.9950 | 6.2s |
| 5 | Multinomial NB | 95.28% | 94.85% | 0.9485 | 0.9832 | 0.0s |

### Key Findings

1. **Linear SVM achieves the best test accuracy (98.23%)** with negligible training time, making it the optimal choice for both accuracy and computational efficiency.

2. **Passive Aggressive and Linear SVM are the top linear models**, consistent with their design for large-scale text classification tasks. Both process 32,469 training documents in under 0.3 seconds.

3. **Logistic Regression achieves the highest ROC-AUC (0.9972)**, indicating excellent probabilistic calibration — valuable when confidence scores are needed alongside binary predictions.

4. **Random Forest overfits** (100% train, 97.68% test) despite using 200 trees, confirming that tree-based ensembles struggle with high-dimensional sparse TF-IDF features relative to linear classifiers.

5. **Including article body text significantly improves performance** over title-only features, providing richer lexical signals for classification.

6. **Bigram features** (ngram_range=(1,2)) capture idiomatic expressions and stylistic patterns (e.g., *"breaking news"*, *"sources say"*) that unigrams miss.

---

## Interactive Demo

A Streamlit web application is included for real-time prediction:

- **Manual input**: Enter any news headline and article body
- **Sample articles**: Pre-loaded real and fake examples accessible from the sidebar
- **Prediction output**: Binary label, confidence percentage, and probability bar chart
- **Dataset explorer**: Class distribution visualization
- **Model leaderboard**: Compare all five trained models from the sidebar

### Launch the App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`.

---

## Repository Structure

```
Fake-News-Prediction/
├── data/
│   └── fake_news_dataset.csv          # 40,587 labeled articles
├── models/
│   ├── best_model.pkl                 # Best trained classifier (Linear SVM)
│   ├── tfidf_vectorizer.pkl           # Fitted TF-IDF vectorizer
│   ├── best_model_name.pkl            # Model name string
│   └── model_results.csv             # Leaderboard across all models
├── Fake News Prediction/
│   └── Fake_News_Prediction.ipynb    # Exploratory analysis notebook
├── app.py                             # Streamlit prediction UI
├── train.py                           # Multi-model training pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Abhi183/Fake-News-Prediction.git
cd Fake-News-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Notebook

```bash
jupyter notebook "Fake News Prediction/Fake_News_Prediction.ipynb"
```

### Retrain All Models

```bash
python train.py
```

Outputs trained artifacts to `models/`.

### Launch the Streamlit App

```bash
streamlit run app.py
```

The pre-trained model is included in `models/`, so retraining is optional.

---

## References

1. Ahmed, H., Traore, I., & Saad, S. (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. *Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments*, 127–138.

2. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake News Detection on Social Media: A Data Mining Perspective. *ACM SIGKDD Explorations Newsletter*, 19(1), 22–36.

3. Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science*, 359(6380), 1146–1151. https://doi.org/10.1126/science.aap9559

4. Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008). LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9, 1871–1874.

5. Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137.

6. GonzaloA. (2023). fake_news [Dataset]. Hugging Face. https://huggingface.co/datasets/GonzaloA/fake_news

---

## License

This project is licensed under the MIT License.
Dataset: CC BY 4.0.

---

*DSDA 385 — Abhishek Shekhar*
