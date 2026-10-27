# 💬 Comment Category Prediction Challenge

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.17+-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)

**Multi-class NLP classification · Single scikit-learn Pipeline · MLflow Tracking · Automated Tests**

[Overview](#-overview) · [Architecture & Pipeline](#-architecture--pipeline) · [Repository Structure](#-repository-structure) · [MLflow Experiment Tracking](#-mlflow-experiment-tracking) · [Quickstart & Usage](#-quickstart--usage) · [Component Documentation](#-component-documentation)

</div>

---

## 📌 Overview

This repository provides an end-to-end machine learning system for the **Comment Category Prediction Challenge**. The task is to classify online comments into one of four toxicity/tone categories (labels 0–3) using raw comment text alongside behavioural metadata (upvotes, downvotes, reaction indicators, platform signals).

### Key Highlights
- **Single scikit-learn `Pipeline`**: Preprocessing, dual TF-IDF vectorisation (word + character n-grams), and feature engineering are contained inside a unified pipeline. All vectorizer vocabularies and scalers are fitted exclusively on training splits to prevent data leakage.
- **MLflow Experiment Tracking**: Standardized logging of model parameters, evaluation metrics (Macro-F1, Accuracy, per-class scores), validation strategies, confusion matrix charts, and serialized model artifacts.
- **Production-ready CLI & Modules**: Modularized data validation, training, evaluation/error analysis, prediction, and test suites.

---

## 🗂️ Repository Structure

```
Comment-Category-Prediction-Challenge/
├── notebooks/
│   └── notebook.ipynb            # Original EDA & exploratory modeling notebook
├── src/
│   ├── __init__.py
│   ├── data.py                   # Data ingestion and schema validation
│   ├── pipeline.py               # Single scikit-learn Pipeline with feature engineering
│   ├── train.py                  # Pipeline training and MLflow tracking
│   ├── evaluate.py               # Metrics computation and error inspection
│   └── predict.py                # Model inference CLI for comments & test files
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # Pytest suite (valid/invalid inputs, shape, labels)
├── models/                       # Serialized pipelines (.joblib)
├── reports/                      # Confusion matrix plots & evaluation summaries
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Pinned project dependencies
└── README.md
```

---

## ⚙️ Component Responsibilities

| Component | Responsibility | Description |
|-----------|----------------|-------------|
| **`src/data.py`** | Load data and perform basic validation | Ingests CSVs, verifies schema, asserts required columns (`comment`, `label`), validates target classes `{0, 1, 2, 3}`, checks for nulls, and provides stratified train/val splits. |
| **`src/pipeline.py`** | Unified preprocessing & modeling | Bundles text extraction, dual TF-IDF (word + char_wb), metadata transformation (log1p votes, ratios, interactions), and classifier into a single `Pipeline`. |
| **`src/train.py`** | Train pipeline & track with MLflow | Executes training (holdout split or Stratified K-Fold CV), records hyperparameters, metrics (Macro-F1, Accuracy, class F1s), validation strategy, and artifacts to MLflow. |
| **`src/evaluate.py`** | Calculate metrics & inspect errors | Computes classification metrics, generates confusion matrix, and inspects top misclassified comment patterns (e.g. rare class confusion). |
| **`src/predict.py`** | Model inference & submissions | Loads saved `.joblib` pipelines and predicts on single comments or generates test CSV submissions (`ID`, `label`). |
| **`tests/test_pipeline.py`** | Automated unit tests | Validates pipeline behavior on valid inputs, edge cases (empty strings, NaNs, missing columns), prediction shape `(N,)`, and label validity. |

---

## 🔬 Leakage Prevention Architecture

In standard NLP workflows, fitting a text vectorizer or scaler on the entire dataset prior to cross-validation causes **data leakage**, inflating validation performance.

Here, all vectorizers and transformers are wrapped inside a single **scikit-learn `Pipeline`**:
```
Raw DataFrame / Comments
           │
     ┌─────┴───────────────────────────────────────────────────────┐
     │                     FeatureUnion                            │
     │                                                             │
     │  [TextExtractor]  ───> [Word TfidfVectorizer (1-2 grams)]   │
     │  [TextExtractor]  ───> [Char TfidfVectorizer (3-5 grams)]   │
     │  [Metadata/Text]  ───> [MinMaxScaler]                      │
     └─────┬───────────────────────────────────────────────────────┘
           │
           ▼
[Classifier (LinearSVC / LogisticRegression)]
```
- Parameters (`vocabulary_`, `idf_`, `data_min_`, `data_max_`) are learned **only** from training folds during `.fit()`.
- The pipeline seamlessly accepts raw pandas DataFrames or text series directly during `.predict()`.

---

## 📊 MLflow Experiment Tracking

Three distinct architectural experiments are recorded and tracked in MLflow:

| Experiment Run | Preprocessing & Features | Classifier | Val Strategy | Macro-F1 | Accuracy | Artifacts Saved |
|---|---|---|---|:---:|:---:|---|
| **`LogisticRegression_Baseline`** | Word TF-IDF (15,000 features, 1–2 n-grams) | Logistic Regression (`class_weight='balanced'`) | Stratified 80/20 split | **0.6627** | 76.27% | `.joblib` model, confusion matrix plot, report |
| **`LinearSVM_DualTFIDF`** | Word TF-IDF + Character-wb TF-IDF (3–5 n-grams) | LinearSVC (`class_weight='balanced'`) | Stratified 80/20 split | **0.6875** | 79.02% | `.joblib` model, confusion matrix plot, report |
| **`LinearSVM_BestFeatureEngineering`** | Dual TF-IDF + Text & Metadata Features (`caps_ratio`, `if_1`, `if_2`, `votes_log1p`, `emotion_density`, interactions) | LinearSVC (`class_weight='balanced'`) | Stratified 80/20 split | **0.6940** | **79.58%** | `.joblib` model, confusion matrix plot, report |

### Per-Class F1 Breakdown (`LinearSVM_BestFeatureEngineering`):
- **Label 0 (Neutral)**: `0.8512`
- **Label 1 (Mildly Toned)**: `0.6775`
- **Label 2 (Moderately Negative)**: `0.7562`
- **Label 3 (Targeted / Aggressive)**: `0.4911`

### Launching the MLflow UI
The local MLflow tracking server provides a web dashboard to compare runs, parameters, metrics curves, and inspect saved artifacts:

```bash
# Start MLflow server
mlflow server --port 5000
```
Open your browser at **[http://localhost:5000](http://localhost:5000)**.

---

## 🚀 Quickstart & Usage

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests

```bash
pytest -v
```
All 14 unit tests verify valid inputs, invalid schema handling, prediction shapes, and label integrity.

### 3. Training Pipelines with MLflow Tracking

```bash
# Experiment 1: Baseline Logistic Regression (word TF-IDF)
python -m src.train --model-type logistic_regression --no-fe --no-dual-tfidf --run-name LogisticRegression_Baseline

# Experiment 2: Linear SVM with Dual TF-IDF (word + character n-grams)
python -m src.train --model-type linear_svm --no-fe --run-name LinearSVM_DualTFIDF

# Experiment 3: Best Feature-Engineering Variant (Dual TF-IDF + Metadata + Text signals)
python -m src.train --model-type linear_svm --run-name LinearSVM_BestFeatureEngineering

# Optional: Run with 5-Fold Stratified Cross-Validation
python -m src.train --model-type linear_svm --cv 5 --run-name LinearSVM_5Fold_CV
```

### 4. Evaluate a Trained Model & Inspect Errors

```bash
python -m src.evaluate \
  --model-path models/LinearSVM_BestFeatureEngineering.joblib \
  --data-path data/train.csv \
  --nrows 5000 \
  --output-report reports/evaluation_summary.json
```
Prints evaluation metrics, confusion matrix, and displays the most frequent error confusion patterns (e.g. True 2 mispredicted as 0, or rare class 3 confusion).

### 5. Predict on New Comments or Generate Submission

```bash
# Predict on a single raw comment string:
python -m src.predict \
  --model-path models/LinearSVM_BestFeatureEngineering.joblib \
  --comment "This is an informative, well-written analysis."

# Predict on full test set and output submission CSV:
python -m src.predict \
  --model-path models/LinearSVM_BestFeatureEngineering.joblib \
  --test-file data/test.csv \
  --output-csv submission.csv
```

---

## 🧪 Error Analysis Insights

1. **Class Imbalance & Rare Class Challenge**: Label 3 represents only ~2.8% of the data. Without balanced class weighting, models collapse toward predicting majority class 0. With balanced weights, Label 3 recall significantly improves.
2. **Text Length Disparity**: Label 3 comments average 35 words compared to 51–57 for others, meaning fewer bag-of-words cues. Character-level TF-IDF (`char_wb`) helps recover subword and morphological patterns.
3. **Metadata Signal**: Adding `if_2` (strongest tabular predictor), `caps_ratio`, and interaction terms (`downvote_x_caps`, `upvote_x_word_len`) pushed the linear model Macro-F1 from **0.6627** up to **0.6940**.
