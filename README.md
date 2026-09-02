# 💬 Comment Category Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-00B4D8?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Multi-class NLP classification · F1-Macro 0.8337 · Accuracy 91.8%**

[Problem Statement](#-problem-statement) · [Dataset](#-dataset) · [Approach](#-approach) · [Results](#-results) · [Installation](#-installation) · [Repository Structure](#-repository-structure)

</div>

---

## 📌 Overview

This project is a solution to the **Comment Category Prediction Challenge** (Kaggle t12026). The task is to classify online comments into one of four tone/toxicity categories (labels 0–3) using a combination of raw comment text and behavioural metadata such as upvotes, downvotes, and internal platform signals.

The final submission uses a **weighted probability blend** of LightGBM and Logistic Regression trained on a 35,016-feature matrix built from dual TF-IDF vectorisation and hand-crafted numeric features.

| Metric | Score |
|--------|-------|
| **F1-Macro (validation)** | **0.8337** |
| Accuracy (validation) | 91.8% |
| Weighted F1 (validation) | 0.920 |

---

## 🎯 Problem Statement

**Type:** Multi-class text classification  
**Target:** `label` — one of four categories (0, 1, 2, 3)  
**Challenge:** Severe class imbalance (~20:1 ratio between Label 0 and Label 3)

| Label | Tone | Training Count | Share |
|-------|------|---------------|-------|
| 0 | Neutral / Informative | 114,173 | 57.7% |
| 1 | Mildly Toned | 15,918 | 8.0% |
| 2 | Moderately Negative | 62,440 | 31.5% |
| 3 | Targeted / Aggressive | 5,469 | 2.8% |

Because accuracy rewards the majority class, **F1-Macro** was used as the primary evaluation metric. It averages the F1-score across all four classes equally, penalising the model for ignoring any category.

---

## 📂 Dataset

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train.csv` | 198,000 | 15 | Includes `label` |
| `test.csv` | 102,000 | 14 | No target column |
| `sample_submission.csv` | 102,000 | 2 | `ID`, `label` |

### Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `comment` | text | Raw comment text |
| `created_date` | datetime | Timestamp of submission |
| `post_id` | int | Discussion thread identifier |
| `emoticon_1/2/3` | int | Emoticon presence counts |
| `upvote` | int | Positive community reactions |
| `downvote` | int | Negative community reactions |
| `if_1` | int | Hidden internal feature 1 |
| `if_2` | int | Hidden internal feature 2 (strongest predictor, ρ=0.83) |
| `race` | object | Race-related content flag |
| `religion` | object | Religion-related content flag |
| `gender` | object | Gender-related content flag |
| `disability` | bool | Disability-related content flag |
| `label` | int | **Target variable** (train only) |

> **Note:** `race`, `religion`, and `gender` are missing in 73.5% of rows across both train and test — they were dropped after EDA confirmed imputation would introduce more noise than signal.

---

## 🔬 Approach

The pipeline follows five stages:

```
Raw Data → EDA & Cleaning → Feature Engineering → Preprocessing → Modelling → Blend & Predict
```

### 1 · Exploratory Data Analysis

Key findings that drove every downstream decision:

- **`if_2`** has a Spearman correlation of **0.83** with the label — the single strongest predictor
- `race`, `religion`, `gender` are **73.5% missing** and drop-worthy
- `emoticon_2` has near-zero correlation and near-zero variance — dropped
- Upvote and downvote are severely right-skewed (max 201 and 107 vs means of 2.6 and 0.67)
- Label 3 comments are on average **35 words** — significantly shorter than other classes (51–57 words), giving fewer text signals

### 2 · Data Cleaning

| Decision | Evidence | Action |
|----------|----------|--------|
| Drop `race`, `religion`, `gender` | 73.5% missing in both splits | `dropna(thresh=0.3 × n)` |
| Drop `emoticon_2` | Spearman ρ ≈ 0; IQR = 0 | Explicit drop |
| Cap outliers | `if_2` max=1,833 vs median=6; `upvote` max=201 vs mean=2.6 | Winsorise at 5th–95th percentile on **train only**; apply same bounds to test |
| Fill missing comment | 1 null in 198K rows | Fill with `""` |

> **Leakage prevention:** All bounds (scalers, TF-IDF vocabulary, winsorisation percentiles, target encoding means) are computed exclusively from `x_train` and applied via `.transform()` to validation and test sets.

### 3 · Feature Engineering

Eight features are extracted **before** text cleaning, because lowercasing and stripping punctuation permanently destroys some signals:

| Feature | Formula | Why it helps |
|---------|---------|-------------|
| `caps_ratio` | `CAPS / total_alpha` | ALL-CAPS is an aggression signal — only visible pre-cleaning |
| `has_url` | `1 if http/www` | URL comments skew neutral (Label 0) |
| `word_len` | `len(text.split())` | Label 3 averages 35 words vs 51–57 for others |
| `char_len` | `len(text)` | Complementary raw-length signal |
| `avg_word_len` | `char_len / (word_len + 1)` | Vocabulary complexity proxy |
| `upvote_log1p` | `log(1 + upvote)` | Compresses right-skewed upvote distribution |
| `downvote_log1p` | `log(1 + downvote)` | Same; aggressive comments attract more downvotes |
| `emotion_density` | `emoticon_3 / (word_len + 1)` | Emoticon-per-word intensity, length-normalised |
| `post_id_encoded` | `mean(label) per post_id in train` | Thread-level toxicity context |
| `downvote_x_caps` | `downvote_log1p × caps_ratio` | Interaction: short + all-caps + downvoted → Label 3 |
| `upvote_x_word_len` | `upvote_log1p × word_len` | Engagement × length interaction |

### 4 · Preprocessing Pipeline

```python
ColumnTransformer([
    ('word_tfidf', TfidfVectorizer(
        analyzer='word', max_features=20_000,
        ngram_range=(1, 2), sublinear_tf=True,
        stop_words='english', min_df=3, max_df=0.95
    ), 'comment'),
    ('char_tfidf', TfidfVectorizer(
        analyzer='char_wb', max_features=15_000,
        ngram_range=(3, 5), sublinear_tf=True,
        min_df=3, max_df=0.95
    ), 'comment'),
    ('scale', MinMaxScaler(), numeric_cols)
])
```

**Output shape:** `(158,400 × 35,016)` — 20,000 word features + 15,000 character features + 16 numeric features.

**Why dual TF-IDF?**

| Level | Captures |
|-------|---------|
| Word (1–2 grams) | Vocabulary and phrases: `'not good'` ≠ `'very good'` |
| Character (3–5 grams) | Misspellings (`h8`, `st*pid`), morphological patterns (`-ism`, `-phob`) |

**Why `MinMaxScaler` not `StandardScaler`?** TF-IDF output is non-negative. Keeping numeric features in `[0, 1]` maintains sign consistency across the full feature matrix.

### 5 · Model Comparison

All models use `class_weight='balanced'` and are evaluated with `f1_macro` on a stratified 80/20 split.

| Model | F1-Macro | Accuracy | Notes |
|-------|----------|----------|-------|
| LightGBM | **0.830** | **91.8%** | Non-linear; breaks linear ceiling |
| LinearSVC | 0.806 | 90.7% | Margin-maximiser; best linear |
| SGD Classifier | 0.806 | 90.2% | Same family as LR; faster convergence |
| Logistic Regression | 0.800 | 90.3% | Classical TF-IDF baseline |
| ComplementNB | 0.658 | 77.8% | Independence assumption violated |

> **The Linear Ceiling:** All three linear models plateau at ~0.806 regardless of tuning — they share a hard ceiling that only a non-linear model can break. LightGBM learns interactions between features (e.g. *short + high downvotes + specific words → Label 3*) that no linear boundary can capture.

### 6 · Hyperparameter Tuning

**LightGBM** was tuned via **early stopping** on validation `multi_logloss`:

```python
LGBMClassifier(
    n_estimators=2000,       # upper bound
    learning_rate=0.05,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced'
)
# Early stopping: patience=30 → stopped at round 293
```

**Logistic Regression** was tuned via `RandomizedSearchCV` (10 iterations × 3 `StratifiedKFold`):

```python
# Best params
{'C': 1.0, 'solver': 'lbfgs'}   # CV F1: 0.8045 → Holdout F1: 0.8051
```

### 7 · Model Blending

A grid search over blend weights (0.1 steps) found the optimal combination:

```python
final_proba = 0.7 * lgb_proba + 0.3 * lr_proba
y_pred = np.argmax(final_proba, axis=1)
```

| LGB weight | LR weight | F1-Macro |
|-----------|----------|----------|
| 0.5 | 0.5 | 0.8295 |
| 0.6 | 0.4 | 0.8317 |
| **0.7** | **0.3** | **0.8337** ← best |
| 0.8 | 0.2 | 0.8330 |
| 0.9 | 0.1 | 0.8312 |

LightGBM excels at non-linear patterns; Logistic Regression excels in high-dimensional linear spaces. Averaging their **probability outputs** (not hard predictions) produces softer, better-calibrated final scores.

---

## 📊 Results

### Per-Class Performance (Validation Set, 39,600 rows)

| Label | Precision | Recall | F1-Score | Support | Interpretation |
|-------|-----------|--------|----------|---------|---------------|
| 0 | 0.982 | 0.946 | **0.964** | 22,835 | Near-perfect; majority class, most training data |
| 1 | 0.739 | 0.858 | **0.794** | 3,183 | Good recall; some Label 0 confused as Label 1 |
| 2 | 0.886 | 0.899 | **0.893** | 12,488 | Strong; second-largest class well covered |
| 3 | 0.643 | 0.732 | **0.685** | 1,094 | Weakest; rare class + short text + Label 2 overlap |

**Overall Accuracy: 91.8% · Macro F1: 0.8337 · Weighted F1: 0.920**

### Test Set Prediction Distribution

| Label | Predicted | Training |
|-------|-----------|---------|
| 0 | 55.4% | 57.7% |
| 1 | 9.2% | 8.0% |
| 2 | 32.4% | 31.5% |
| 3 | 3.1% | 2.8% |

Close alignment between predicted and training distributions confirms no systematic class prediction bias.

### Why Label 3 is the Hardest

1. **Smallest class** — only 5,469 training examples (2.8%)
2. **Shortest comments** — average 35 words vs 51–57 for others → fewer TF-IDF signals
3. **Vocabulary overlap** — shares similar negative-tone vocabulary with Label 2

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/23f3001800/Comment-Category-Prediction-Challenge.git
cd Comment-Category-Prediction-Challenge

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
lightgbm>=4.0
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
wordcloud>=1.9
```

---

## 🗂️ Repository Structure

```
Comment-Category-Prediction-Challenge/
│
├── data/
│   ├── train.csv                  # Training data (198,000 rows)
│   ├── test.csv                   # Test data (102,000 rows)
│   └── sample_submission.csv      # Submission format
│
├── notebooks/
│   └── comment_category_prediction.ipynb   # Full pipeline notebook
│
├── outputs/
│   └── submission.csv             # Final predictions
│
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

Open the notebook and run all cells in order:

```bash
jupyter notebook notebooks/comment_category_prediction.ipynb
```

The notebook is self-contained and structured in the following sections:

1. **Importing Libraries** — all dependencies loaded upfront
2. **Data Loading** — reads `train.csv`, `test.csv`, `sample_submission.csv`
3. **EDA** — shape, types, missing values, class distribution, word clouds, correlation heatmaps
4. **Data Cleaning** — drops, fills, and winsorises based on EDA findings
5. **Feature Engineering** — 11 engineered features extracted pre-cleaning
6. **Preprocessing Pipeline** — `ColumnTransformer` with dual TF-IDF + `MinMaxScaler`
7. **Model Comparison** — trains and evaluates 5 models with confusion matrices
8. **Hyperparameter Tuning** — LightGBM early stopping + LR `RandomizedSearchCV`
9. **Model Blending** — grid search over blend weights
10. **Submission** — generates `submission.csv`

---

## 💡 Key Learnings

| # | Insight |
|---|---------|
| 1 | **Text is the richest signal.** 35,000 TF-IDF features (word + character) dominate all other features combined |
| 2 | **Don't dismiss unknown columns.** `if_2` had no description but Spearman ρ = 0.83 with the label — the strongest predictor |
| 3 | **Class imbalance needs layered defence.** `class_weight='balanced'` + F1-Macro scoring + stratified sampling must all work together |
| 4 | **Linear models have a hard ceiling.** All three plateaued at ~0.806 regardless of tuning; non-linearity was essential to improve further |
| 5 | **Blending diverse models beats any single model.** 0.7 LGB + 0.3 LR → 0.8337 vs 0.830 for LightGBM alone |

---

## 🔮 Potential Improvements

- **Transformer embeddings** (DistilBERT, RoBERTa) — contextual understanding: `"NOT happy"` ≠ `"very happy"`
- **Focal loss** — concentrates training effort on hard, misclassified examples (targets Label 3 specifically)
- **Retrain on 100% of data** — current model uses only 80%; the held-out 20% contains ~1,400 additional Label 3 examples
- **Three-model blend** — adding LinearSVC as a third diverse ensemble member
- **SMOTE / oversampling** — synthetic minority oversampling for Label 3

---

## 📋 Evaluation Criterion

> **Primary metric: F1-Macro**
>
> F1-Macro averages the F1-score across all four label classes with equal weight. This prevents the model from gaming the metric by over-predicting the majority class (Label 0 = 57.7%), and holds it accountable to correctly classifying rare categories like Label 3 (2.8%).

```
F1_macro = (F1_label0 + F1_label1 + F1_label2 + F1_label3) / 4

where F1_i = 2 × (Precision_i × Recall_i) / (Precision_i + Recall_i)
```

---

## 🤝 Acknowledgements

- Dataset provided by the Kaggle competition: **Comment Category Prediction Challenge (t12026)**
- Built with [scikit-learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/), and [pandas](https://pandas.pydata.org/)

---

<div align="center">

**F1-Macro: 0.8337 · Accuracy: 91.8% · 35,016 features · 5 models tested**

*If you found this useful, please consider leaving a ⭐*

</div>
