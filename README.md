# 💬 Comment Category Prediction Challenge

Welcome to the **Comment Category Prediction Challenge**!

This competition provides insight into how an online platform processes and categorizes user-generated comments. Each record represents a single comment and includes metadata such as engagement feedback, symbolic expressions, topic reference indicators, and internal system-generated signals.

Your goal is to analyze this dataset and build predictive models that accurately determine the final category assigned to each comment.

---

## 📖 Overview

The dataset contains short textual entries submitted to an online discussion system. Along with the raw text, additional features capture:

- User engagement signals (upvotes and downvotes)
- Emoticon presence indicators
- Topic reference flags (e.g., race, religion, gender, disability)
- Internal hidden system features

Participants are expected to explore textual, numerical, categorical, and temporal signals to predict the **final classification label** assigned by the platform.

This is a **multi-class classification problem**, where the target variable can take one of four distinct values.

---

## 🎯 Objective

Build a machine learning model that:

- Learns patterns from comment text and metadata
- Combines structured and unstructured features
- Predicts the correct **category label**
- Generalizes effectively to unseen comments in the test set

---

## 📂 Data Files

The dataset includes three CSV files:

### 1️⃣ `train.csv`
- Contains all feature columns
- Includes the target variable `label`

### 2️⃣ `test.csv`
- Contains all feature columns
- Does **not** include the target variable

### 3️⃣ `sample_submission.csv`
- Example submission format
- Contains:
  - `ID`
  - `label`

---

## 📊 Column Description

| Column | Description |
|---------|-------------|
| `comment` | Raw text content of the comment |
| `created_date` | Timestamp when the comment was posted |
| `post_id` | Unique identifier linking to a discussion thread |
| `emoticon_1` | Indicator for presence of first emoticon group |
| `emoticon_2` | Indicator for presence of second emoticon group |
| `emoticon_3` | Indicator for presence of third emoticon group |
| `upvote` | Number of positive reactions |
| `downvote` | Number of negative reactions |
| `if_1` | Hidden internal feature one |
| `if_2` | Hidden internal feature two |
| `race` | Indicator for detected race-related reference |
| `religion` | Indicator for detected religion-related reference |
| `gender` | Indicator for detected gender-related reference |
| `disability` | Indicator for detected disability-related reference |
| `label` | Final category assigned by the system (Target Variable) |

---

## 🧠 Problem Type

- Multi-class classification
- Four distinct target categories
- Mixed feature types:
  - Textual
  - Numerical
  - Binary indicators
  - Temporal features

---

## 📈 Dataset Details

- **Total Size:** 112.07 MB  
- **File Type:** CSV  
- **License:** Subject to Competition Rules  
- **Competition Status:**  
  - Started: 1 month ago  
  - Closing in: 20 days  

The dataset contains approximately **102,000 samples**, evenly distributed across label bins in the sample structure.

---

## 🔍 Key Modeling Considerations

To build a strong predictive system, participants may explore:

- Text preprocessing (tokenization, stemming, lemmatization)
- TF-IDF or embedding-based representations
- Transformer-based models (e.g., BERT)
- Feature engineering from timestamps
- Engagement-based derived features (e.g., upvote ratio)
- Handling categorical indicators effectively
- Class balancing techniques (if needed)

---

## 🚀 Suggested Approaches

- Logistic Regression with TF-IDF
- Random Forest / Gradient Boosting
- XGBoost / LightGBM / CatBoost
- Deep Learning (LSTM, CNN for text)
- Transformer-based fine-tuning (e.g., BERT)

Combining textual embeddings with structured metadata features may significantly improve performance.

## 🧩 Challenge Summary

This competition tests your ability to:

- Analyze multi-modal datasets
- Combine NLP with structured data modeling
- Perform feature engineering
- Build scalable classification systems
- Optimize predictive performance

Use your analytical skills to uncover patterns across text, engagement signals, and hidden system indicators to accurately predict how comments are categorized.
