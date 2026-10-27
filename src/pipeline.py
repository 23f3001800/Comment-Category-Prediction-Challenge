"""Pipeline definition integrating preprocessing, feature engineering, and classifiers."""

import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


def clean_text_str(text: Any) -> str:
    """Clean single comment string."""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class TextExtractor(BaseEstimator, TransformerMixin):
    """Extract comment column as a clean 1D iterable of strings."""

    def __init__(self, text_column: str = "comment"):
        self.text_column = text_column

    def fit(self, X: Any, y: Optional[Any] = None) -> "TextExtractor":
        return self

    def transform(self, X: Any) -> List[str]:
        if isinstance(X, pd.DataFrame):
            if self.text_column in X.columns:
                series = X[self.text_column]
            else:
                # If dataframe has 1 string/object column, use it, else empty
                series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        elif isinstance(X, (list, np.ndarray)):
            series = pd.Series(X)
        else:
            series = pd.Series([str(X)])

        return [clean_text_str(item) for item in series.fillna("")]


class MetadataAndTextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Engineers text-derived features (caps, length, urls) and numerical metadata features."""

    def __init__(self, text_column: str = "comment"):
        self.text_column = text_column
        self.feature_names_: List[str] = []

    def fit(self, X: Any, y: Optional[Any] = None) -> "MetadataAndTextFeatureExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, pd.Series):
            df = pd.DataFrame({self.text_column: X})
        elif isinstance(X, (list, np.ndarray)):
            df = pd.DataFrame({self.text_column: X})
        else:
            df = pd.DataFrame({self.text_column: [str(X)]})

        raw_txt = df[self.text_column].fillna("").astype(str) if self.text_column in df.columns else pd.Series([""] * len(df))

        # Text engineered features
        char_len = raw_txt.str.len().to_numpy()
        alpha_chars = raw_txt.str.count(r"[a-zA-Z]").replace(0, 1)
        caps_ratio = (raw_txt.str.count(r"[A-Z]") / alpha_chars).to_numpy()
        has_url = raw_txt.str.contains(r"http\S+|www\S+", regex=True).astype(float).to_numpy()
        exclamation_count = raw_txt.str.count(r"!").to_numpy()
        word_len = raw_txt.str.split().str.len().fillna(0).to_numpy()
        avg_word_len = char_len / (word_len + 1.0)

        # Metadata features (with safe fallbacks if columns not present in inference)
        upvote = df["upvote"].astype(float).fillna(0.0).to_numpy() if "upvote" in df.columns else np.zeros(len(df))
        downvote = df["downvote"].astype(float).fillna(0.0).to_numpy() if "downvote" in df.columns else np.zeros(len(df))
        emoticon_3 = df["emoticon_3"].astype(float).fillna(0.0).to_numpy() if "emoticon_3" in df.columns else np.zeros(len(df))
        if_1 = df["if_1"].astype(float).fillna(0.0).to_numpy() if "if_1" in df.columns else np.zeros(len(df))
        if_2 = df["if_2"].astype(float).fillna(0.0).to_numpy() if "if_2" in df.columns else np.zeros(len(df))

        if "disability" in df.columns:
            disability = df["disability"].fillna(False).astype(float).to_numpy()
        else:
            disability = np.zeros(len(df))

        # Engineered combinations
        upvote_log1p = np.log1p(np.maximum(0.0, upvote))
        downvote_log1p = np.log1p(np.maximum(0.0, downvote))
        emotion_density = emoticon_3 / (word_len + 1.0)
        downvote_x_caps = downvote_log1p * caps_ratio
        upvote_x_word_len = upvote_log1p * word_len

        feats = np.column_stack([
            char_len,
            caps_ratio,
            has_url,
            exclamation_count,
            word_len,
            avg_word_len,
            upvote_log1p,
            downvote_log1p,
            emotion_density,
            downvote_x_caps,
            upvote_x_word_len,
            if_1,
            if_2,
            disability,
        ])
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def create_pipeline(
    model_type: str = "logistic_regression",
    use_feature_engineering: bool = True,
    use_dual_tfidf: bool = True,
    word_max_features: int = 15000,
    char_max_features: int = 10000,
    min_df: int = 1,
    classifier_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Create a self-contained scikit-learn Pipeline with preprocessing and classifier.

    Args:
        model_type: One of 'logistic_regression', 'linear_svm', 'sgd'.
        use_feature_engineering: Whether to include hand-crafted text/metadata features.
        use_dual_tfidf: Whether to use both word and character-level TF-IDF.
        word_max_features: Maximum vocabulary size for word TF-IDF.
        char_max_features: Maximum vocabulary size for character-level TF-IDF.
        min_df: Minimum document frequency for TF-IDF.
        classifier_params: Additional kwargs to pass to the classifier.

    Returns:
        Complete scikit-learn Pipeline instance.
    """
    clf_params = classifier_params.copy() if classifier_params else {}

    # Define Text Preprocessing feature unions
    features_list: List[tuple] = []

    # 1. Word TF-IDF
    word_tfidf = Pipeline([
        ("extract_text", TextExtractor()),
        ("tfidf_word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=word_max_features,
            sublinear_tf=True,
            stop_words="english",
            min_df=min_df,
        )),
    ])
    features_list.append(("word_tfidf", word_tfidf))

    # 2. Character-level TF-IDF
    if use_dual_tfidf:
        char_tfidf = Pipeline([
            ("extract_text", TextExtractor()),
            ("tfidf_char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=char_max_features,
                sublinear_tf=True,
                min_df=min_df,
            )),
        ])
        features_list.append(("char_tfidf", char_tfidf))

    # 3. Hand-crafted metadata + text features
    if use_feature_engineering:
        meta_feats = Pipeline([
            ("meta_extract", MetadataAndTextFeatureExtractor()),
            ("scaler", MinMaxScaler()),
        ])
        features_list.append(("meta_features", meta_feats))

    preprocessor = FeatureUnion(features_list)

    # Classifier selection
    if model_type == "logistic_regression":
        default_params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "C": 1.0,
            "random_state": 42,
        }
        default_params.update(clf_params)
        clf = LogisticRegression(**default_params)
    elif model_type in ("linear_svm", "linear_svc"):
        default_params = {
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42,
            "C": 1.0,
            "dual": False,
        }
        default_params.update(clf_params)
        clf = LinearSVC(**default_params)
    elif model_type == "sgd":
        default_params = {
            "loss": "log_loss",
            "max_iter": 1000,
            "class_weight": "balanced",
            "penalty": "l2",
            "random_state": 42,
        }
        default_params.update(clf_params)
        clf = SGDClassifier(**default_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose from 'logistic_regression', 'linear_svm', 'sgd'.")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
