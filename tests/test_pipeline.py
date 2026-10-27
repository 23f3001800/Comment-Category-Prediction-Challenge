"""Unit tests for the comment classification pipeline."""

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import clean_dataframe, load_data, validate_data
from src.pipeline import create_pipeline
from src.predict import predict_comments


@pytest.fixture
def sample_train_df():
    """Provides a synthetic valid training DataFrame."""
    return pd.DataFrame({
        "comment": [
            "This is a great informative article.",
            "I completely disagree, you are totally wrong!",
            "Neutral discussion point about the news.",
            "Terrible hate speech attacking people.",
            "Can anyone explain this formula to me?",
            "Awful nonsense, ridiculous statement.",
        ],
        "upvote": [10, 0, 2, 0, 5, 1],
        "downvote": [0, 8, 1, 15, 0, 4],
        "if_1": [0, 1, 0, 2, 0, 1],
        "if_2": [1, 8, 2, 12, 1, 7],
        "emoticon_3": [0, 2, 0, 0, 1, 0],
        "disability": [False, False, False, True, False, False],
        "label": [0, 2, 0, 3, 0, 1],
    })


@pytest.fixture
def fitted_pipeline(sample_train_df):
    """Provides a fitted pipeline ready for inference testing."""
    pipe = create_pipeline(
        model_type="logistic_regression",
        use_feature_engineering=True,
        use_dual_tfidf=True,
        word_max_features=50,
        char_max_features=50,
        min_df=1,
    )
    y = sample_train_df["label"]
    X = sample_train_df.drop(columns=["label"])
    pipe.fit(X, y)
    return pipe


class TestValidInputs:
    """Test valid inputs to data validation and pipeline."""

    def test_valid_dataframe_fit_and_predict(self, sample_train_df):
        pipe = create_pipeline(
            model_type="logistic_regression",
            word_max_features=50,
            char_max_features=50,
            min_df=1,
        )
        X = sample_train_df.drop(columns=["label"])
        y = sample_train_df["label"]
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(sample_train_df)

    def test_text_only_dataframe(self, fitted_pipeline):
        test_df = pd.DataFrame({"comment": ["Hello world", "Another post"]})
        preds = fitted_pipeline.predict(test_df)
        assert len(preds) == 2

    def test_raw_comment_list_predict(self, fitted_pipeline):
        preds = predict_comments(fitted_pipeline, ["Great comment", "Angry shout!"])
        assert len(preds) == 2
        assert isinstance(preds[0], int)


class TestInvalidInputs:
    """Test handling of invalid inputs and edge cases."""

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_data(pd.DataFrame(), is_train=True)

    def test_missing_required_column_train(self):
        df_missing = pd.DataFrame({"comment": ["Test text"]})
        with pytest.raises(ValueError, match="Missing required columns.*label"):
            validate_data(df_missing, is_train=True)

    def test_missing_required_column_test(self):
        df_missing = pd.DataFrame({"upvote": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns.*comment"):
            validate_data(df_missing, is_train=False)

    def test_invalid_label_values(self):
        df_bad_label = pd.DataFrame({
            "comment": ["Good comment", "Bad comment"],
            "label": [0, 99],
        })
        with pytest.raises(ValueError, match="invalid target labels"):
            validate_data(df_bad_label, is_train=True)

    def test_nan_comment_handling(self, fitted_pipeline):
        df_nan = pd.DataFrame({"comment": [None, np.nan, ""]})
        # Pipeline should handle null comment without crashing
        preds = fitted_pipeline.predict(df_nan)
        assert len(preds) == 3


class TestPredictionShapeAndOutputLabels:
    """Test output shapes and label constraints."""

    @pytest.mark.parametrize("n_samples", [1, 4, 10])
    def test_prediction_shape(self, fitted_pipeline, n_samples):
        df = pd.DataFrame({"comment": [f"Sample comment number {i}" for i in range(n_samples)]})
        preds = fitted_pipeline.predict(df)
        assert preds.shape == (n_samples,)

    def test_output_labels_subset(self, fitted_pipeline):
        test_comments = pd.DataFrame({
            "comment": [
                "Clear news report",
                "Offensive attack against users",
                "Casual remark",
                "Mildly critical feedback",
            ]
        })
        preds = fitted_pipeline.predict(test_comments)
        for pred in preds:
            assert pred in {0, 1, 2, 3}


class TestPipelineLeakagePreventionAndPersistence:
    """Ensure pipeline encapsulates preprocessing and can be saved/loaded."""

    def test_pipeline_roundtrip_serialization(self, fitted_pipeline, sample_train_df):
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            joblib.dump(fitted_pipeline, tmp.name)
            loaded_pipe = joblib.load(tmp.name)

            preds_orig = fitted_pipeline.predict(sample_train_df)
            preds_loaded = loaded_pipe.predict(sample_train_df)
            np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_linear_svm_pipeline(self, sample_train_df):
        pipe_svm = create_pipeline(
            model_type="linear_svm",
            word_max_features=50,
            char_max_features=50,
            min_df=1,
        )
        X = sample_train_df.drop(columns=["label"])
        y = sample_train_df["label"]
        pipe_svm.fit(X, y)
        preds = pipe_svm.predict(X)
        assert len(preds) == len(X)
