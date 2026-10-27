"""Data loading and validation module for comment category prediction."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_TEST_COLUMNS = ["comment"]
REQUIRED_TRAIN_COLUMNS = ["comment", "label"]
VALID_LABELS = {0, 1, 2, 3}


def load_data(file_path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load dataset from CSV file.

    Args:
        file_path: Path to the CSV file.
        nrows: Optional maximum number of rows to read.

    Returns:
        pd.DataFrame containing the loaded data.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_csv(path, nrows=nrows)
    if df.empty:
        raise ValueError(f"Loaded file {file_path} is empty.")
    return df


def validate_data(df: pd.DataFrame, is_train: bool = True) -> bool:
    """Perform basic schema and content validation on the dataframe.

    Args:
        df: Input DataFrame to validate.
        is_train: True if validating training data (with 'label'), False for test/inference.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")

    if len(df) == 0:
        raise ValueError("DataFrame is empty (0 rows).")

    required_cols = REQUIRED_TRAIN_COLUMNS if is_train else REQUIRED_TEST_COLUMNS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Found: {list(df.columns)}")

    if is_train:
        # Check target labels
        non_null_labels = df["label"].dropna()
        if len(non_null_labels) == 0:
            raise ValueError("The 'label' column contains only null values.")

        unique_labels = set(non_null_labels.unique())
        invalid_labels = unique_labels - VALID_LABELS
        if invalid_labels:
            raise ValueError(f"Found invalid target labels: {invalid_labels}. Expected subset of {VALID_LABELS}")

    return True


def clean_dataframe(df: pd.DataFrame, drop_sparse: bool = True) -> pd.DataFrame:
    """Perform basic cleaning such as filling missing comments and dropping high-null columns.

    Args:
        df: Input DataFrame.
        drop_sparse: Whether to drop known high-null/zero-variance metadata columns.

    Returns:
        Cleaned copy of the DataFrame.
    """
    df_clean = df.copy()

    # Fill missing comment text with empty string
    if "comment" in df_clean.columns:
        df_clean["comment"] = df_clean["comment"].fillna("").astype(str)

    if drop_sparse:
        cols_to_drop = ["race", "religion", "gender", "emoticon_2"]
        df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], inplace=True, errors="ignore")

    return df_clean


def get_train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and validation sets with stratification.

    Args:
        df: Training DataFrame containing features and 'label'.
        test_size: Fraction of samples for validation.
        random_state: Random state seed.
        stratify: Whether to perform stratified split based on target.

    Returns:
        (X_train, X_val, y_train, y_val)
    """
    validate_data(df, is_train=True)
    df_clean = clean_dataframe(df)

    y = df_clean["label"].astype(int)
    X = df_clean.drop(columns=["label"])

    stratify_target = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
    )
    return X_train, X_val, y_train, y_val
