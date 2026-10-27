"""Prediction module to load saved model and generate predictions on new comments or test files."""

import argparse
from pathlib import Path
from typing import Any, List, Optional, Union
import joblib
import pandas as pd

from src.data import clean_dataframe, load_data, validate_data


def load_model(model_path: Union[str, Path]) -> Any:
    """Load serialized scikit-learn pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(path)


def predict_comments(model: Any, comments: Union[str, List[str]]) -> List[int]:
    """Predict label categories for one or more raw comments.

    Args:
        model: Trained pipeline object.
        comments: Single comment string or list of comment strings.

    Returns:
        List of integer predicted labels (0, 1, 2, or 3).
    """
    if isinstance(comments, str):
        comments = [comments]

    df_input = pd.DataFrame({"comment": comments})
    preds = model.predict(df_input)
    return [int(p) for p in preds]


def predict_test_file(
    model_path: str,
    test_path: str = "data/test.csv",
    output_path: str = "submission.csv",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Generate predictions for a test dataset and output standard submission CSV.

    Args:
        model_path: Path to serialized model pipeline.
        test_path: Path to test dataset CSV.
        output_path: Path to write submission CSV.
        nrows: Optional row limit.

    Returns:
        pd.DataFrame containing 'ID' and 'label'.
    """
    model = load_model(model_path)
    raw_test = load_data(test_path, nrows=nrows)
    validate_data(raw_test, is_train=False)
    cleaned_test = clean_dataframe(raw_test)

    preds = model.predict(cleaned_test)

    submission = pd.DataFrame({
        "ID": range(1, len(preds) + 1),
        "label": [int(p) for p in preds],
    })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_file, index=False)

    print(f"Generated predictions for {len(submission):,} rows.")
    print(f"Saved submission file to: {out_file}")
    print("\nPredicted Class Distribution:")
    for lbl, count in submission["label"].value_counts().sort_index().items():
        print(f"  Label {lbl}: {count:>6,} ({count / len(submission) * 100:.1f}%)")

    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict comment category labels")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved pipeline .joblib")
    parser.add_argument("--test-file", type=str, default=None, help="Path to test CSV file")
    parser.add_argument("--output-csv", type=str, default="submission.csv", help="Output submission CSV path")
    parser.add_argument("--comment", type=str, default=None, help="Single comment text to predict")
    parser.add_argument("--nrows", type=int, default=None, help="Row limit for test file")
    args = parser.parse_args()

    if args.comment:
        model = load_model(args.model_path)
        pred = predict_comments(model, args.comment)[0]
        label_names = {
            0: "Neutral / Informative",
            1: "Mildly Toned",
            2: "Moderately Negative",
            3: "Targeted / Aggressive",
        }
        print(f"\nComment   : \"{args.comment}\"")
        print(f"Predicted : Label {pred} ({label_names.get(pred, 'Unknown')})")
    elif args.test_file:
        predict_test_file(
            model_path=args.model_path,
            test_path=args.test_file,
            output_path=args.output_csv,
            nrows=args.nrows,
        )
    else:
        print("Please provide either --test-file or --comment. Use -h for help.")


if __name__ == "__main__":
    main()
