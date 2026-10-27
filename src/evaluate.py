"""Evaluation module to calculate metrics and inspect model errors."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.data import clean_dataframe, load_data, validate_data


def evaluate_model(
    model_path: str,
    data_path: str,
    nrows: Optional[int] = None,
    output_report_path: Optional[str] = None,
    num_error_samples: int = 5,
) -> Dict[str, Any]:
    """Evaluate saved model pipeline on dataset and perform error inspection.

    Args:
        model_path: Path to serialized pipeline (.joblib).
        data_path: Path to dataset with 'comment' and 'label'.
        nrows: Maximum rows to evaluate.
        output_report_path: Optional path to save JSON metrics report.
        num_error_samples: Number of sample misclassifications per error type to display.

    Returns:
        Dictionary of computed metrics and error analysis.
    """
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_file)

    raw_df = load_data(data_path, nrows=nrows)
    validate_data(raw_df, is_train=True)
    df = clean_dataframe(raw_df)

    y_true = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"])

    y_pred = model.predict(X)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, output_dict=True)

    # Inspect errors
    errors_mask = y_true != y_pred
    total_errors = int(np.sum(errors_mask))
    error_rate = float(total_errors / len(y_true))

    error_df = df[errors_mask].copy()
    error_df["true_label"] = y_true[errors_mask]
    error_df["predicted_label"] = y_pred[errors_mask]

    # Find most common error confusion pairs
    confusion_pairs: List[Dict[str, Any]] = []
    for (true_lbl, pred_lbl), count in (
        error_df.groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False).items()
    ):
        samples = (
            error_df[(error_df["true_label"] == true_lbl) & (error_df["predicted_label"] == pred_lbl)]["comment"]
            .head(num_error_samples)
            .tolist()
        )
        confusion_pairs.append({
            "true_label": int(true_lbl),
            "predicted_label": int(pred_lbl),
            "count": int(count),
            "sample_comments": samples,
        })

    results = {
        "model_path": str(model_path),
        "total_samples": len(y_true),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "error_count": total_errors,
        "error_rate": error_rate,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "top_confusion_pairs": confusion_pairs[:5],
    }

    print("\n" + "=" * 50)
    print("           MODEL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Evaluated Samples : {len(y_true):,}")
    print(f"Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"Weighted-F1       : {weighted_f1:.4f}")
    print(f"Total Errors      : {total_errors:,} ({error_rate*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nTop Error Confusion Patterns:")
    for pair in confusion_pairs[:3]:
        print(f" - True {pair['true_label']} -> Predicted {pair['predicted_label']}: {pair['count']} instances")
        if pair["sample_comments"]:
            print(f"   Sample: \"{pair['sample_comments'][0][:80]}...\"")

    if output_report_path:
        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed evaluation report to: {out_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model and inspect error cases")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved pipeline .joblib")
    parser.add_argument("--data-path", type=str, default="data/train.csv", help="Dataset to evaluate on")
    parser.add_argument("--nrows", type=int, default=None, help="Row limit for quick evaluation")
    parser.add_argument("--output-report", type=str, default="reports/evaluation_summary.json")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        nrows=args.nrows,
        output_report_path=args.output_report,
    )


if __name__ == "__main__":
    main()
