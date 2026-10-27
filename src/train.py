"""Training module integrating scikit-learn Pipeline and MLflow tracking."""

import argparse
from pathlib import Path
import json
from typing import Any, Dict, Optional
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from src.data import clean_dataframe, get_train_val_split, load_data, validate_data
from src.pipeline import create_pipeline


def train_and_track(
    data_path: str = "data/train.csv",
    model_type: str = "logistic_regression",
    use_feature_engineering: bool = True,
    use_dual_tfidf: bool = True,
    word_max_features: int = 15000,
    char_max_features: int = 10000,
    classifier_params: Optional[Dict[str, Any]] = None,
    experiment_name: str = "comment-category-prediction",
    run_name: Optional[str] = None,
    cv_folds: int = 0,
    nrows: Optional[int] = None,
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Train pipeline, evaluate performance, and log run details to MLflow.

    Args:
        data_path: Path to train.csv.
        model_type: Classifier architecture name ('logistic_regression', 'linear_svm', 'sgd').
        use_feature_engineering: Include engineered text & metadata features.
        use_dual_tfidf: Include char n-gram TF-IDF in addition to word TF-IDF.
        word_max_features: Max features for word TF-IDF.
        char_max_features: Max features for char TF-IDF.
        classifier_params: Additional hyperparameters for the classifier.
        experiment_name: MLflow experiment name.
        run_name: Optional descriptive run name.
        cv_folds: Number of StratifiedKFold splits (0 for simple 80/20 train/val split).
        nrows: Optional row limit for fast experimentation.
        models_dir: Path to directory for saving model binaries.
        reports_dir: Path to directory for saving evaluation reports.

    Returns:
        Dictionary of validation metrics.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    raw_df = load_data(data_path, nrows=nrows)
    validate_data(raw_df, is_train=True)
    df = clean_dataframe(raw_df)

    if not run_name:
        fe_suffix = "with_fe" if use_feature_engineering else "baseline"
        run_name = f"{model_type}_{fe_suffix}"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # 2. Build Pipeline
        pipeline = create_pipeline(
            model_type=model_type,
            use_feature_engineering=use_feature_engineering,
            use_dual_tfidf=use_dual_tfidf,
            word_max_features=word_max_features,
            char_max_features=char_max_features,
            classifier_params=classifier_params,
        )

        # Log parameters
        params_to_log = {
            "model_type": model_type,
            "use_feature_engineering": use_feature_engineering,
            "use_dual_tfidf": use_dual_tfidf,
            "word_max_features": word_max_features,
            "char_max_features": char_max_features if use_dual_tfidf else 0,
            "nrows": len(df),
            "cv_folds": cv_folds,
        }
        if classifier_params:
            for k, v in classifier_params.items():
                params_to_log[f"clf__{k}"] = v
        mlflow.log_params(params_to_log)

        # 3. Validation Strategy
        if cv_folds > 1:
            val_strategy = f"StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state=42)"
            mlflow.log_param("validation_strategy", val_strategy)

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            X = df.drop(columns=["label"])
            y = df["label"].astype(int)

            fold_f1_macro = []
            fold_acc = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

                fold_pipe = create_pipeline(
                    model_type=model_type,
                    use_feature_engineering=use_feature_engineering,
                    use_dual_tfidf=use_dual_tfidf,
                    word_max_features=word_max_features,
                    char_max_features=char_max_features,
                    classifier_params=classifier_params,
                )
                fold_pipe.fit(X_tr, y_tr)
                preds = fold_pipe.predict(X_va)
                f1_m = f1_score(y_val, preds, average="macro")
                acc = accuracy_score(y_val, preds)
                fold_f1_macro.append(f1_m)
                fold_acc.append(acc)

            val_macro_f1 = float(np.mean(fold_f1_macro))
            val_accuracy = float(np.mean(fold_acc))
            val_f1_std = float(np.std(fold_f1_macro))

            # Train final model on full available training set
            pipeline.fit(X, y)
            y_val_preds = pipeline.predict(X)
            y_true_eval = y
        else:
            val_strategy = "Stratified 80/20 train/validation split (random_state=42)"
            mlflow.log_param("validation_strategy", val_strategy)

            X_train, X_val, y_train, y_val = get_train_val_split(df, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            y_val_preds = pipeline.predict(X_val)
            y_true_eval = y_val

            val_macro_f1 = float(f1_score(y_val, y_val_preds, average="macro"))
            val_accuracy = float(accuracy_score(y_val, y_val_preds))
            val_f1_std = 0.0

        val_weighted_f1 = float(f1_score(y_true_eval, y_val_preds, average="weighted"))

        # Per-class F1 metrics
        per_class_f1 = f1_score(y_true_eval, y_val_preds, average=None)
        metrics = {
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_accuracy,
            "val_weighted_f1": val_weighted_f1,
            "val_f1_std": val_f1_std,
        }
        for label_idx, score in enumerate(per_class_f1):
            metrics[f"val_f1_class_{label_idx}"] = float(score)

        mlflow.log_metrics(metrics)

        # 4. Generate & Save Artifacts
        model_filename = f"{run_name}.joblib"
        model_filepath = Path(models_dir) / model_filename
        joblib.dump(pipeline, model_filepath)
        mlflow.log_artifact(str(model_filepath))

        # Classification report artifact
        cls_report = classification_report(y_true_eval, y_val_preds, digits=4)
        report_path = Path(reports_dir) / f"{run_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(cls_report)
        mlflow.log_artifact(str(report_path))

        # Confusion Matrix artifact
        cm = confusion_matrix(y_true_eval, y_val_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va="center", ha="center")
        ax.set_title(f"Confusion Matrix: {run_name}", pad=20)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        cm_path = Path(reports_dir) / f"{run_name}_confusion_matrix.png"
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        print(f"\n================ Run: {run_name} ================")
        print(f"Validation Strategy : {val_strategy}")
        print(f"Validation Macro-F1 : {val_macro_f1:.4f}")
        print(f"Validation Accuracy : {val_accuracy:.4f}")
        print(f"Per-class F1-scores : {[round(x, 4) for x in per_class_f1]}")
        print(f"Model saved to      : {model_filepath}")
        print(f"MLflow Run ID       : {mlflow.active_run().info.run_id}")

        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train comment category prediction pipeline")
    parser.add_argument("--data-path", type=str, default="data/train.csv")
    parser.add_argument("--model-type", type=str, default="logistic_regression", choices=["logistic_regression", "linear_svm", "sgd"])
    parser.add_argument("--no-fe", action="store_true", help="Disable feature engineering")
    parser.add_argument("--no-dual-tfidf", action="store_true", help="Disable char TF-IDF")
    parser.add_argument("--word-max-features", type=int, default=15000)
    parser.add_argument("--char-max-features", type=int, default=10000)
    parser.add_argument("--cv", type=int, default=0, help="Number of CV folds (0 for 80/20 train/val split)")
    parser.add_argument("--nrows", type=int, default=None, help="Row limit for fast testing")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="comment-category-prediction")
    args = parser.parse_args()

    train_and_track(
        data_path=args.data_path,
        model_type=args.model_type,
        use_feature_engineering=not args.no_fe,
        use_dual_tfidf=not args.no_dual_tfidf,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        cv_folds=args.cv,
        nrows=args.nrows,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
