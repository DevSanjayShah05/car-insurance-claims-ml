"""
Single Feature Logistic Regression Evaluation Module.
Evaluates the predictive power of each individual feature on car insurance claim outcomes
to identify the single most accurate predictor.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_single_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, str, float]:
    """
    Fits a simple logistic regression model for each individual feature,
    calculates test accuracy and ROC-AUC, and identifies the best performing feature.
    """
    results = []

    for col in X_train.columns:
        # Fit model on single feature
        model = LogisticRegression(random_state=42, solver="lbfgs")
        model.fit(X_train[[col]], y_train)

        # Predictions
        preds = model.predict(X_test[[col]])
        probs = model.predict_proba(X_test[[col]])[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results.append({
            "feature": col,
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4)
        })

    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_feature = str(best_row["feature"])
    best_accuracy = float(best_row["accuracy"])

    return results_df, best_feature, best_accuracy


if __name__ == "__main__":
    from src.data_loader import load_and_preprocess_data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    df_res, best_feat, best_acc = evaluate_single_features(X_train, X_test, y_train, y_test)
    print("Top Single Predictors:")
    print(df_res.head())
    print(f"\nBest Feature: {best_feat} with Accuracy: {best_acc:.4f}")
