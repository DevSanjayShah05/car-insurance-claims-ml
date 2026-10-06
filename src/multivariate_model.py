"""
Multivariate Logistic Regression & Risk Analytics Module.
Fits a comprehensive regularized logistic regression model, computes odds ratios (OR)
with 95% confidence intervals, evaluates diagnostic metrics, and segments risk deciles.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def train_multivariate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Trains scaled multivariate logistic regression, extracts odds ratios and risk deciles.
    """
    # 1. Pipeline with StandardScaler and L2 Logistic Regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000, C=1.0))
    ])
    pipeline.fit(X_train, y_train)

    # 2. Predictions & Diagnostic Metrics
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    # 3. Statistical Inference with Statsmodels for exact Odds Ratios and 95% CIs
    X_train_sm = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_train_sm).fit(disp=False)

    params = logit_model.params
    conf = logit_model.conf_int()
    conf.columns = ["CI_lower", "CI_upper"]

    odds_ratios = pd.DataFrame({
        "feature": params.index,
        "coefficient": params.values,
        "odds_ratio": np.exp(params.values),
        "or_ci_lower": np.exp(conf["CI_lower"].values),
        "or_ci_upper": np.exp(conf["CI_upper"].values),
        "p_value": logit_model.pvalues.values
    }).query("feature != 'const'").sort_values(by="odds_ratio", ascending=False).reset_index(drop=True)

    # 4. Risk Decile Segmentation on Test Set
    df_eval = pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob})
    df_eval["decile"] = pd.qcut(df_eval["y_prob"], q=10, labels=False, duplicates="drop")
    # Invert decile so decile 1 is highest risk
    df_eval["risk_tier"] = 10 - df_eval["decile"]

    decile_summary = df_eval.groupby("risk_tier").agg(
        total_policies=("y_true", "count"),
        actual_claims=("y_true", "sum"),
        observed_claim_rate=("y_true", "mean"),
        predicted_claim_rate=("y_prob", "mean")
    ).reset_index().sort_values(by="risk_tier")

    decile_summary["observed_claim_rate"] = decile_summary["observed_claim_rate"].round(4)
    decile_summary["predicted_claim_rate"] = decile_summary["predicted_claim_rate"].round(4)
    baseline_rate = y_test.mean()
    decile_summary["lift"] = (decile_summary["observed_claim_rate"] / baseline_rate).round(2)

    # 5. Save model artifact
    model_path = ARTIFACTS_DIR / "best_car_insurance_model.joblib"
    joblib.dump(pipeline, model_path)

    return pipeline, metrics, odds_ratios, decile_summary
