"""
Car Insurance Claim Outcomes: Master Pipeline & Analysis Runner.
Executes end-to-end data processing, single-feature evaluation, multivariate modeling,
odds ratio statistical inference, and publication-grade visualization generation.
"""

import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_preprocess_data
from src.single_feature_evaluation import evaluate_single_features
from src.multivariate_model import train_multivariate_model
from src.visualizations import InsuranceVisualizer


def run_pipeline():
    print("\n" + "=" * 80)
    print("🚗 MODELING CAR INSURANCE CLAIM OUTCOMES (ON THE ROAD CAR INSURANCE)")
    print("=" * 80)
    start_time = time.time()

    # Step 1: Load and Preprocess
    print("\n[STEP 1/4] 📂 LOADING DATASET & IMPUTING MISSING VALUES")
    X_train, X_test, y_train, y_test, df_raw = load_and_preprocess_data()
    print(f"  ✓ Loaded {len(df_raw):,} records with {X_train.shape[1]} explanatory features")
    print(f"  ✓ Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"  ✓ Portfolio Claim Base Rate: {y_train.mean():.2%}")
    print("  ✓ Imputed missing credit_score & annual_mileage with column medians")

    # Step 2: Single Feature Logistic Regression Evaluation
    print("\n[STEP 2/4] 🔍 EVALUATING INDIVIDUAL FEATURE PREDICTIVE POWER")
    results_df, best_feature, best_accuracy = evaluate_single_features(X_train, X_test, y_train, y_test)
    print("  ✓ Top 5 Single Predictors by Test Accuracy:")
    for idx, row in results_df.head(5).iterrows():
        print(f"     {idx+1}. {row['feature']:<22} Accuracy: {row['accuracy']:.4f} | ROC-AUC: {row['roc_auc']:.4f}")

    best_feature_df = pd.DataFrame({
        "best_feature": [best_feature],
        "best_accuracy": [best_accuracy]
    })
    print("\n  ★ OFFICIAL RESULT: BEST SINGLE PREDICTOR")
    print(best_feature_df.to_string(index=False))

    # Step 3: Multivariate Regularized Logistic Regression & Odds Ratios
    print("\n[STEP 3/4] 🧠 TRAINING MULTIVARIATE MODEL & ODDS RATIOS INFERENCE")
    pipeline, metrics, odds_df, decile_df = train_multivariate_model(X_train, X_test, y_train, y_test)
    print(f"  ✓ Multivariate Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  ✓ Multivariate ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"  ✓ Multivariate Precision:      {metrics['precision']:.4f}")
    print(f"  ✓ Multivariate Recall:         {metrics['recall']:.4f}")
    print(f"  ✓ Multivariate F1-Score:       {metrics['f1_score']:.4f}")
    print(f"  ✓ Brier Score (Calibration):   {metrics['brier_score']:.4f}")

    print("\n  ✓ Top Risk Drivers (Odds Ratio > 1.0):")
    for _, row in odds_df[odds_df["odds_ratio"] > 1.0].head(4).iterrows():
        print(f"     • {row['feature']:<20} OR = {row['odds_ratio']:.2f} (p-value: {row['p_value']:.4e})")

    print("\n  ✓ Top Protective Factors (Odds Ratio < 1.0):")
    for _, row in odds_df[odds_df["odds_ratio"] < 1.0].tail(4).iterrows():
        print(f"     • {row['feature']:<20} OR = {row['odds_ratio']:.2f} (p-value: {row['p_value']:.4e})")

    # Step 4: Visualizations
    print("\n[STEP 4/4] 📊 GENERATING PUBLICATION-GRADE VISUALIZATION ARTIFACTS")
    viz = InsuranceVisualizer()
    artifacts = viz.generate_all(df_raw, results_df, best_feature, odds_df, pipeline, X_test, y_test, decile_df)
    for name, path in artifacts.items():
        print(f"  ✓ Saved artifact: {Path(path).name}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 80 + "\n")
    return best_feature_df, metrics


if __name__ == "__main__":
    run_pipeline()
