"""
Unit and Integration Tests for Car Insurance Claim Outcomes ML Pipeline.
"""

from pathlib import Path
import pytest
import pandas as pd
import joblib

from src.data_loader import load_and_preprocess_data
from src.single_feature_evaluation import evaluate_single_features
from src.multivariate_model import train_multivariate_model
from src.visualizations import InsuranceVisualizer


@pytest.fixture(scope="session")
def processed_data():
    return load_and_preprocess_data()


def test_data_loader_imputation(processed_data):
    X_train, X_test, y_train, y_test, df_raw = processed_data
    assert len(df_raw) == 10000
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert set(X_train.columns) == set(X_test.columns)
    assert "id" not in X_train.columns
    assert "outcome" not in X_train.columns


def test_single_feature_evaluation(processed_data):
    X_train, X_test, y_train, y_test, _ = processed_data
    results_df, best_feature, best_accuracy = evaluate_single_features(X_train, X_test, y_train, y_test)

    assert len(results_df) == X_train.shape[1]
    assert best_feature == "driving_experience"
    assert best_accuracy >= 0.70
    assert "roc_auc" in results_df.columns


def test_multivariate_model_performance(processed_data):
    X_train, X_test, y_train, y_test, _ = processed_data
    pipeline, metrics, odds_df, decile_df = train_multivariate_model(X_train, X_test, y_train, y_test)

    assert metrics["accuracy"] >= 0.75
    assert metrics["roc_auc"] >= 0.80
    assert metrics["f1_score"] >= 0.70
    assert len(odds_df) == X_train.shape[1]
    assert len(decile_df) == 10

    # Driving experience should be protective (OR < 1.0)
    exp_or = odds_df[odds_df["feature"] == "driving_experience"]["odds_ratio"].values[0]
    assert exp_or < 1.0

    # Past accidents should increase risk (OR > 1.0)
    acc_or = odds_df[odds_df["feature"] == "past_accidents"]["odds_ratio"].values[0]
    assert acc_or > 1.0

    # Model artifact saved
    model_file = Path("artifacts/best_car_insurance_model.joblib")
    assert model_file.exists()
    loaded_model = joblib.load(model_file)
    assert hasattr(loaded_model, "predict")


def test_visualization_generation(processed_data, tmp_path):
    X_train, X_test, y_train, y_test, df_raw = processed_data
    results_df, best_feature, _ = evaluate_single_features(X_train, X_test, y_train, y_test)
    pipeline, _, odds_df, decile_df = train_multivariate_model(X_train, X_test, y_train, y_test)

    viz = InsuranceVisualizer(artifacts_dir=tmp_path)
    artifacts = viz.generate_all(df_raw, results_df, best_feature, odds_df, pipeline, X_test, y_test, decile_df)

    assert len(artifacts) == 5
    for k, p in artifacts.items():
        path = Path(p)
        assert path.exists()
        assert path.stat().st_size > 10000
