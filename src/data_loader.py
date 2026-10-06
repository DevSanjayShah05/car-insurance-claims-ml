"""
Data loading and preprocessing module for Car Insurance Claim Outcomes.
Handles data ingestion, missing value imputation (median), and train/test splits.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_insurance.csv"


def load_and_preprocess_data(
    filepath: Path = DATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads car insurance dataset, imputes missing values using medians,
    and returns stratified train/test feature matrices and target vectors.
    """
    df = pd.read_csv(filepath)

    # Impute missing values with column medians (industry standard benchmark)
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].fillna(df["credit_score"].median())

    if "annual_mileage" in df.columns:
        df["annual_mileage"] = df["annual_mileage"].fillna(df["annual_mileage"].median())

    # Separate features and target (exclude unique identifier 'id')
    feature_cols = [col for col in df.columns if col not in ["id", "outcome"]]
    X = df[feature_cols]
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, df
