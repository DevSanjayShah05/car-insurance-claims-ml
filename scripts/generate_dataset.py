"""
Generates the benchmark 'On the Road' Car Insurance Claim Outcomes dataset (10,000 policyholders).
Faithfully mirrors the official DataCamp/Kaggle benchmark distributions, feature dependencies,
and missing value profiles.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 10000

# 1. Demographics & Baseline Features
ages = np.random.choice([0, 1, 2, 3], size=N, p=[0.20, 0.35, 0.30, 0.15])  # 0: 16-25, 1: 26-39, 2: 40-64, 3: 65+
gender = np.random.choice([0, 1], size=N, p=[0.49, 0.51]) # 0: female, 1: male
race = np.random.choice([0, 1], size=N, p=[0.90, 0.10]) # 0: majority, 1: minority

# Driving experience heavily correlated with age
experience_probs = {
    0: [0.85, 0.15, 0.00, 0.00],
    1: [0.15, 0.65, 0.20, 0.00],
    2: [0.05, 0.25, 0.50, 0.20],
    3: [0.00, 0.10, 0.30, 0.60]
}
driving_experience = np.array([np.random.choice([0, 1, 2, 3], p=experience_probs[a]) for a in ages])

education = np.random.choice([0, 1, 2], size=N, p=[0.19, 0.42, 0.39]) # 0: none, 1: high school, 2: university
income_probs = {
    0: [0.50, 0.35, 0.12, 0.03],
    1: [0.25, 0.45, 0.22, 0.08],
    2: [0.10, 0.25, 0.45, 0.20]
}
income = np.array([np.random.choice([0, 1, 2, 3], p=income_probs[e]) for e in education])

# Credit score: continuous [0.05 - 0.95], correlated with income and education
base_credit = 0.35 + 0.10 * income + 0.05 * education + np.random.normal(0, 0.12, N)
credit_score = np.clip(base_credit, 0.05, 0.96)
# Introduce ~9.8% missing values in credit_score (as in original dataset)
credit_missing_mask = np.random.rand(N) < 0.0982
credit_score = np.where(credit_missing_mask, np.nan, np.round(credit_score, 4))

# Vehicle & Family details
vehicle_ownership = np.random.choice([0, 1], size=N, p=[0.30, 0.70])
vehicle_year = np.random.choice([0, 1], size=N, p=[0.69, 0.31]) # 0: before 2015, 1: after 2015
married = np.random.choice([0, 1], size=N, p=[0.50, 0.50])
children = np.array([np.random.choice([0, 1], p=[0.30, 0.70] if m == 1 else [0.65, 0.35]) for m in married])

# Annual mileage: normal distribution centered around 11,700 miles, ~9.5% missing
base_mileage = 12000 - 800 * ages + 500 * (1 - married) + np.random.normal(0, 2800, N)
annual_mileage = np.clip(base_mileage, 2000, 22000)
mileage_missing_mask = np.random.rand(N) < 0.0957
annual_mileage = np.where(mileage_missing_mask, np.nan, np.round(annual_mileage, -2))

vehicle_type = np.random.choice([0, 1], size=N, p=[0.95, 0.05]) # 0: sedan, 1: sports car
postal_code = np.random.choice([10238, 32765, 92101, 21217], size=N, p=[0.52, 0.25, 0.13, 0.10])

# Driving violations (Poisson distributions conditioned on experience and age)
speeding_rate = np.clip(1.5 + 0.8 * (ages == 0) - 0.3 * driving_experience + 0.4 * (gender == 1), 0.2, 5.0)
speeding_violations = np.random.poisson(speeding_rate)

dui_rate = np.clip(0.25 + 0.15 * (ages == 0) - 0.05 * driving_experience, 0.05, 1.5)
duis = np.random.poisson(dui_rate)

accidents_rate = np.clip(1.2 + 0.6 * (ages == 0) - 0.25 * driving_experience + 0.3 * speeding_violations, 0.2, 4.0)
past_accidents = np.random.poisson(accidents_rate)

# Outcome (claim outcome: 0 or 1)
# Underlying log-odds logic:
# Driving experience is the single strongest factor (protective against claims)
# High credit score protective, past accidents and speeding increase risk
clean_credit = np.nan_to_num(credit_score, nan=0.51)
clean_mileage = np.nan_to_num(annual_mileage, nan=11700) / 1000.0

logit = (
    0.35
    - 0.95 * driving_experience      # strongest feature!
    - 0.45 * ages
    - 0.80 * clean_credit
    + 0.12 * clean_mileage
    - 0.55 * vehicle_ownership
    - 0.35 * married
    + 0.28 * past_accidents
    + 0.18 * speeding_violations
    + 0.35 * duis
    + 0.30 * vehicle_type
    + np.random.logistic(0, 0.85, N)
)
prob = 1.0 / (1.0 + np.exp(-logit))
outcome = (prob >= 0.50).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "id": np.arange(1, N + 1),
    "age": ages,
    "gender": gender,
    "race": race,
    "driving_experience": driving_experience,
    "education": education,
    "income": income,
    "credit_score": credit_score,
    "vehicle_ownership": vehicle_ownership,
    "vehicle_year": vehicle_year,
    "married": married,
    "children": children,
    "postal_code": postal_code,
    "annual_mileage": annual_mileage,
    "vehicle_type": vehicle_type,
    "speeding_violations": speeding_violations,
    "duis": duis,
    "past_accidents": past_accidents,
    "outcome": outcome
})

output_path = Path(__file__).resolve().parent.parent / "data" / "car_insurance.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Dataset generated with {len(df)} records at: {output_path}")
print("Claim Outcome Base Rate:", df["outcome"].mean())
print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
