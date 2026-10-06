# 🚗 Modeling Car Insurance Claim Outcomes: Predictive Risk Modeling & Actuarial Analytics

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-Inference-00599C)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end predictive machine learning and actuarial analytics project based on historical policyholder data from **On the Road Car Insurance** (10,000 policyholders).

This project identifies the single most accurate predictor of claim outcomes, fits regularized multivariate logistic regression models, conducts statistical Odds Ratio ($OR = e^{\beta}$) inference with 95% confidence intervals, and constructs an actuarial risk decile segmentation pipeline for underwriting risk scoring.

---

## 🎯 Executive Summary & Official Benchmark

| Task | Metric / Result | Business Impact |
| :--- | :--- | :--- |
| **Best Single Feature** | **`driving_experience`** | Outperforms all other single variables; provides the strongest individual predictive signal for claim incidence. |
| **Best Single Accuracy** | **`74.70%`** (ROC-AUC: `0.8355`) | Demonstrates that tenure and road experience heavily suppress collision and claim probability. |
| **Multivariate Accuracy** | **`79.65%`** (ROC-AUC: `0.8792`) | Integrating driver demographics, vehicle ownership, credit score, and violation history improves classification accuracy. |
| **Underwriting Decile Lift** | **`2.15x`** in Top Risk Decile | Risk decile 1 exhibits $>2\times$ average baseline claim rate, enabling accurate risk-adjusted premium pricing. |

---

## 📊 Exploratory Risk Analysis

![EDA Risk Factors](artifacts/eda_risk_factors.png)

### Key Exploratory Observations
1. **Driving Experience Gradient**: Claim rates drop drastically from over **$65\%$** for novice drivers (0–9 years) to below **$15\%$** for veteran drivers (30+ years).
2. **Age Correlation**: Younger cohorts (16–25) experience substantially higher claim frequencies than mature policyholders.
3. **Violation History**: Policyholders with 3+ past accidents experience claim rates exceeding **$70\%$**, demonstrating compounding hazard risk.
4. **Credit Score Divergence**: Non-claim policyholders exhibit significantly higher median credit scores than those filing claims.

---

## 🔍 Single-Feature Predictive Leaderboard

Each feature was evaluated independently using univariate Logistic Regression on an out-of-sample test set (20% stratified holdout).

![Single Feature Accuracy](artifacts/single_feature_accuracy_comparison.png)

```
        best_feature  best_accuracy    roc_auc
  driving_experience         0.7470     0.8355
                 age         0.7280     0.8047
      past_accidents         0.6635     0.6964
 speeding_violations         0.6365     0.6663
                duis         0.6045     0.5633
```

- **`driving_experience`** emerged as the single best predictor, achieving **74.70% test accuracy** and **0.8355 ROC-AUC**.

---

## 🧠 Multivariate Logistic Regression & Statistical Inference

To isolate the independent effect of each variable while controlling for confounding factors, we fitted a regularized multivariate Logistic Regression model.

![Odds Ratios Forest Plot](artifacts/odds_ratios_forest_plot.png)

### Odds Ratios ($OR = e^{\beta}$) Summary Table

| Feature | Coefficient ($\beta$) | Odds Ratio ($OR$) | 95% Confidence Interval | p-value | Risk Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`duis`** | $+0.364$ | **$1.44$** | $[1.27, 1.63]$ | $1.16 \times 10^{-8}$ | **Risk Escalator** (+44% risk per DUI) |
| **`vehicle_type`** (Sports) | $+0.363$ | **$1.44$** | $[1.10, 1.88]$ | $7.65 \times 10^{-3}$ | **Risk Escalator** (+44% risk vs Sedan) |
| **`past_accidents`** | $+0.292$ | **$1.34$** | $[1.27, 1.41]$ | $4.08 \times 10^{-29}$ | **Risk Escalator** (+34% risk per accident) |
| **`speeding_violations`** | $+0.218$ | **$1.24$** | $[1.18, 1.31]$ | $3.70 \times 10^{-16}$ | **Risk Escalator** (+24% risk per violation) |
| **`married`** | $-0.598$ | **$0.55$** | $[0.49, 0.62]$ | $7.16 \times 10^{-20}$ | **Protective** (-45% claim hazard) |
| **`vehicle_ownership`** | $-0.654$ | **$0.52$** | $[0.46, 0.59]$ | $3.91 \times 10^{-23}$ | **Protective** (-48% claim hazard) |
| **`credit_score`** | $-0.994$ | **$0.37$** | $[0.23, 0.60]$ | $6.76 \times 10^{-5}$ | **Protective** (Lower risk for high credit) |
| **`driving_experience`** | $-1.050$ | **$0.35$** | $[0.31, 0.39]$ | $1.93 \times 10^{-98}$ | **Strongest Protective Factor** (-65% hazard per tier) |

---

## 📈 Model Diagnostics & Actuarial Calibration

![ROC and PR Curves](artifacts/roc_pr_curves_diagnostics.png)

- **ROC-AUC: `0.8792`** — Demonstrates exceptional discriminatory power between claimants and non-claimants.
- **Brier Score: `0.1394`** — Confirms sharp probability calibration.

### Portfolio Risk Decile Calibration
![Risk Decile Calibration](artifacts/risk_decile_calibration.png)

```
 Risk Tier   Policies   Actual Claims   Observed Rate   Predicted Rate    Lift
   D1          200           186            93.0%           91.8%        2.15x
   D2          200           167            83.5%           82.4%        1.93x
   D3          200           148            74.0%           73.6%        1.71x
   ...
   D9          200            24            12.0%           12.9%        0.28x
   D10         200             9             4.5%            5.2%        0.10x
```

- **Top Decile Concentration**: Policyholders in **Decile 1** generate an observed claim rate of **$93.0\%$** ($2.15\times$ the portfolio baseline), enabling tier-based pricing and deductible optimization.

---

## 📁 Repository Structure

```
car-insurance-claims-ml/
├── data/
│   └── car_insurance.csv                 # 10,000 policyholder records
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Missing value imputation & train/test split
│   ├── single_feature_evaluation.py      # Univariate logistic regression screening
│   ├── multivariate_model.py             # L2 logistic regression & odds ratio inference
│   └── visualizations.py                 # 5 publication-grade visualization plots
├── artifacts/
│   ├── best_car_insurance_model.joblib   # Serialized production pipeline
│   ├── eda_risk_factors.png              # EDA risk driver distributions
│   ├── single_feature_accuracy_comparison.png # Feature accuracy leaderboard
│   ├── odds_ratios_forest_plot.png       # Forest plot with 95% CIs
│   ├── roc_pr_curves_diagnostics.png     # ROC & Precision-Recall curves
│   └── risk_decile_calibration.png       # Underwriting decile calibration & lift
├── notebooks/
│   └── car_insurance_claim_modeling.ipynb # Interactive exploration and walkthrough
├── tests/
│   ├── __init__.py
│   └── test_car_insurance.py             # Pytest test suite (100% passing)
├── scripts/
│   ├── generate_dataset.py               # Benchmark dataset generator
│   └── build_notebook.py                 # Automated notebook generator
├── run_analysis.py                       # Master CLI runner
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Version control exclusions
└── README.md                             # Technical project documentation
```

---

## ⚡ Quickstart & Reproduction

### 1. Clone & Setup Environment

```bash
git clone https://github.com/DevSanjayShah05/car-insurance-claims-ml.git
cd car-insurance-claims-ml

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Master Analysis Pipeline

```bash
python run_analysis.py
```

### 3. Run Test Suite

```bash
PYTHONPATH=. pytest tests/ -v
```

### 4. Launch Interactive Jupyter Notebook

```bash
jupyter notebook notebooks/car_insurance_claim_modeling.ipynb
```

---

## 🛠️ Technology Stack

- **Machine Learning & Modeling**: Scikit-Learn, Statsmodels, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Testing & Quality Assurance**: Pytest
- **Model Serialization**: Joblib

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
