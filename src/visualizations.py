"""
Car Insurance Claim Modeling Visualizations Module.
Generates 5 publication-grade visualization artifacts:
1. eda_risk_factors.png: Claim rates by experience, age, credit score, accidents.
2. single_feature_accuracy_comparison.png: Single-feature model accuracy leaderboard.
3. odds_ratios_forest_plot.png: Forest plot of multivariate odds ratios with 95% CIs.
4. roc_pr_curves_diagnostics.png: Dual ROC and Precision-Recall diagnostic curves.
5. risk_decile_calibration.png: Underwriting risk decile calibration and lift curves.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class InsuranceVisualizer:
    """
    Renders high-resolution publication-quality plots for car insurance claim risk modeling.
    """

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", font="sans-serif")

    def plot_eda_risk_factors(self, df: pd.DataFrame) -> Path:
        """
        Plots 4-panel exploratory overview of primary claim risk drivers.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

        # 1. Claim rate by Driving Experience
        exp_labels = {0: "0-9y", 1: "10-19y", 2: "20-29y", 3: "30y+"}
        df_exp = df.copy()
        df_exp["exp_group"] = df_exp["driving_experience"].map(exp_labels)
        exp_rates = df_exp.groupby("exp_group")["outcome"].mean().loc[["0-9y", "10-19y", "20-29y", "30y+"]]
        
        bars1 = axes[0, 0].bar(exp_rates.index, exp_rates.values * 100, color="#2563eb", edgecolor="#1e3a8a", width=0.55)
        axes[0, 0].set_title("Claim Rate by Driving Experience", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Claim Incidence (%)", fontweight="bold")
        axes[0, 0].set_ylim(0, max(exp_rates.values * 100) * 1.2)
        for bar in bars1:
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                            f"{bar.get_height():.1f}%", ha="center", fontweight="bold", fontsize=10)

        # 2. Claim rate by Age Group
        age_labels = {0: "16-25", 1: "26-39", 2: "40-64", 3: "65+"}
        df_exp["age_group"] = df_exp["age"].map(age_labels)
        age_rates = df_exp.groupby("age_group")["outcome"].mean().loc[["16-25", "26-39", "40-64", "65+"]]
        
        bars2 = axes[0, 1].bar(age_rates.index, age_rates.values * 100, color="#0d9488", edgecolor="#115e59", width=0.55)
        axes[0, 1].set_title("Claim Rate by Age Cohort", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Claim Incidence (%)", fontweight="bold")
        axes[0, 1].set_ylim(0, max(age_rates.values * 100) * 1.2)
        for bar in bars2:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                            f"{bar.get_height():.1f}%", ha="center", fontweight="bold", fontsize=10)

        # 3. Claim rate by Past Accidents
        acc_rates = df.groupby("past_accidents")["outcome"].agg(["mean", "count"]).query("count >= 30")["mean"]
        bars3 = axes[1, 0].bar(acc_rates.index, acc_rates.values * 100, color="#e11d48", edgecolor="#9f1239", width=0.6)
        axes[1, 0].set_title("Claim Rate Escalation by Past Accidents", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Number of Past Accidents", fontweight="bold")
        axes[1, 0].set_ylabel("Claim Incidence (%)", fontweight="bold")
        axes[1, 0].set_ylim(0, 100)
        for bar in bars3:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                            f"{bar.get_height():.1f}%", ha="center", fontweight="bold", fontsize=9)

        # 4. Credit Score Distribution by Claim Outcome
        sns.kdeplot(data=df[df["outcome"] == 0]["credit_score"].dropna(),
                    label="No Claim (0)", color="#10b981", fill=True, alpha=0.3, ax=axes[1, 1], linewidth=2)
        sns.kdeplot(data=df[df["outcome"] == 1]["credit_score"].dropna(),
                    label="Claim Filed (1)", color="#ef4444", fill=True, alpha=0.3, ax=axes[1, 1], linewidth=2)
        axes[1, 1].set_title("Credit Score Density: Claim vs. Non-Claim", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Credit Score (0.0 to 1.0)", fontweight="bold")
        axes[1, 1].set_ylabel("Density", fontweight="bold")
        axes[1, 1].legend(loc="upper left")

        plt.suptitle("Exploratory Risk Factor Analysis: On the Road Insurance Portfolio", fontsize=16, fontweight="heavy", y=0.99)
        plt.tight_layout()
        out_path = self.artifacts_dir / "eda_risk_factors.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_single_feature_accuracies(self, results_df: pd.DataFrame, best_feature: str) -> Path:
        """
        Bar chart ranking single-feature logistic regression accuracy.
        """
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        df_sorted = results_df.sort_values(by="accuracy", ascending=True)

        colors = ["#e11d48" if f == best_feature else "#64748b" for f in df_sorted["feature"]]
        bars = ax.barh(df_sorted["feature"], df_sorted["accuracy"], color=colors, edgecolor="#0f172a", height=0.65)
        ax.set_xlabel("Test Accuracy", fontweight="bold")
        ax.set_xlim(0.50, max(df_sorted["accuracy"]) + 0.08)
        ax.set_title("Single-Feature Logistic Regression Performance Leaderboard", fontsize=14, fontweight="heavy")

        # Baseline accuracy line (majority class)
        for bar in bars:
            val = bar.get_width()
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.4f}",
                    va="center", fontweight="bold", fontsize=9.5)

        # Highlight best feature
        ax.annotate(
            f"BEST PREDICTOR: {best_feature.upper()}\nAccuracy: {results_df[results_df['feature'] == best_feature]['accuracy'].values[0]:.4f}",
            xy=(results_df[results_df['feature'] == best_feature]['accuracy'].values[0], len(df_sorted) - 1),
            xytext=(-150, -40), textcoords="offset points",
            arrowprops=dict(facecolor="#e11d48", shrink=0.08, width=2, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#e11d48", lw=1.5),
            fontweight="bold"
        )

        plt.tight_layout()
        out_path = self.artifacts_dir / "single_feature_accuracy_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_odds_ratios_forest(self, odds_df: pd.DataFrame) -> Path:
        """
        Forest plot displaying multivariate odds ratios with 95% confidence intervals.
        """
        fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
        df_sorted = odds_df.sort_values(by="odds_ratio", ascending=True).reset_index(drop=True)

        y_pos = np.arange(len(df_sorted))
        or_vals = df_sorted["odds_ratio"].values
        ci_lower = df_sorted["or_ci_lower"].values
        ci_upper = df_sorted["or_ci_upper"].values

        # Error bar calculation
        xerr_left = or_vals - ci_lower
        xerr_right = ci_upper - or_vals
        xerr = [xerr_left, xerr_right]

        colors = ["#dc2626" if val > 1.0 else "#16a34a" for val in or_vals]

        # Draw 1.0 reference line (neutral hazard)
        ax.axvline(1.0, color="#475569", linestyle="--", linewidth=1.5, label="Neutral Risk (OR = 1.0)")

        for i in range(len(df_sorted)):
            ax.errorbar(
                or_vals[i], y_pos[i], xerr=[[xerr_left[i]], [xerr_right[i]]],
                fmt='o', color=colors[i], ecolor=colors[i], elinewidth=2, capsize=4, markersize=7
            )
            ax.text(ci_upper[i] + 0.03, y_pos[i], f"OR: {or_vals[i]:.2f} [{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]",
                    va='center', fontsize=9, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["feature"], fontweight="bold")
        ax.set_xlabel("Odds Ratio (95% Confidence Interval) [OR > 1: Risk Factor | OR < 1: Protective]", fontweight="bold")
        ax.set_title("Multivariate Logistic Regression: Impact of Factors on Insurance Claims (Odds Ratios)", fontsize=13, fontweight="heavy")
        ax.legend(loc="lower right")

        plt.tight_layout()
        out_path = self.artifacts_dir / "odds_ratios_forest_plot.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_roc_pr_curves(self, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Path:
        """
        Renders dual ROC and Precision-Recall diagnostic curves.
        """
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc_val = average_precision_score(y_test, y_prob)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        # ROC Curve
        ax1.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc_val:.3f})")
        ax1.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
        ax1.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
        ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
        ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        ax1.legend(loc="lower right", framealpha=0.9)

        # Precision-Recall Curve
        baseline_rate = y_test.mean()
        ax2.plot(rec, prec, color="#059669", lw=2.5, label=f"PR Curve (AP = {pr_auc_val:.3f})")
        ax2.axhline(baseline_rate, color="#94a3b8", linestyle="--", lw=1.5, label=f"Baseline Prevalance ({baseline_rate:.1%})")
        ax2.fill_between(rec, prec, alpha=0.15, color="#059669")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("Recall", fontweight="bold")
        ax2.set_ylabel("Precision", fontweight="bold")
        ax2.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
        ax2.legend(loc="lower left", framealpha=0.9)

        plt.suptitle("Model Discriminative Power & Diagnostic Curves", fontsize=15, fontweight="heavy", y=0.99)
        plt.tight_layout()
        out_path = self.artifacts_dir / "roc_pr_curves_diagnostics.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_risk_decile_calibration(self, decile_df: pd.DataFrame) -> Path:
        """
        Renders risk decile calibration and lift curves for underwriting deployment.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

        # Decile Calibration: Predicted vs Observed
        width = 0.35
        x = np.arange(len(decile_df))
        ax1.bar(x - width/2, decile_df["observed_claim_rate"] * 100, width, label="Observed Claims (%)", color="#1d4ed8")
        ax1.bar(x + width/2, decile_df["predicted_claim_rate"] * 100, width, label="Model Predicted Risk (%)", color="#93c5fd")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"D{t}" for t in decile_df["risk_tier"]], fontweight="bold")
        ax1.set_xlabel("Risk Decile (D1 = Highest Risk, D10 = Lowest Risk)", fontweight="bold")
        ax1.set_ylabel("Claim Rate (%)", fontweight="bold")
        ax1.set_title("Underwriting Calibration Across Risk Deciles", fontsize=12, fontweight="bold")
        ax1.legend()

        # Lift Curve
        ax2.plot(range(1, len(decile_df) + 1), decile_df["lift"], marker='o', lw=2.5, color="#ea580c")
        ax2.axhline(1.0, color="#64748b", linestyle="--", label="Neutral Baseline (Lift = 1.0)")
        ax2.set_xticks(range(1, len(decile_df) + 1))
        ax2.set_xticklabels([f"D{t}" for t in decile_df["risk_tier"]], fontweight="bold")
        ax2.set_xlabel("Risk Decile Tier", fontweight="bold")
        ax2.set_ylabel("Lift Multiplier (x Baseline)", fontweight="bold")
        ax2.set_title("Portfolio Lift Curve: Claim Concentration by Decile", fontsize=12, fontweight="bold")
        ax2.legend()
        for _, row in decile_df.iterrows():
            ax2.text(row["risk_tier"], row["lift"] + 0.08, f"{row['lift']:.2f}x", ha="center", fontweight="bold", fontsize=9)

        plt.suptitle("Actuarial Risk Segmentation & Calibration Analytics", fontsize=15, fontweight="heavy", y=1.02)
        plt.tight_layout()
        out_path = self.artifacts_dir / "risk_decile_calibration.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def generate_all(
        self,
        df_raw: pd.DataFrame,
        results_df: pd.DataFrame,
        best_feature: str,
        odds_df: pd.DataFrame,
        pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        decile_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Executes all visual rendering tasks.
        """
        p1 = self.plot_eda_risk_factors(df_raw)
        p2 = self.plot_single_feature_accuracies(results_df, best_feature)
        p3 = self.plot_odds_ratios_forest(odds_df)
        p4 = self.plot_roc_pr_curves(pipeline, X_test, y_test)
        p5 = self.plot_risk_decile_calibration(decile_df)

        return {
            "eda_risk_factors": str(p1),
            "single_feature_accuracies": str(p2),
            "odds_ratios_forest": str(p3),
            "roc_pr_curves": str(p4),
            "risk_decile_calibration": str(p5)
        }
