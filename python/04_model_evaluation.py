"""
=============================================================
Supply Chain Demand Forecasting & Anomaly Detection
Step 4: Model Evaluation & Visualisation Report
=============================================================
Generates a consolidated visual evaluation report of all models.

Run after: 02_demand_forecasting.py and 03_anomaly_detection.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
sns.set_palette("Set2")

DATA_PATH    = "../data/cleaned_data.csv"
RESULTS_PATH = "../data/model_results.csv"
ANOMALY_PATH = "../data/detected_anomalies.csv"
OUTPUT_DIR   = "../data/"
MODEL_DIR    = "../python/models/"

TARGET_COL   = "order_item_quantity"  # !! UPDATE !!

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data and results...")
df       = pd.read_csv(DATA_PATH)
anomalies = pd.read_csv(ANOMALY_PATH) if os.path.exists(ANOMALY_PATH) else pd.DataFrame()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if TARGET_COL not in numeric_cols:
    TARGET_COL = numeric_cols[0]

# ── Figure 1: Distribution of Target Variable ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Target Variable Analysis: {TARGET_COL}", fontsize=14, fontweight="bold")

axes[0].hist(df[TARGET_COL], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].set_title("Distribution")
axes[0].set_xlabel(TARGET_COL)
axes[0].set_ylabel("Count")

axes[1].boxplot(df[TARGET_COL].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.7))
axes[1].set_title("Boxplot")
axes[1].set_ylabel(TARGET_COL)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=150)
print("  Saved: target_distribution.png")

# ── Figure 2: Correlation Heatmap ─────────────────────────────────────────────
corr_cols = [c for c in numeric_cols if c not in
             ["iso_anomaly", "iso_score", "zscore_anomaly", "iqr_anomaly",
              "dbscan_anomaly", "anomaly_votes", "is_anomaly"]][:12]

if len(corr_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    print("  Saved: correlation_heatmap.png")

# ── Figure 3: Model Comparison Bar Chart ─────────────────────────────────────
if os.path.exists(RESULTS_PATH):
    results_df = pd.read_csv(RESULTS_PATH)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = ax.bar(results_df["model"], results_df[metric],
                      color=sns.color_palette("Set2", len(results_df)), edgecolor="white")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xticklabels(results_df["model"], rotation=20, ha="right")
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    print("  Saved: model_comparison.png")

# ── Figure 4: Anomaly Summary ─────────────────────────────────────────────────
if not anomalies.empty and TARGET_COL in anomalies.columns:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(df.index, df[TARGET_COL], c="steelblue", s=8, alpha=0.4, label="Normal")
    ax.scatter(anomalies.index, anomalies[TARGET_COL], c="crimson", s=40,
               marker="x", linewidths=1.5, label=f"Anomalies (n={len(anomalies)})")
    ax.set_title(f"Detected Anomalies in '{TARGET_COL}'", fontsize=13)
    ax.set_xlabel("Record Index")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_summary.png"), dpi=150)
    print("  Saved: anomaly_summary.png")

# ── Print Summary Stats ───────────────────────────────────────────────────────
print("\n=== Summary Statistics ===")
print(df[corr_cols].describe().round(2).to_string())
print(f"\n✅ Evaluation complete. All charts saved to {OUTPUT_DIR}")
