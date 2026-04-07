"""
=============================================================
Supply Chain Demand Forecasting & Anomaly Detection
Step 3: Anomaly Detection
=============================================================
Models used:
    - Isolation Forest   (unsupervised, tree-based)
    - Z-Score / IQR      (statistical baseline)
    - DBSCAN             (density-based clustering)

Run after: 01_data_preprocessing.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH    = "../data/cleaned_data.csv"
OUTPUT_DIR   = "../data/"
MODEL_DIR    = "../python/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# !! UPDATE to match your column names !!
TARGET_COL   = "order_item_quantity"  # Numeric column to detect anomalies in
DATE_COL     = "order_date"           # Date column (or None)
CONTAMINATION = 0.05             # Expected fraction of anomalies (5%)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading cleaned data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# Use numeric columns for anomaly features
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if TARGET_COL not in numeric_cols:
    TARGET_COL = numeric_cols[0]
    print(f"  ⚠  TARGET_COL not found. Using: '{TARGET_COL}'")

feature_cols = numeric_cols  # Use all numeric cols for multivariate detection
X = df[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Method 1: Isolation Forest ────────────────────────────────────────────────
print("\n=== Isolation Forest ===")
iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
df["iso_anomaly"] = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
df["iso_score"]   = iso.decision_function(X_scaled)

n_anomalies = (df["iso_anomaly"] == -1).sum()
print(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")

joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ── Method 2: Z-Score ─────────────────────────────────────────────────────────
print("\n=== Z-Score Anomaly Detection ===")
z_scores = np.abs((df[TARGET_COL] - df[TARGET_COL].mean()) / df[TARGET_COL].std())
df["zscore_anomaly"] = (z_scores > 3).astype(int)  # 1 = anomaly
print(f"  Anomalies (|Z| > 3): {df['zscore_anomaly'].sum()}")

# ── Method 3: IQR ─────────────────────────────────────────────────────────────
print("\n=== IQR Anomaly Detection ===")
Q1  = df[TARGET_COL].quantile(0.25)
Q3  = df[TARGET_COL].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df["iqr_anomaly"] = ((df[TARGET_COL] < lower_bound) | (df[TARGET_COL] > upper_bound)).astype(int)
print(f"  IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"  Anomalies:  {df['iqr_anomaly'].sum()}")

# ── Method 4: DBSCAN ─────────────────────────────────────────────────────────
print("\n=== DBSCAN Anomaly Detection ===")
# Use top 2 numeric features to keep it interpretable
top_features = feature_cols[:2]
X_db = scaler.fit_transform(df[top_features])
db = DBSCAN(eps=0.5, min_samples=10)
db_labels = db.fit_predict(X_db)
df["dbscan_anomaly"] = (db_labels == -1).astype(int)
print(f"  Anomalies (noise points): {df['dbscan_anomaly'].sum()}")

# ── Consensus Anomaly Flag ─────────────────────────────────────────────────────
df["anomaly_votes"] = (
    (df["iso_anomaly"] == -1).astype(int) +
    df["zscore_anomaly"] +
    df["iqr_anomaly"] +
    df["dbscan_anomaly"]
)
df["is_anomaly"] = (df["anomaly_votes"] >= 2).astype(int)  # flagged by ≥ 2 methods
print(f"\n✅ Consensus anomalies (≥2 methods agree): {df['is_anomaly'].sum()}")

# ── Visualisations ────────────────────────────────────────────────────────────
sns.set_style("whitegrid")

# Plot 1: Anomaly scatter on target column over index
fig, ax = plt.subplots(figsize=(14, 5))
normal   = df[df["is_anomaly"] == 0]
anomalies = df[df["is_anomaly"] == 1]
ax.scatter(normal.index,   normal[TARGET_COL],   c="steelblue", s=10, alpha=0.5, label="Normal")
ax.scatter(anomalies.index, anomalies[TARGET_COL], c="crimson",    s=30, marker="x", label="Anomaly")
ax.set_title(f"Anomaly Detection: {TARGET_COL}", fontsize=13)
ax.set_xlabel("Index")
ax.set_ylabel(TARGET_COL)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_scatter.png"), dpi=150)
print("  Chart saved: ../data/anomaly_scatter.png")

# Plot 2: Isolation Forest anomaly score distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df[df["is_anomaly"] == 0]["iso_score"], bins=50, color="steelblue", alpha=0.7, label="Normal")
ax.hist(df[df["is_anomaly"] == 1]["iso_score"], bins=20, color="crimson",   alpha=0.7, label="Anomaly")
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision Boundary")
ax.set_title("Isolation Forest: Anomaly Score Distribution", fontsize=13)
ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_score_dist.png"), dpi=150)
print("  Chart saved: ../data/anomaly_score_dist.png")

# ── Save Results ──────────────────────────────────────────────────────────────
anomaly_output = df[df["is_anomaly"] == 1].copy()
anomaly_output.to_csv(os.path.join(OUTPUT_DIR, "detected_anomalies.csv"), index=False)
print(f"\n✅ Anomaly records saved to ../data/detected_anomalies.csv")
print(f"   Total anomalies flagged: {len(anomaly_output)}")
