"""
=============================================================
Supply Chain Demand Forecasting & Anomaly Detection
Step 1: Data Preprocessing
=============================================================
Dataset: Kaggle - Logistics Supply Chain Real World Data
https://www.kaggle.com/datasets/pushpitkamboj/logistics-data-containing-real-world-data

Instructions:
    1. Download the dataset from Kaggle
    2. Place CSV file(s) in the ../data/ folder
    3. Update FILE_PATH below to match your filename
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
FILE_PATH = "../data/logistics_data.csv"   # <-- update if filename differs
OUTPUT_PATH = "../data/cleaned_data.csv"

# ── Load Data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(FILE_PATH)
print(f"  Raw shape: {df.shape}")
print(f"  Columns: {list(df.columns)}\n")

# ── Basic Info ───────────────────────────────────────────────────────────────
print("=== Data Overview ===")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# ── Standardise Column Names ─────────────────────────────────────────────────
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)
print("\nCleaned column names:", list(df.columns))

# ── Detect & Parse Date Columns ──────────────────────────────────────────────
date_candidates = [c for c in df.columns if any(kw in c for kw in ["date", "time", "day", "month", "year"])]
print(f"\nDetected date candidates: {date_candidates}")

for col in date_candidates:
    try:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        print(f"  Parsed '{col}' as datetime.")
    except Exception:
        pass

# ── Drop High-Null Columns ────────────────────────────────────────────────────
threshold = 0.5
null_frac = df.isnull().mean()
drop_cols = null_frac[null_frac > threshold].index.tolist()
if drop_cols:
    print(f"\nDropping high-null columns (>{int(threshold*100)}% missing): {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)

# ── Fill Remaining Nulls ──────────────────────────────────────────────────────
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)

# ── Remove Duplicate Rows ─────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {before - len(df)} rows")

# ── Feature Engineering ───────────────────────────────────────────────────────
# Try to build time features if a date column exists
date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
if date_cols:
    primary_date = date_cols[0]
    df["year"]        = df[primary_date].dt.year
    df["month"]       = df[primary_date].dt.month
    df["day_of_week"] = df[primary_date].dt.dayofweek
    df["week_of_year"]= df[primary_date].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    print(f"\nTime features engineered from '{primary_date}'.")

# ── Save Cleaned Data ─────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned data saved to: {OUTPUT_PATH}")
print(f"   Final shape: {df.shape}")
