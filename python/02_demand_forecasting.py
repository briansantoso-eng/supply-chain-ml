"""
=============================================================
Supply Chain Demand Forecasting & Anomaly Detection
Step 2: Demand Forecasting with Prophet + scikit-learn
=============================================================
Models used:
    - Facebook Prophet  (time-series forecasting)
    - Random Forest     (feature-based regression baseline)
    - XGBoost           (gradient boosted regression)

Run after: 01_data_preprocessing.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Optional: XGBoost (install with: pip install xgboost)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model. (pip install xgboost)")

# Optional: Prophet (install with: pip install prophet)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Skipping Prophet model. (pip install prophet)")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH    = "../data/cleaned_data.csv"
OUTPUT_DIR   = "../data/"
MODEL_DIR    = "../python/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# !! UPDATE THESE to match your actual column names after preprocessing !!
TARGET_COL   = "order_item_quantity"  # The column you want to forecast (e.g. units shipped)
DATE_COL     = "order_date"           # Primary date column
CATEGORY_COL = "category_name"        # Optional grouping column (or None)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading cleaned data...")
df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL] if DATE_COL in pd.read_csv(DATA_PATH, nrows=1).columns else [])
print(f"  Shape: {df.shape}")

if TARGET_COL not in df.columns:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    TARGET_COL = numeric_cols[0]
    print(f"  ⚠  TARGET_COL not found. Falling back to first numeric column: '{TARGET_COL}'")

# ── Prepare Feature Matrix ────────────────────────────────────────────────────
print("\nPreparing features...")
feature_cols = []

# Time features
time_feats = ["year", "month", "day_of_week", "week_of_year", "is_weekend"]
for f in time_feats:
    if f in df.columns:
        feature_cols.append(f)

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    feature_cols.append(f"{col}_enc")

# Add remaining numeric features (excluding target)
num_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c != TARGET_COL and c not in feature_cols]
feature_cols.extend(num_cols)

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

print(f"  Features used: {len(feature_cols)}")
print(f"  Target: '{TARGET_COL}'")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n  Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── Helper: Evaluate Model ────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n📊 {name}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []

# ── Model 1: Random Forest ────────────────────────────────────────────────────
print("\n=== Training Random Forest ===")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, rf_preds))
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
print("  Model saved.")

# ── Model 2: XGBoost ──────────────────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("\n=== Training XGBoost ===")
    xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                   random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    results.append(evaluate("XGBoost", y_test, xgb_preds))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost.pkl"))
    print("  Model saved.")

# ── Model 3: Prophet (Time-Series) ────────────────────────────────────────────
if PROPHET_AVAILABLE and DATE_COL in df.columns:
    print("\n=== Training Prophet (Time-Series Forecast) ===")

    # Aggregate by date
    prophet_df = df[[DATE_COL, TARGET_COL]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], utc=True).dt.tz_localize(None)
    prophet_df = prophet_df.groupby("ds", as_index=False)["y"].sum()
    prophet_df = prophet_df.dropna().sort_values("ds")

    # Split: last 20% of time as test
    split_idx = int(len(prophet_df) * 0.8)
    train_prophet = prophet_df.iloc[:split_idx]
    test_prophet  = prophet_df.iloc[split_idx:]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(train_prophet)

    future   = m.make_future_dataframe(periods=len(test_prophet))
    forecast = m.predict(future)

    # Evaluate on test period — normalize to date only to avoid DST timestamp mismatches
    forecast["ds"] = forecast["ds"].dt.normalize()
    test_prophet["ds"] = test_prophet["ds"].dt.normalize()
    merged = test_prophet.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    if not merged.empty:
        results.append(evaluate("Prophet", merged["y"].values, merged["yhat"].values))

    # Save forecast chart
    fig = m.plot(forecast)
    fig.suptitle("Prophet Demand Forecast", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(OUTPUT_DIR, "prophet_forecast.png"), dpi=150, bbox_inches="tight")
    print("  Forecast chart saved to ../data/prophet_forecast.png")

# ── Feature Importance (Random Forest) ───────────────────────────────────────
print("\n=== Feature Importance (Random Forest) ===")
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
top_n = min(15, len(importances))
plt.figure(figsize=(10, 5))
importances.head(top_n).plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Top Feature Importances (Random Forest)", fontsize=13)
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
print(f"  Chart saved to ../data/feature_importance.png")

# ── Save Results Summary ──────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)
print(f"\n✅ Model comparison saved to ../data/model_results.csv")
print("\n", results_df.to_string(index=False))
