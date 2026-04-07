# 🚚 Supply Chain Demand Forecasting & Anomaly Detection

> A full-stack data science project showcasing machine learning, statistical analysis, and business intelligence dashboards for supply chain optimisation.

---

## 📌 Project Overview

This project applies machine learning and statistical techniques to real-world logistics data to:

- **Forecast demand** using Prophet (time-series) and ensemble models (Random Forest, XGBoost)
- **Detect anomalies** in shipment data using Isolation Forest, Z-Score, IQR, and DBSCAN
- **Visualise insights** through interactive Power BI dashboards
- **Validate findings** with rigorous statistical analysis in R

---

## 🗂️ Repository Structure

```
supply-chain-ml/
├── data/                        # Raw and processed datasets (gitignored for large files)
│   ├── cleaned_data.csv
│   ├── detected_anomalies.csv
│   └── model_results.csv
│
├── python/                      # Python ML pipeline
│   ├── 01_data_preprocessing.py
│   ├── 02_demand_forecasting.py
│   ├── 03_anomaly_detection.py
│   ├── 04_model_evaluation.py
│   ├── models/                  # Saved trained models (.pkl)
│   └── requirements.txt
│
├── R/                           # R statistical analysis
│   ├── analysis.R
│   └── requirements.R
│
├── dashboards/                  # BI dashboard files and guides
│   ├── powerbi_guide.md
│   ├── supply_chain_dashboard.pbix
│   └── supply_chain_dashboard.pdf
│
├── notebooks/                   # Jupyter notebooks for exploration
│
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | ML pipeline, feature engineering, model training |
| **R** | Statistical analysis, ARIMA forecasting, correlation analysis |
| **scikit-learn** | Random Forest, Isolation Forest, DBSCAN |
| **XGBoost** | Gradient boosted regression |
| **Prophet** | Time-series demand forecasting |
| **Power BI** | Interactive KPI & anomaly reporting dashboards |
| **VS Code** | Development environment (with Claude AI co-pilot) |
| **GitHub** | Version control & project showcase |

---

## 📊 Dataset

**Source:** [Logistics Supply Chain Real World Data — Kaggle](https://www.kaggle.com/datasets/pushpitkamboj/logistics-data-containing-real-world-data)

Download the dataset and place the CSV file in the `data/` folder, renaming it to `logistics_data.csv`.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/supply-chain-ml.git
cd supply-chain-ml
```

### 2. Install Python Dependencies
```bash
pip install -r python/requirements.txt
```

### 3. Install R Dependencies
```r
source("R/requirements.R")
```

### 4. Download the Dataset
- Go to [Kaggle dataset link](https://www.kaggle.com/datasets/pushpitkamboj/logistics-data-containing-real-world-data)
- Download and place the CSV in `data/logistics_data.csv`

### 5. Run the Pipeline (in order)
```bash
# Step 1: Clean and preprocess data
python python/01_data_preprocessing.py

# Step 2: Train forecasting models
python python/02_demand_forecasting.py

# Step 3: Detect anomalies
python python/03_anomaly_detection.py

# Step 4: Generate evaluation report
python python/04_model_evaluation.py
```

### 6. Run R Analysis

```r
source("R/analysis.R")
```

R provides statistical validation alongside the Python ML pipeline:

| Output | Purpose |
|---|---|
| `R_distribution.png` | Distribution of order quantities |
| `R_time_series.png` | Demand trend over time with LOESS smoothing |
| `R_arima_forecast.png` | Classical ARIMA 30-period forecast |
| `R_category_breakdown.png` | Top 15 categories by demand |
| `R_correlation.png` | Feature correlation matrix |
| `R_anomaly_distribution.png` | Z-score anomaly distribution |

> Using both Python and R demonstrates the full data science stack: Python for ML modelling, R for statistical rigour, Power BI for business reporting.

### 7. Build Dashboards
- Follow `dashboards/powerbi_guide.md` for Power BI
- Or open `dashboards/supply_chain_dashboard.pbix` directly

---

## 📈 Key Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 0.03 | 0.14 | 0.99 |
| XGBoost | 0.04 | 0.14 | 0.99 |
| Prophet | 11.86 | 13.67 | 0.10 |

**Anomalies detected:** 67 records flagged across 4 detection methods (anomaly rate: 43%).

> Random Forest and XGBoost significantly outperform Prophet, achieving near-perfect R² of 0.99. Prophet underperforms on this dataset as demand patterns are feature-driven rather than purely time-series.

---

## 📊 Power BI Dashboard

**Download:** [supply_chain_dashboard.pbix](dashboards/supply_chain_dashboard.pbix) | [supply_chain_dashboard.pdf](dashboards/supply_chain_dashboard.pdf)

### Page 1 — Executive Summary

![Executive Summary](dashboards/screenshots/page1.png)

### Page 2 — Anomaly Report

![Anomaly Report](dashboards/screenshots/page2.png)

### Page 3 — ML Model Comparison

![Model Comparison](dashboards/screenshots/page3.png)

---

## 🔧 Configuration

Before running, update the column name constants at the top of each script to match your dataset:

```python
TARGET_COL   = "quantity"        # Column to forecast/detect anomalies in
DATE_COL     = "order_date"      # Primary date column
CATEGORY_COL = "product_category"# Grouping column
```

---

## 📁 Outputs

After running the full pipeline, the `data/` folder will contain:

- `cleaned_data.csv` — preprocessed dataset
- `detected_anomalies.csv` — records flagged as anomalies
- `model_results.csv` — model performance comparison
- `prophet_forecast.png` — Prophet forecast chart
- `feature_importance.png` — Random Forest feature importances
- `anomaly_scatter.png` — anomaly visualisation
- `model_comparison.png` — model metrics bar chart
- `R_*.png` — R-generated statistical charts

---

## 👤 Author

**Brian Santoso**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is licensed under the MIT License.
