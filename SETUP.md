# Quick Setup Guide

## Step 1 — Open in VS Code

```
File → Open Folder → select supply-chain-ml/
```

When prompted, click **"Install Recommended Extensions"** (from `.vscode/extensions.json`).

## Step 2 — Create Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r python/requirements.txt
```

## Step 3 — Install R Packages

Open R or RStudio and run:

```r
source("R/requirements.R")
```

## Step 4 — Download Dataset from Kaggle

1. Go to: <https://www.kaggle.com/datasets/pushpitkamboj/logistics-data-containing-real-world-data>
2. Download the CSV
3. Place it in `data/` and rename to `logistics_data.csv`

## Step 5 — Run the Pipeline

Use VS Code's **Run & Debug** panel (Ctrl+Shift+D) to run scripts in order:

1. `01 - Data Preprocessing`
2. `02 - Demand Forecasting`
3. `03 - Anomaly Detection`
4. `04 - Model Evaluation`

Or run from terminal:

```bash
cd python
python 01_data_preprocessing.py
python 02_demand_forecasting.py
python 03_anomaly_detection.py
python 04_model_evaluation.py
```

## Step 6 — Run R Analysis

```bash
Rscript R/analysis.R
```

## Step 7 — Build Dashboards

- Open `dashboards/supply_chain_dashboard.pbix` directly in Power BI Desktop, or
- Follow `dashboards/powerbi_guide.md` to build from scratch

## Step 8 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: supply chain ML project"
git remote add origin https://github.com/YOUR_USERNAME/supply-chain-ml.git
git push -u origin main
```

## ⚠️ Important — Update Column Names

Before running, open each Python script and update these constants to match your dataset:

```python
TARGET_COL   = "quantity"         # column to forecast
DATE_COL     = "order_date"       # date column
CATEGORY_COL = "product_category" # grouping column
```

The same applies in `R/analysis.R`.
