# Power BI Dashboard Guide
## Supply Chain Demand Forecasting & Anomaly Detection

---

## Data Sources to Connect
In Power BI Desktop: **Home → Get Data → Text/CSV**

| File | Use |
|---|---|
| `cleaned_data.csv` | Main fact table |
| `detected_anomalies.csv` | Anomaly dimension table |
| `model_results.csv` | Model metrics table |

---

## Data Model (Relationships)
In the **Model view**, create these relationships:

- `cleaned_data[order_item_id]` → `detected_anomalies[order_item_id]` (1:1, or left join)

---

## DAX Measures to Create

Open the **Data view**, click `cleaned_data`, then click **New measure** for each one below (enter them one at a time):

**Measure 1:**

```dax
Total Demand = SUM(cleaned_data[order_item_quantity])
```

**Measure 2:**

```dax
Avg Daily Demand = AVERAGEX(
    VALUES(cleaned_data[order_date]),
    CALCULATE(SUM(cleaned_data[order_item_quantity]))
)
```

**Measure 3:**

```dax
Anomaly Count = COUNTROWS(detected_anomalies)
```

**Measure 4:**

```dax
Anomaly Rate = DIVIDE([Anomaly Count], COUNTROWS(cleaned_data), 0)
```

**Measure 5:**

```dax
MoM Growth =
VAR CurrentMonth = [Total Demand]
VAR PrevMonth = CALCULATE([Total Demand], DATEADD(cleaned_data[order_date], -1, MONTH))
RETURN DIVIDE(CurrentMonth - PrevMonth, PrevMonth, 0)
```

---

## Page 1: Executive Summary

**Visuals to add:**

1. **Card** → `[Total Demand]` — headline KPI
2. **Card** → `[Anomaly Count]` — with red conditional formatting if > threshold
3. **Card** → `[Anomaly Rate]` — formatted as percentage
4. **Line Chart** → X: `order_date` (Month), Y: `[Total Demand]`, Legend: `category_name` — limit to top 5 categories via Top N filter
5. **Clustered Bar** → Y: `category_name`, X: `[Total Demand]` — Top N filter: top 10 by Total Demand
6. **Table** → Columns: `year`, `month`, `[Total Demand]` — clicking a row filters all visuals by that period
7. **Clustered Bar** → Y: `category_name`, X: `[Total Demand]` — clicking a bar filters all visuals by that category

> **Note:** Steps 6 and 7 replace slicers. Power BI cross-filtering means clicking any row or bar automatically filters the whole page — no slicers needed.

---

## Page 2: Anomaly Report

**Visuals to add:**

1. **Scatter Chart** → X: `order_date`, Y: `order_item_quantity`, Color: `is_anomaly`
   - Set anomaly color to Red (#E84040), normal to Steel Blue (#2E75B6)
2. **Table** → Columns: `order_date`, `category_name`, `order_item_quantity`, `iso_score`, `anomaly_votes`
   - Add **Conditional Formatting** on `quantity` (data bars)
   - Sort by `anomaly_votes` descending
3. **Donut Chart** → `is_anomaly` legend, `[Count]` measure — shows normal vs anomaly split
4. **KPI Visual** → Current anomaly count vs. target threshold

---

## Page 3: ML Model Comparison

**Visuals to add:**

1. **Clustered Column Chart** → X: `model`, Y: `MAE` — with data labels
2. **Clustered Column Chart** → X: `model`, Y: `RMSE`
3. **Clustered Column Chart** → X: `model`, Y: `R2`
4. **Matrix** → Rows: `model`, Values: `MAE`, `RMSE`, `R2` — add conditional formatting

---

## Publishing Tips
- Save as `.pbix` and add to the `dashboards/` folder
- **File → Export → Export to PDF** for a static version to include in the README
- Consider publishing to Power BI Service and embedding the link in the README
- Use **Themes** (View → Themes) to apply a consistent color scheme
