# =============================================================
# Supply Chain Demand Forecasting & Anomaly Detection
# R Statistical Analysis & Visualisation
# =============================================================
# Packages required:
#   install.packages(c("tidyverse", "lubridate", "forecast",
#                      "ggplot2", "corrplot", "scales",
#                      "tseries", "anomalize", "readr"))
# =============================================================

library(tidyverse)
library(lubridate)
library(ggplot2)
library(scales)
library(readr)
library(corrplot)

# Optional: time-series packages
if (requireNamespace("forecast", quietly = TRUE)) library(forecast)
if (requireNamespace("tseries",  quietly = TRUE)) library(tseries)
if (requireNamespace("anomalize",quietly = TRUE)) library(anomalize)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  <- "../data/cleaned_data.csv"
OUTPUT_DIR <- "../data/"

# !! UPDATE these to match your actual column names !!
TARGET_COL   <- "order_item_quantity"
DATE_COL     <- "order_date"
CATEGORY_COL <- "category_name"      # or NULL if not present

# ── Load Data ─────────────────────────────────────────────────────────────────
cat("Loading data...\n")
df <- read_csv(DATA_PATH, show_col_types = FALSE)
cat(sprintf("  Shape: %d rows × %d cols\n", nrow(df), ncol(df)))
cat(sprintf("  Columns: %s\n\n", paste(names(df), collapse = ", ")))

# Parse date if it exists
if (DATE_COL %in% names(df)) {
  df[[DATE_COL]] <- as.Date(df[[DATE_COL]])
}

# ── 1. Summary Statistics ─────────────────────────────────────────────────────
cat("=== Summary Statistics ===\n")
print(summary(df))

# ── 2. Distribution of Target Variable ───────────────────────────────────────
if (TARGET_COL %in% names(df)) {
  cat("\n=== Distribution Plot ===\n")

  p1 <- ggplot(df, aes_string(x = TARGET_COL)) +
    geom_histogram(aes(y = after_stat(density)), bins = 50,
                   fill = "steelblue", colour = "white", alpha = 0.8) +
    geom_density(colour = "darkred", linewidth = 1) +
    labs(title = paste("Distribution of", TARGET_COL),
         subtitle = sprintf("Mean: %.2f | Median: %.2f | SD: %.2f",
                            mean(df[[TARGET_COL]], na.rm = TRUE),
                            median(df[[TARGET_COL]], na.rm = TRUE),
                            sd(df[[TARGET_COL]], na.rm = TRUE)),
         x = TARGET_COL, y = "Density") +
    theme_minimal(base_size = 13)

  ggsave(file.path(OUTPUT_DIR, "R_distribution.png"), p1, width = 8, height = 5, dpi = 150)
  cat("  Saved: R_distribution.png\n")
}

# ── 3. Time-Series Trend ──────────────────────────────────────────────────────
if (DATE_COL %in% names(df) && TARGET_COL %in% names(df)) {
  cat("\n=== Time-Series Trend ===\n")

  ts_df <- df %>%
    group_by(date = .data[[DATE_COL]]) %>%
    summarise(total = sum(.data[[TARGET_COL]], na.rm = TRUE), .groups = "drop") %>%
    arrange(date)

  p2 <- ggplot(ts_df, aes(x = date, y = total)) +
    geom_line(colour = "steelblue", linewidth = 0.8) +
    geom_smooth(method = "loess", span = 0.3, colour = "darkred",
                fill = "lightcoral", alpha = 0.2, se = TRUE) +
    scale_y_continuous(labels = comma) +
    labs(title = paste("Daily", TARGET_COL, "Over Time"),
         subtitle = "Blue = actual, Red = LOESS trend",
         x = "Date", y = paste("Total", TARGET_COL)) +
    theme_minimal(base_size = 13)

  ggsave(file.path(OUTPUT_DIR, "R_time_series.png"), p2, width = 12, height = 5, dpi = 150)
  cat("  Saved: R_time_series.png\n")

  # ARIMA stationarity test
  if (requireNamespace("tseries", quietly = TRUE)) {
    cat("\n=== ADF Stationarity Test ===\n")
    adf_result <- adf.test(ts_df$total, alternative = "stationary")
    cat(sprintf("  ADF p-value: %.4f\n", adf_result$p.value))
    if (adf_result$p.value < 0.05) {
      cat("  ✅ Series is likely stationary (p < 0.05)\n")
    } else {
      cat("  ⚠  Series may be non-stationary — consider differencing\n")
    }
  }

  # Auto ARIMA forecast
  if (requireNamespace("forecast", quietly = TRUE)) {
    cat("\n=== Auto ARIMA Forecast (next 30 periods) ===\n")
    ts_obj <- ts(ts_df$total, frequency = 7)  # weekly seasonality
    arima_model <- auto.arima(ts_obj, stepwise = TRUE, approximation = TRUE)
    cat("  Best model: "); print(arima_model)

    fc <- forecast(arima_model, h = 30)
    png(file.path(OUTPUT_DIR, "R_arima_forecast.png"), width = 1200, height = 500, res = 120)
    plot(fc, main = "ARIMA Forecast (30 periods ahead)",
         xlab = "Time", ylab = TARGET_COL,
         col = "steelblue", fcol = "darkred")
    dev.off()
    cat("  Saved: R_arima_forecast.png\n")
  }
}

# ── 4. Category Breakdown ─────────────────────────────────────────────────────
if (!is.null(CATEGORY_COL) && CATEGORY_COL %in% names(df)) {
  cat("\n=== Category Breakdown ===\n")

  cat_df <- df %>%
    group_by(category = .data[[CATEGORY_COL]]) %>%
    summarise(total = sum(.data[[TARGET_COL]], na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(total)) %>%
    slice_head(n = 15)

  p3 <- ggplot(cat_df, aes(x = reorder(category, total), y = total)) +
    geom_col(fill = "steelblue", colour = "white", alpha = 0.85) +
    coord_flip() +
    scale_y_continuous(labels = comma) +
    labs(title = paste(TARGET_COL, "by", CATEGORY_COL),
         subtitle = "Top 15 categories",
         x = CATEGORY_COL, y = paste("Total", TARGET_COL)) +
    theme_minimal(base_size = 13)

  ggsave(file.path(OUTPUT_DIR, "R_category_breakdown.png"), p3, width = 9, height = 6, dpi = 150)
  cat("  Saved: R_category_breakdown.png\n")
}

# ── 5. Correlation Matrix ─────────────────────────────────────────────────────
cat("\n=== Correlation Matrix ===\n")
num_df <- df %>% select(where(is.numeric)) %>% select(1:min(12, ncol(.)))
if (ncol(num_df) >= 2) {
  corr_mat <- cor(num_df, use = "pairwise.complete.obs")
  png(file.path(OUTPUT_DIR, "R_correlation.png"), width = 900, height = 800, res = 120)
  corrplot(corr_mat, method = "color", type = "lower",
           addCoef.col = "black", number.cex = 0.7,
           tl.cex = 0.8, tl.col = "black", cl.cex = 0.8,
           title = "Feature Correlation Matrix", mar = c(0, 0, 2, 0))
  dev.off()
  cat("  Saved: R_correlation.png\n")
}

# ── 6. Statistical Anomaly Detection (Z-Score) ────────────────────────────────
if (TARGET_COL %in% names(df)) {
  cat("\n=== Statistical Anomaly Detection ===\n")
  df <- df %>%
    mutate(
      z_score     = abs((.data[[TARGET_COL]] - mean(.data[[TARGET_COL]], na.rm=TRUE)) /
                        sd(.data[[TARGET_COL]], na.rm=TRUE)),
      is_anomaly_r = if_else(z_score > 3, "Anomaly", "Normal")
    )

  n_anomalies <- sum(df$is_anomaly_r == "Anomaly", na.rm = TRUE)
  cat(sprintf("  Anomalies detected (|Z| > 3): %d (%.2f%%)\n",
              n_anomalies, n_anomalies / nrow(df) * 100))

  p4 <- ggplot(df, aes_string(x = TARGET_COL, fill = "is_anomaly_r")) +
    geom_histogram(bins = 60, colour = "white", alpha = 0.8) +
    scale_fill_manual(values = c("Normal" = "steelblue", "Anomaly" = "crimson")) +
    labs(title = paste("Anomaly Detection:", TARGET_COL),
         subtitle = sprintf("%d anomalies flagged via Z-score (|Z| > 3)", n_anomalies),
         x = TARGET_COL, y = "Count", fill = "") +
    theme_minimal(base_size = 13)

  ggsave(file.path(OUTPUT_DIR, "R_anomaly_distribution.png"), p4, width = 9, height = 5, dpi = 150)
  cat("  Saved: R_anomaly_distribution.png\n")
}

cat("\n✅ R analysis complete. All charts saved to", OUTPUT_DIR, "\n")
