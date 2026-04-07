# Run this script once to install all required R packages
packages <- c(
  "tidyverse",
  "lubridate",
  "ggplot2",
  "scales",
  "readr",
  "corrplot",
  "forecast",
  "tseries",
  "anomalize"
)

install.packages(packages, repos = "https://cloud.r-project.org")
cat("✅ All R packages installed.\n")
