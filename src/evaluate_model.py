
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

df = pd.read_csv(
    REPORTS / "forecast_predictions.csv",
    parse_dates=["Date"],
    dtype={"StockCode": str}
)

with open(REPORTS / "model_metrics.json") as f:
    metrics = json.load(f)

print("\n=== SMARTRETAIL AI: PHASE 4 ===")

print("\nRandom Forest:")
print(metrics["random_forest"])

print("\nSeasonal Naive:")
print(metrics["seasonal_naive"])

rf_mae = metrics["random_forest"]["MAE"]
base_mae = metrics["seasonal_naive"]["MAE"]

if base_mae > 0:
    improvement = 100 * (base_mae - rf_mae) / base_mae
    print(f"\nMAE improvement: {improvement:.2f}%")

# 1. Aggregate forecasts across the selected products
daily = (
    df.groupby("Date", as_index=False)
    .agg(
        Actual=("UnitsSold", "sum"),
        Predicted=("PredictedUnits", "sum"),
        Baseline=("BaselineUnits", "sum")
    )
)

chart_data = daily.melt(
    id_vars="Date",
    value_vars=["Actual", "Predicted", "Baseline"],
    var_name="Series",
    value_name="Units"
)

fig = px.line(
    chart_data,
    x="Date",
    y="Units",
    color="Series",
    title="Actual vs Predicted Daily Demand"
)
fig.write_html(REPORTS / "forecast_comparison.html")

# 2. Prediction error analysis
df["AbsoluteError"] = (
    df["UnitsSold"] - df["PredictedUnits"]
).abs()

fig = px.histogram(
    df,
    x="AbsoluteError",
    nbins=60,
    title="Random Forest Absolute Error Distribution"
)
fig.write_html(REPORTS / "error_distribution.html")

# 3. Performance by product
product_errors = (
    df.groupby("StockCode")
    .agg(
        MAE=("AbsoluteError", "mean"),
        TotalActual=("UnitsSold", "sum"),
        TotalAbsoluteError=("AbsoluteError", "sum")
    )
    .reset_index()
)

product_errors["WAPE_percent"] = np.where(
    product_errors["TotalActual"] > 0,
    100 * product_errors["TotalAbsoluteError"]
    / product_errors["TotalActual"],
    np.nan
)

product_errors = product_errors.sort_values(
    "MAE", ascending=False
)

fig = px.bar(
    product_errors,
    x="StockCode",
    y="MAE",
    title="Forecast MAE by Product"
)
fig.write_html(REPORTS / "product_errors.html")

product_errors.to_csv(
    REPORTS / "product_errors.csv",
    index=False
)

# 4. Save combined daily predictions
daily.to_csv(
    REPORTS / "daily_forecast_comparison.csv",
    index=False
)

print("\n=== WORST 10 PRODUCTS BY MAE ===")
print(product_errors.head(10).to_string(index=False))

print("\nPhase 4 evaluation reports generated!")
