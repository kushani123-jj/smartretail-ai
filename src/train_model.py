
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

print("=== SMARTRETAIL AI: PHASE 3 ===")

# 1. Load daily sales data
df = pd.read_parquet(
    DATA / "daily_product_sales.parquet"
)

df["Date"] = pd.to_datetime(df["Date"])
df["StockCode"] = df["StockCode"].astype(str)

# 2. Select top 20 products by total units
# For this first experiment, product selection
# uses an initial historical selection period.
cutoff = df["Date"].min() + pd.Timedelta(days=365)

history = df[df["Date"] < cutoff]

top_products = (
    history.groupby("StockCode")["UnitsSold"]
    .sum()
    .nlargest(20)
    .index
)

df = df[df["StockCode"].isin(top_products)].copy()

# 3. Complete the daily calendar
# Missing rows are provisionally assumed zero sales.
# Phase 4 will examine this assumption.
all_dates = pd.date_range(
    df["Date"].min(),
    df["Date"].max(),
    freq="D"
)

all_index = pd.MultiIndex.from_product(
    [top_products, all_dates],
    names=["StockCode", "Date"]
)

df = (
    df.set_index(["StockCode", "Date"])
    .reindex(all_index)
    .fillna({"UnitsSold": 0, "RevenueGBP": 0})
    .reset_index()
)

df = df.sort_values(
    ["StockCode", "Date"]
).reset_index(drop=True)

# 4. Feature engineering
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["DayOfMonth"] = df["Date"].dt.day

grouped = df.groupby("StockCode")["UnitsSold"]

for lag in [1, 7, 14, 28]:
    df[f"Lag_{lag}"] = grouped.shift(lag)

for window in [7, 28]:
    df[f"RollingMean_{window}"] = (
        df.groupby("StockCode")["UnitsSold"]
        .transform(
            lambda s: s.shift(1)
            .rolling(window).mean()
        )
    )

# Encode product identity
df["ProductID"] = pd.Categorical(
    df["StockCode"],
    categories=list(top_products)
).codes

features = [
    "ProductID",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "RollingMean_7",
    "RollingMean_28"
]

df = df.dropna(subset=features + ["UnitsSold"])

# 5. Time-based train/test split
dates = sorted(df["Date"].unique())
split_date = dates[int(len(dates) * 0.8)]

train = df[df["Date"] < split_date]
test = df[df["Date"] >= split_date].copy()

X_train = train[features]
y_train = train["UnitsSold"]

X_test = test[features]
y_test = test["UnitsSold"]

print("Training records:", len(train))
print("Testing records:", len(test))
print("Split date:", split_date)

# 6. Train Random Forest
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=16,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 7. Generate predictions
predictions = np.maximum(
    model.predict(X_test), 0
)

# Seasonal naive baseline: same day last week
baseline = test["Lag_7"].to_numpy()

def evaluate(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    denominator = np.abs(actual).sum()
    wape = (
        100 * np.abs(actual - predicted).sum()
        / denominator
        if denominator > 0
        else None
    )

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "WAPE_percent": (
            round(float(wape), 2)
            if wape is not None else None
        )
    }

metrics = {
    "random_forest": evaluate(
        y_test, predictions
    ),
    "seasonal_naive": evaluate(
        y_test, baseline
    ),
    "train_rows": len(train),
    "test_rows": len(test),
    "split_date": str(pd.Timestamp(split_date).date()),
    "forecast_horizon": "one day ahead",
    "evaluation_type": "rolling one-step-ahead"
}

print("\n=== MODEL EVALUATION ===")
print(json.dumps(metrics, indent=2))

# 8. Save model and results
joblib.dump(
    {
        "model": model,
        "features": features,
        "products": list(top_products)
    },
    MODELS / "demand_forecast_rf.joblib"
)

test["PredictedUnits"] = predictions
test["BaselineUnits"] = baseline

test[
    [
        "Date",
        "StockCode",
        "UnitsSold",
        "PredictedUnits",
        "BaselineUnits"
    ]
].to_csv(
    REPORTS / "forecast_predictions.csv",
    index=False
)

with open(
    REPORTS / "model_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

print("\nModel and reports saved successfully.")
