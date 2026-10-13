
from pathlib import Path
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# 1. Load processed data
sales = pd.read_parquet(DATA / "sales.parquet")
daily = pd.read_parquet(DATA / "daily_product_sales.parquet")

daily["Date"] = pd.to_datetime(daily["Date"])

print("\n=== SMARTRETAIL AI — PHASE 2 ===")
print("Sales transactions:", len(sales))
print("Daily product records:", len(daily))
print("Missing values:")
print(daily.isna().sum())

# 2. Daily sales trends
trends = (
    daily.groupby("Date", as_index=False)
    .agg(
        RevenueGBP=("RevenueGBP", "sum"),
        UnitsSold=("UnitsSold", "sum")
    )
    .sort_values("Date")
)

fig = px.line(
    trends,
    x="Date",
    y="RevenueGBP",
    title="Daily Retail Sales Revenue",
    labels={"RevenueGBP": "Revenue (GBP)"}
)
fig.write_html(REPORTS / "daily_revenue.html")

# 3. Monthly revenue
monthly = (
    trends.set_index("Date")
    .resample("MS")[["RevenueGBP", "UnitsSold"]]
    .sum()
    .reset_index()
)

fig = px.bar(
    monthly,
    x="Date",
    y="RevenueGBP",
    title="Monthly Retail Revenue",
    labels={"RevenueGBP": "Revenue (GBP)"}
)
fig.write_html(REPORTS / "monthly_revenue.html")

# 4. Top 10 products by revenue
products = (
    daily.groupby("StockCode", as_index=False)
    .agg(
        RevenueGBP=("RevenueGBP", "sum"),
        UnitsSold=("UnitsSold", "sum")
    )
    .sort_values("RevenueGBP", ascending=False)
    .head(10)
)

fig = px.bar(
    products,
    x="StockCode",
    y="RevenueGBP",
    title="Top 10 Products by Revenue"
)
fig.write_html(REPORTS / "top_products.html")

# 5. Save analysis tables
trends.to_csv(REPORTS / "daily_summary.csv", index=False)
monthly.to_csv(REPORTS / "monthly_summary.csv", index=False)
products.to_csv(REPORTS / "top_products.csv", index=False)

print("\n=== TOP 10 PRODUCTS ===")
print(products.to_string(index=False))

print("\n=== MONTHLY SALES SUMMARY ===")
print(monthly.tail(12).to_string(index=False))

print("\nPhase 2 initial EDA completed!")
print("Reports saved to:", REPORTS)
