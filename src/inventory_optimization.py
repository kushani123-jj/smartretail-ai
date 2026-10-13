
from pathlib import Path
from statistics import NormalDist
import json
import math

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "inventory"

LEAD_TIME = 7
REVIEW_PERIOD = 7
SERVICE_FACTOR = NormalDist().inv_cdf(0.95)
TOP_PRODUCTS = 20
HISTORY_DAYS = 56
FORECAST_DAYS = 28


def calculate_inventory(
    avg_daily_demand,
    daily_std,
    on_hand,
    on_order=0,
    backorders=0,
    lead_time=LEAD_TIME,
    review_period=REVIEW_PERIOD,
    z=SERVICE_FACTOR
):
    if min(
        avg_daily_demand, daily_std, on_hand,
        on_order, backorders
    ) < 0:
        raise ValueError("Inventory inputs cannot be negative")

    if lead_time <= 0 or review_period <= 0:
        raise ValueError("Planning periods must be positive")

    # Standard deviation assumes independent daily demand.
    lead_demand = avg_daily_demand * lead_time
    safety_stock = z * daily_std * math.sqrt(lead_time)

    reorder_point = lead_demand + safety_stock

    horizon = lead_time + review_period
    target_stock = (
        avg_daily_demand * horizon
        + z * daily_std * math.sqrt(horizon)
    )

    inventory_position = on_hand + on_order - backorders

    # Order only when inventory reaches reorder point.
    should_reorder = inventory_position <= reorder_point

    quantity = (
        max(0, math.ceil(target_stock - inventory_position))
        if should_reorder else 0
    )

    if inventory_position <= 0:
        status = "Critical"
    elif should_reorder:
        status = "Reorder"
    else:
        status = "Adequate"

    return {
        "SafetyStock": math.ceil(safety_stock),
        "ReorderPoint": math.ceil(reorder_point),
        "TargetStock": math.ceil(target_stock),
        "InventoryPosition": inventory_position,
        "RecommendedOrder": quantity,
        "StockStatus": status
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=== SMARTRETAIL AI: PHASE 6 ===")

    df = pd.read_parquet(
        DATA / "daily_product_sales.parquet"
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df["StockCode"] = df["StockCode"].astype(str)

    # Exclude obvious non-standard product codes.
    df = df[
        df["StockCode"].str.fullmatch(r"\d{5}[A-Z]?")
    ].copy()

    # Exclude the final date as it may be incomplete.
    last_day = df["Date"].max().normalize()
    end_date = last_day - pd.Timedelta(days=1)

    dates = pd.date_range(
        end=end_date, periods=HISTORY_DAYS, freq="D"
    )

    recent = df[df["Date"].isin(dates)].copy()

    top_codes = (
        recent.groupby("StockCode")["UnitsSold"]
        .sum()
        .nlargest(TOP_PRODUCTS)
        .index.tolist()
    )

    if not top_codes:
        raise ValueError("No product sales available")

    recent = recent[
        recent["StockCode"].isin(top_codes)
    ].copy()

    # Complete the calendar for every selected product.
    # Missing product-days are provisionally treated as zero.
    index = pd.MultiIndex.from_product(
        [top_codes, dates],
        names=["StockCode", "Date"]
    )

    panel = (
        recent.set_index(["StockCode", "Date"])
        .reindex(index)
        .fillna({"UnitsSold": 0, "RevenueGBP": 0})
        .reset_index()
    )

    results = []

    for product, group in panel.groupby("StockCode"):
        group = group.sort_values("Date")
        history = group["UnitsSold"].tail(FORECAST_DAYS)

        avg = max(0.0, float(history.mean()))
        std = max(0.0, float(history.std(ddof=0)))

        # Scenario assumption only:
        # 5 days of expected sales held as current stock.
        assumed_on_hand = math.ceil(avg * 5)

        policy = calculate_inventory(
            avg_daily_demand=avg,
            daily_std=std,
            on_hand=assumed_on_hand
        )

        results.append({
            "StockCode": product,
            "AvgDailyDemand": round(avg, 2),
            "DemandStd": round(std, 2),
            "AssumedOnHand": assumed_on_hand,
            "LeadTimeDays": LEAD_TIME,
            "ReviewPeriodDays": REVIEW_PERIOD,
            **policy
        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        ["RecommendedOrder", "StockCode"],
        ascending=[False, True]
    ).reset_index(drop=True)

    results.to_csv(
        REPORTS / "inventory_recommendations.csv",
        index=False
    )

    # Chart 1: Reorder recommendations
    fig = px.bar(
        results,
        x="StockCode",
        y="RecommendedOrder",
        color="StockStatus",
        title="Scenario-Based Recommended Order Quantities",
        labels={"RecommendedOrder": "Units to Order"}
    )

    fig.write_html(
        REPORTS / "reorder_recommendations.html"
    )

    # Chart 2: Assumed stock vs reorder point
    comparison = results[
        ["StockCode", "AssumedOnHand", "ReorderPoint"]
    ].melt(
        id_vars="StockCode",
        var_name="Metric",
        value_name="Units"
    )

    fig = px.bar(
        comparison,
        x="StockCode",
        y="Units",
        color="Metric",
        barmode="group",
        title="Assumed Stock vs Reorder Point"
    )

    fig.write_html(
        REPORTS / "inventory_levels.html"
    )

    # Chart 3: Stock status distribution
    status = (
        results["StockStatus"]
        .value_counts()
        .reset_index()
    )
    status.columns = ["Status", "Products"]

    fig = px.pie(
        status,
        names="Status",
        values="Products",
        title="Simulated Inventory Status"
    )

    fig.write_html(
        REPORTS / "inventory_risk.html"
    )

    metrics = {
        "analysis_end_date": str(end_date.date()),
        "products_analyzed": len(results),
        "critical_products": int(
            (results["StockStatus"] == "Critical").sum()
        ),
        "reorder_products": int(
            (results["StockStatus"] == "Reorder").sum()
        ),
        "total_recommended_units": int(
            results["RecommendedOrder"].sum()
        ),
        "assumptions": {
            "stock": "Five days of average demand",
            "on_order": 0,
            "backorders": 0,
            "lead_time_days": LEAD_TIME,
            "review_period_days": REVIEW_PERIOD,
            "historical_window_days": FORECAST_DAYS,
            "daily_demand_method": "Historical mean",
            "missing_product_days": "Assumed zero",
            "safety_factor": round(SERVICE_FACTOR, 4)
        }
    }

    with open(
        REPORTS / "inventory_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n=== INVENTORY RECOMMENDATIONS ===")
    print(results.to_string(index=False))

    print("\n=== SUMMARY ===")
    print(json.dumps(metrics, indent=2))
    print("\nPhase 6 completed successfully!")


if __name__ == "__main__":
    main()
