
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "customer_segmentation"

FEATURES = ["Recency", "Frequency", "Monetary"]


def find_column(df, alternatives):
    lookup = {
        str(c).lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }

    for name in alternatives:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Missing column. Expected one of {alternatives}. "
        f"Available: {list(df.columns)}"
    )


def build_rfm(sales):
    # Support common column names used in retail datasets.
    customer_col = find_column(
        sales, ["CustomerID", "Customer ID", "Customer_ID"]
    )
    invoice_col = find_column(
        sales, ["Invoice", "InvoiceNo", "InvoiceNumber"]
    )
    date_col = find_column(
        sales, ["Date", "InvoiceDate", "Invoice Date"]
    )
    revenue_col = find_column(
        sales, ["RevenueGBP", "Revenue"]
    )

    df = sales[
        [customer_col, invoice_col, date_col, revenue_col]
    ].copy()

    df.columns = [
        "CustomerID", "Invoice", "Date", "RevenueGBP"
    ]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["RevenueGBP"] = pd.to_numeric(
        df["RevenueGBP"], errors="coerce"
    )

    total_rows = len(df)

    # Remove records without the information needed for RFM.
    df = df.dropna(
        subset=["CustomerID", "Invoice", "Date", "RevenueGBP"]
    ).copy()

    df = df[df["RevenueGBP"] > 0].copy()

    # Normalize customer and invoice identifiers.
    df["CustomerID"] = (
        df["CustomerID"].astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    df["Invoice"] = (
        df["Invoice"].astype(str).str.strip()
    )

    df = df[
        ~df["CustomerID"].str.lower().isin(
            ["", "nan", "none", "nat", "<na>"]
        )
    ].copy()

    if df.empty:
        raise ValueError("No eligible identified-customer sales.")

    # Snapshot is one day after the last transaction.
    snapshot = df["Date"].max().normalize() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            LastPurchase=("Date", "max"),
            Frequency=("Invoice", "nunique"),
            Monetary=("RevenueGBP", "sum")
        )
        .reset_index()
    )

    rfm["Recency"] = (
        snapshot - rfm["LastPurchase"]
    ).dt.days

    rfm = rfm[
        ["CustomerID", "Recency", "Frequency", "Monetary"]
    ].copy()

    assert rfm["CustomerID"].is_unique
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()

    if not np.isclose(
        rfm["Monetary"].sum(),
        df["RevenueGBP"].sum(),
        rtol=1e-8,
        atol=0.01
    ):
        raise AssertionError("Customer revenue reconciliation failed.")

    quality = {
        "source_sales_rows": total_rows,
        "eligible_sales_rows": len(df),
        "excluded_sales_rows": total_rows - len(df),
        "identified_customers": len(rfm),
        "snapshot_date": str(snapshot.date()),
        "identified_sales_revenue_gbp": round(
            float(df["RevenueGBP"].sum()), 2
        )
    }

    return rfm, quality


def make_features(rfm):
    # K-Means is sensitive to extreme values.
    # Retain original RFM values for reporting.
    transformed = np.log1p(rfm[FEATURES])

    # Clip extremes for model fitting only.
    upper = transformed.quantile(0.99)
    transformed = transformed.clip(upper=upper, axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(transformed)

    return X, scaler, upper


def select_clusters(X):
    n = len(X)

    if n < 4:
        raise ValueError(
            "At least four identified customers are required."
        )

    max_k = min(6, n - 1)
    results = []
    fitted = {}

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
            max_iter=300
        )

        labels = model.fit_predict(X)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(
            X,
            labels,
            sample_size=min(3000, n),
            random_state=42
        )

        cluster_sizes = np.bincount(labels, minlength=k)
        minimum_share = float(cluster_sizes.min() / n)

        results.append({
            "K": k,
            "SilhouetteScore": float(score),
            "SmallestClusterShare": minimum_share
        })

        fitted[k] = model

    scores = pd.DataFrame(results)

    if scores.empty:
        raise ValueError(
            "Clustering could not identify valid segments."
        )

    # Prefer solutions without extremely tiny clusters.
    candidates = scores[
        scores["SmallestClusterShare"] >= 0.01
    ]

    if candidates.empty:
        candidates = scores

    best_k = int(
        candidates.sort_values(
            ["SilhouetteScore", "K"],
            ascending=[False, True]
        ).iloc[0]["K"]
    )

    return fitted[best_k], scores, best_k


def summarize_segments(rfm):
    summary = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            MedianRecency=("Recency", "median"),
            MedianFrequency=("Frequency", "median"),
            MedianMonetaryGBP=("Monetary", "median"),
            TotalRevenueGBP=("Monetary", "sum")
        )
        .reset_index()
    )

    summary["CustomerSharePercent"] = (
        100 * summary["Customers"] / len(rfm)
    )

    summary["RevenueSharePercent"] = (
        100 * summary["TotalRevenueGBP"]
        / rfm["Monetary"].sum()
    )

    return summary.sort_values(
        "TotalRevenueGBP", ascending=False
    )


def create_charts(rfm, summary, scores):
    # Chart 1: Cluster selection
    fig = px.line(
        scores,
        x="K",
        y="SilhouetteScore",
        markers=True,
        title="K-Means Cluster Selection"
    )
    fig.write_html(
        REPORTS / "silhouette_scores.html"
    )

    # Chart 2: Customer cluster scatter plot
    chart_df = rfm.copy()
    chart_df["BubbleSize"] = (
        chart_df["Frequency"].clip(upper=30)
    )

    fig = px.scatter(
        chart_df,
        x="Recency",
        y="Monetary",
        color="Segment",
        size="BubbleSize",
        hover_data=["Frequency"],
        log_y=True,
        title="Customer Segments: Recency vs Spending",
        labels={
            "Recency": "Days Since Last Purchase",
            "Monetary": "Total Purchase Value (GBP)"
        }
    )
    fig.write_html(
        REPORTS / "customer_clusters.html"
    )

    # Chart 3: Segment customer counts
    fig = px.bar(
        summary,
        x="Segment",
        y="Customers",
        color="Segment",
        title="Number of Customers by Segment"
    )
    fig.write_html(
        REPORTS / "segment_distribution.html"
    )

    # Chart 4: Revenue contribution
    fig = px.bar(
        summary,
        x="Segment",
        y="RevenueSharePercent",
        color="Segment",
        title="Revenue Contribution by Customer Segment",
        labels={
            "RevenueSharePercent": "Revenue Share (%)"
        }
    )
    fig.write_html(
        REPORTS / "segment_revenue.html"
    )


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=== SMARTRETAIL AI: PHASE 5 ===")

    sales = pd.read_parquet(DATA / "sales.parquet")

    print("Loaded transactions:", len(sales))
    print("Available columns:", list(sales.columns))

    # Build customer-level RFM features.
    rfm, quality = build_rfm(sales)

    print("\n=== DATA QUALITY ===")
    print(json.dumps(quality, indent=2))

    # Prepare features for clustering.
    X, scaler, upper = make_features(rfm)

    # Select and train the model.
    model, scores, best_k = select_clusters(X)

    rfm["Segment"] = [
        f"Segment {i + 1}"
        for i in model.labels_
    ]

    summary = summarize_segments(rfm)

    # Validate assignments and segment totals.
    assert rfm["Segment"].notna().all()
    assert summary["Customers"].sum() == len(rfm)
    assert np.isclose(
        summary["RevenueSharePercent"].sum(),
        100.0
    )

    # Save customer and segment reports.
    rfm.to_csv(
        REPORTS / "customer_rfm_segments.csv",
        index=False
    )

    summary.to_csv(
        REPORTS / "segment_summary.csv",
        index=False
    )

    scores.to_csv(
        REPORTS / "cluster_validation.csv",
        index=False
    )

    create_charts(rfm, summary, scores)

    metadata = {
        **quality,
        "selected_clusters": best_k,
        "features": FEATURES,
        "method": "KMeans on log1p RFM features",
        "scaling": "StandardScaler",
        "outlier_treatment": "99th percentile clipping",
        "random_state": 42
    }

    with open(
        REPORTS / "segmentation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "upper_clip": upper,
            "features": FEATURES,
            "metadata": metadata
        },
        MODELS / "customer_kmeans.joblib"
    )

    print("\n=== CLUSTER VALIDATION ===")
    print(scores.round(4).to_string(index=False))

    print("\nSelected clusters:", best_k)

    print("\n=== CUSTOMER SEGMENTS ===")
    print(summary.round(2).to_string(index=False))

    print("\nPhase 5 completed successfully!")
    print("Reports saved to:", REPORTS)


if __name__ == "__main__":
    main()
