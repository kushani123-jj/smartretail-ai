
import numpy as np
import pandas as pd

from src.customer_segmentation import (
    build_rfm,
    make_features,
    select_clusters,
    summarize_segments
)


def test_rfm_calculations():
    sample = pd.DataFrame({
        "CustomerID": [101, 101, 102, 103, None],
        "Invoice": ["A1", "A2", "B1", "C1", "X1"],
        "Date": [
            "2011-12-01",
            "2011-12-08",
            "2011-12-05",
            "2011-12-07",
            "2011-12-08"
        ],
        "RevenueGBP": [100, 50, 40, 80, 200]
    })

    rfm, quality = build_rfm(sample)

    assert len(rfm) == 3
    assert quality["eligible_sales_rows"] == 4

    customer = rfm.set_index("CustomerID").loc["101"]

    assert customer["Recency"] == 1
    assert customer["Frequency"] == 2
    assert np.isclose(customer["Monetary"], 150)

    # Unidentified customer's £200 must not enter RFM.
    assert np.isclose(rfm["Monetary"].sum(), 270)


def test_clustering_and_summary():
    rng = np.random.default_rng(42)

    sample = pd.DataFrame({
        "CustomerID": [str(i) for i in range(60)],
        "Recency": np.concatenate([
            rng.integers(1, 30, 30),
            rng.integers(120, 300, 30)
        ]),
        "Frequency": np.concatenate([
            rng.integers(8, 20, 30),
            rng.integers(1, 4, 30)
        ]),
        "Monetary": np.concatenate([
            rng.uniform(500, 2000, 30),
            rng.uniform(20, 200, 30)
        ])
    })

    X, _, _ = make_features(sample)
    model, scores, best_k = select_clusters(X)

    assert X.shape == (60, 3)
    assert 2 <= best_k <= 6
    assert not scores.empty

    sample["Segment"] = [
        f"Segment {i + 1}" for i in model.labels_
    ]

    summary = summarize_segments(sample)

    assert summary["Customers"].sum() == 60
    assert np.isclose(
        summary["RevenueSharePercent"].sum(), 100
    )
