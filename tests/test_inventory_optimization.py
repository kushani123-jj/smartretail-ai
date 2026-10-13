
import pytest

from src.inventory_optimization import calculate_inventory


def test_zero_demand():
    result = calculate_inventory(
        avg_daily_demand=0,
        daily_std=0,
        on_hand=10
    )

    assert result["RecommendedOrder"] == 0
    assert result["StockStatus"] == "Adequate"


def test_reorder_needed():
    result = calculate_inventory(
        avg_daily_demand=10,
        daily_std=0,
        on_hand=20,
        lead_time=7,
        review_period=7
    )

    assert result["ReorderPoint"] == 70
    assert result["RecommendedOrder"] == 120
    assert result["StockStatus"] == "Reorder"


def test_safety_stock():
    result = calculate_inventory(
        avg_daily_demand=10,
        daily_std=5,
        on_hand=20
    )

    assert result["SafetyStock"] > 0
    assert result["ReorderPoint"] > 70


def test_invalid_values():
    with pytest.raises(ValueError):
        calculate_inventory(
            avg_daily_demand=-5,
            daily_std=2,
            on_hand=10
        )
