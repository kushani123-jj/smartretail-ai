import pandas as pd
from src.prepare_data import clean_sheet


def test_normalizes_both_years_and_flags_cancellations():
    sample = pd.DataFrame({
        'Invoice': ['123', 'C124'], 'StockCode': ['12345', '12345'],
        'Quantity': [2, -2], 'InvoiceDate': ['2010-01-01', '2010-01-02'],
        'Price': [3.5, 3.5], 'Customer ID': [12345.0, None], 'Country': ['United Kingdom'] * 2,
    })
    cleaned = clean_sheet(sample)
    assert list(cleaned['IsCancellation']) == [False, True]
    assert cleaned.iloc[0]['RevenueGBP'] == 7.0
    assert cleaned.iloc[0]['CustomerID'] == '12345'
