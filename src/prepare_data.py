"""Phase 1: Normalize, validate, and preserve retail records.

Run from project root: python -m src.prepare_data
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'raw' / 'online_retail_II.xlsx'
OUT = ROOT / 'data' / 'processed'

RENAMES = {
    'Invoice': 'InvoiceNo', 'Customer ID': 'CustomerID',
    'Price': 'UnitPrice',
}
REQUIRED = {'InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'Country'}


def clean_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAMES).copy()
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if 'CustomerID' not in df:
        df['CustomerID'] = pd.NA
    if 'Description' not in df:
        df['Description'] = pd.NA
    df = df.drop_duplicates()
    df['InvoiceNo'] = df['InvoiceNo'].astype('string').str.strip()
    df['StockCode'] = df['StockCode'].astype('string').str.strip()
    df['Description'] = df['Description'].astype('string').str.strip()
    df['Country'] = df['Country'].astype('string').str.strip()
    df['CustomerID'] = df['CustomerID'].astype('string').str.replace(r'\.0$', '', regex=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df = df.dropna(subset=['InvoiceNo', 'StockCode', 'InvoiceDate', 'Quantity', 'UnitPrice'])
    df['IsCancellation'] = df['InvoiceNo'].str.upper().str.startswith('C', na=False)
    df['RevenueGBP'] = df['Quantity'] * df['UnitPrice']
    return df


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f'Download UCI Online Retail II and save it as: {SOURCE}')
    OUT.mkdir(parents=True, exist_ok=True)
    workbook = pd.ExcelFile(SOURCE)
    frames = []
    for sheet in workbook.sheet_names:
        print(f'Reading sheet: {sheet}')
        frames.append(clean_sheet(pd.read_excel(workbook, sheet_name=sheet)))
    transactions = pd.concat(frames, ignore_index=True)
    transactions.to_parquet(OUT / 'transactions.parquet', index=False)

    # Conservative first-pass demand view: exclude cancellations, negative/zero
    # quantities, non-positive unit prices, and special adjustment rows.
    sales = transactions.loc[
        ~transactions['IsCancellation']
        & transactions['Quantity'].gt(0)
        & transactions['UnitPrice'].gt(0)
        & transactions['StockCode'].str.fullmatch(r'\d{5}[A-Za-z]?', na=False)
    ].copy()
    sales['Date'] = sales['InvoiceDate'].dt.floor('D')
    sales.to_parquet(OUT / 'sales.parquet', index=False)
    daily = (sales.groupby(['Date', 'StockCode'], as_index=False)
             .agg(UnitsSold=('Quantity', 'sum'), RevenueGBP=('RevenueGBP', 'sum')))
    daily.to_parquet(OUT / 'daily_product_sales.parquet', index=False)
    summary = {
        'source_rows_after_dedup_and_required_field_checks': int(len(transactions)),
        'valid_sales_rows': int(len(sales)),
        'cancellation_rows': int(transactions['IsCancellation'].sum()),
        'missing_customer_ids_in_sales': int(sales['CustomerID'].isna().sum()),
        'start_date': str(sales['Date'].min().date()) if len(sales) else None,
        'end_date': str(sales['Date'].max().date()) if len(sales) else None,
        'unique_products': int(sales['StockCode'].nunique()),
        'total_revenue_gbp': round(float(sales['RevenueGBP'].sum()), 2),
    }
    (OUT / 'data_quality_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Phase 1 complete: outputs saved to {OUT}')


if __name__ == '__main__':
    main()
