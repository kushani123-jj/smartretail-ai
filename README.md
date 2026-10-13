# SmartRetail AI

End-to-end retail demand forecasting and customer intelligence portfolio project.

## Dataset
Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D (CC BY 4.0).

Download `online_retail_II.xlsx` from https://archive.ics.uci.edu/dataset/502/online_retail+ii and place it under `data/raw/`.

## Phase 1 setup (PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.prepare_data
```

Outputs: `transactions.parquet` (retained cleaned transactions), `sales.parquet` (conservative valid sales), `daily_product_sales.parquet` (observed sold dates), `data_quality_report.json`.

Important: `daily_product_sales.parquet` includes *observed selling dates only*. Later phases must create a continuous per-product calendar and fill genuine no-sales dates with zero cautiously; do not treat unknown dates as zero without checking coverage.

Do not train on future information. Final evaluation will use time-aware splits and naive seasonal baselines.
