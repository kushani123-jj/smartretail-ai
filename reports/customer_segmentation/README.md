# SmartRetail AI — Customer Segmentation

## Project objective

Develop an unsupervised machine learning system to identify meaningful customer purchasing patterns using historical retail transaction data.

## Dataset

**Source:** UCI Machine Learning Repository — Online Retail II (Daqing Chen, 2012).

**Coverage:** December 2009 – December 2011.

The dataset contains transactional records from a UK-based online retailer. The analysis uses valid positive sales records with identifiable customers.

## Methodology

1. Load and validate cleaned retail sales transactions.
2. Exclude records with missing customer IDs.
3. Calculate Recency, Frequency, and Monetary (RFM) features.
4. Apply logarithmic transformation and outlier clipping to reduce skew.
5. Standardize the features.
6. Evaluate K-Means clustering with candidate values of K from 2 to 6.
7. Select the model using silhouette scores and cluster-size checks.
8. Analyze segments by customer count, purchase frequency, recency, and gross revenue contribution.

## Results

- **Identified customers:** [Insert result]
- **Selected number of clusters:** [Insert K]
- **Best selected silhouette score:** [Insert score]
- **Largest customer segment:** [Insert result]
- **Segment with highest gross revenue contribution:** [Insert result]

## Business applications

Customer segments may help retailers tailor customer communications, identify frequently purchasing customers, prioritize re-engagement analysis, and investigate different spending behaviors.

These applications are potential uses, not experimentally verified business outcomes.

## Validation and limitations

Automated tests validate RFM calculations, missing-ID exclusions, and segment reconciliation.

Silhouette scores assess cluster separation but do not establish marketing effectiveness. The data is historical, stock availability is unknown, and cancellations are not netted against sales. Customer segments describe observed behavior within the dataset rather than future outcomes.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Plotly, Joblib, Pytest.

## Reproducibility

From the project root, run:

```powershell
python -m src.customer_segmentation
python -m pytest -q
```

## Dataset citation

Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
