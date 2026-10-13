# SmartRetail AI — Inventory Optimization

## Objective

Develop a scenario-based inventory decision support module using historical retail demand to estimate replenishment requirements.

## Methodology

- Analyze daily product sales histories.
- Select 20 high-volume products.
- Estimate average demand and daily variability from 28-day historical windows.
- Calculate safety stock, reorder points, and order-up-to targets.
- Generate replenishment quantities using configurable inventory assumptions.
- Visualize potential replenishment needs in an interactive dashboard.

## Scenario assumptions

The analysis assumes seven-day supplier lead times, seven-day review periods, no existing purchase orders or backorders, and on-hand stock equivalent to five days of estimated demand.

## Results

- Products analyzed: [Insert actual result]
- Products requiring reorder: [Insert actual result]
- Total recommended units: [Insert actual result]
- Selected safety factor: [Insert actual value]

## Limitations

Actual inventory balances, supplier performance, delivery variability, stockout records, and procurement costs are unavailable.

Missing product-day transactions are provisionally treated as zero sales, which may not represent actual product availability. The safety stock equation assumes independent daily demand and normally approximated cumulative uncertainty.

The recommendations are scenario-based and should not be interpreted as production inventory decisions.

## Technology stack

Python, Pandas, NumPy, Plotly, Pytest.

## Future improvements

Integrate validated machine learning forecasts, real inventory balances, supplier lead-time distributions, backorder information, holding costs, and replenishment simulation.
