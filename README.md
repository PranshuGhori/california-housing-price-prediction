# California Housing Price Prediction

Predicting median house value for California districts (1990 census) with a scikit-learn preprocessing pipeline and gradient boosting.

## Results

RMSE in dollars. CV is 5-fold on the training set (16,512 districts); test is a single evaluation on the held-out, income-stratified 20% (4,128 districts).

| Model | CV RMSE | Test RMSE |
|---|---|---|
| LinearRegression | 73,717 ± 13,910 | 67,092 |
| RandomForest (200 trees) | 47,628 ± 757 | - |
| HistGradientBoosting (default) | 46,565 ± 1,132 | - |
| HistGradientBoosting (tuned) | 44,172 ± 745 | **42,204** (95% CI 40,375 - 43,975) |

The test CI is a 2,000-sample bootstrap over test districts. The linear CV spread comes from one fold with a few districts that have extreme household ratios. Errors are largest for districts at the $500k cap, which the model underpredicts by about $50k on average.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace california_housing_price_prediction.ipynb
```

The dataset (Géron's `housing.tgz`) is downloaded into `datasets/` on first run. The full notebook runs in about 40 seconds on a 10-core laptop.

## What's inside

- `california_housing_price_prediction.ipynb`: EDA, stratified split, model comparison, randomized search, test evaluation, error analysis, permutation importance.
- `housing.py`: data download and the preprocessing `ColumnTransformer`: imputation, one-hot `ocean_proximity`, ratio features (bedrooms/rooms, rooms/household, people/household), `log1p` on heavy-tailed counts, RBF similarity to KMeans geo clusters, scaling.
