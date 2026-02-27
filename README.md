# California Housing Price Prediction

## Overview
Predict California median house values using the classic California Housing dataset and a full Scikit-Learn preprocessing + regression pipeline.

## What’s in the Pipeline
- **Missing values:** median imputation for numeric, most-frequent for categorical  
- **Categorical encoding:** OneHotEncoder (`ocean_proximity`, unknowns ignored)
- **Feature engineering:** ratio features  
  - bedrooms_ratio = total_bedrooms / total_rooms  
  - rooms_per_house = total_rooms / households  
  - people_per_house = population / households
- **Heavy-tail handling:** log transform (`log1p`) for key numeric features
- **Geo features:** KMeans cluster centers + RBF similarities from (latitude, longitude)
- **Scaling:** StandardScaler for numeric features

## Model
- **Linear Regression** trained on the transformed features.

## Results
- **Test RMSE:** ~66,856 (your latest run)

## Files
- `california_housing_price_prediction.ipynb` — main notebook
- `requirements.txt` — dependencies

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate   # windows

pip install -r requirements.txt
