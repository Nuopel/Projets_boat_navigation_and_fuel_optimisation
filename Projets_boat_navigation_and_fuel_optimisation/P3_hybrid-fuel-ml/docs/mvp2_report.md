# MVP-2 Report: Baseline ML Models

## Target and Context
Establish baseline predictive performance to benchmark any hybrid or advanced model. Given the strong linear signal observed in MVP-1, Ridge regression is expected to be a strong baseline; XGBoost is included to capture non-linearities and interactions.

## Work Done
- Built physics-informed features and interactions in `src/feature_engineering.py`.
- Implemented Ridge/Lasso (`src/models/linear_model.py`) and XGBoost (`src/models/xgboost_model.py`).
- Trained and evaluated using `src/train_baseline_models.py`.
- Generated comparison plots and saved metrics in `outputs/baseline_models/`.

## Results (Validation)
From `outputs/baseline_models/model_comparison.csv`:
- Ridge: R2 0.9526, RMSE 998.76, MAPE 18.97%
- XGBoost: R2 0.9494, RMSE 1,032.53, MAPE 13.74%

## Analysis (Contextual)
Ridge edges out XGBoost on R2/RMSE, which matches the EDA: the relationship is strongly linear and dominated by distance. XGBoost improves MAPE, implying better relative accuracy on smaller/medium trips, but overall variance is already well captured by a linear model. This is a meaningful outcome: it sets a very high bar for any hybrid model and suggests that model complexity is not the limiting factor; data or feature richness might be.
