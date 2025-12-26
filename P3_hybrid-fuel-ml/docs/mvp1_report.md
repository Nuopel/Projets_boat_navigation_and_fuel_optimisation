# MVP-1 Report: Data Foundation and EDA

## Target and Context
Build a trustworthy data foundation and validate domain assumptions before modeling. The dataset is operational maritime data with mixed categorical and numeric features, so the MVP focuses on cleaning, encoding strategy, and visual validation of expected physical relationships (distance, weather, efficiency).

## Work Done
- Implemented `src/data_preprocessing.py` for cleaning, outlier handling, categorical encoding, and stratified splitting.
- Generated EDA visuals with `src/eda_visualizations.py` and stored them in `outputs/eda/`.
- Produced train/validation/test splits in `data/processed/` (70/15/15), stratified by ship type.

## Results (Observed)
- Data quality: no missing values reported; outliers handled by IQR capping.
- Dominant signal: distance vs. fuel shows very high correlation (r ≈ 0.945) and a strong linear trend.
- Weather: stormy conditions increase variability, consistent with higher resistance.
- CO2 emissions are nearly perfectly correlated with fuel consumption (leakage risk).

Artifacts:
- `outputs/eda/01_fuel_distribution.png`
- `outputs/eda/02_correlation_matrix.png`
- `outputs/eda/03_weather_impact.png`
- `outputs/eda/04_distance_vs_fuel.png`
- `outputs/eda/05_route_efficiency.png`

## Analysis (Contextual)
The EDA confirms a simple physical intuition: fuel scales with distance and worsens under harsher weather. This is why a linear model can be competitive later. It also highlights a key modeling risk: CO2 is essentially a proxy for fuel consumption and should not be used for prediction. The data is clean enough to proceed directly to modeling without heavy imputation or complex denoising, and the stratified split reduces the risk that a ship type dominates one split.
