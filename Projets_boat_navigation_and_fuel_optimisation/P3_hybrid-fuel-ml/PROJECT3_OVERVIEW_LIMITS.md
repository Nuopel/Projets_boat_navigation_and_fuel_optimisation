# Project Summary: ML-Enhanced Ship Fuel Prediction

## Context

This project builds a hybrid physics + machine learning system to predict ship fuel consumption from sparse operational and environmental inputs. It combines a physics-inspired baseline with ML residual learning and adds uncertainty quantification to support operational decision-making.

## Aim

- Produce accurate fuel predictions with interpretable physics priors.
- Benchmark baseline ML models vs. hybrid approaches.
- Provide calibrated uncertainty intervals for risk-aware planning.

## Results

- **MVP-1 (Data + EDA):** Clean dataset (1,440 rows, 120 vessels), strong distance-fuel correlation (r ~ 0.945), and publication-grade EDA plots.
- **MVP-2 (Baseline ML):** Ridge and XGBoost reach R2 ~ 0.95 on validation; ridge slightly stronger overall.
- **MVP-3 (Hybrid Physics-ML):** Physics-only model is weaker (R2 ~ 0.91); hybrid residual and feature-augmented models are competitive but do not beat ridge on test.
- **MVP-4 (Uncertainty):** Quantile and bootstrap intervals are under-covered (PICP < 0.85), indicating calibration gaps.
- **MVP-5 (API):** Not implemented.

## Core hypotheses

- Fuel scales with distance and operational conditions, with a physics prior (fuel ~ V^3 / efficiency) that can be corrected by ML residuals.
- Aggregated categorical features (ship type, route, weather, fuel type) capture most variance beyond distance.
- Uncertainty can be approximated with quantile regression or bootstrap ensembles on this dataset size.

## Limits

- **Data scope:** Single-region operational dataset with limited size and feature diversity; may not generalize to other fleets or routes.
- **Physics simplification:** The physics baseline is a coarse proxy and omits key hydrodynamic factors (speed profile, hull geometry, sea state).
- **Feature completeness:** No direct speed or engine load time series; weather is categorical, not continuous.
- **Calibration:** UQ intervals are under-covered; confidence bands are not yet reliable for risk-critical use.
- **Validation depth:** Cross-validation and external validation are not implemented.
  
  
