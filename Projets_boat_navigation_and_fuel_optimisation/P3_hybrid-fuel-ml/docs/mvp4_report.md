# MVP-4 Report: Uncertainty Quantification

## Target and Context
Provide prediction intervals rather than point estimates, aiming for a 90% interval that actually contains ~90% of true outcomes (calibration target 85-92%). This is important for routing and fuel risk decisions.

## Work Done
- Implemented quantile regression and bootstrap ensembles (`src/models/uncertainty.py`).
- Added calibration metrics and plots (`src/calibration.py`).
- Trained and evaluated via `src/train_uncertainty.py`.
- Saved plots and metrics in `outputs/uncertainty/`.

## Results (Coverage)
From `outputs/uncertainty/uncertainty_comparison.csv`:
- Quantile: PICP 74.54% (val), 80.56% (test)
- Bootstrap: PICP 38.89% (val), 33.33% (test)

## Analysis (Contextual)
Both methods under-cover the target 90% interval, meaning the predicted intervals are too narrow and overly confident. Quantile regression is closer and provides a usable starting point, but still misses ~1 in 5 cases on test. Bootstrap performs poorly, likely because model variance alone does not capture data noise and bias in this setup. For decision-making, the intervals are not reliable yet; a post-hoc calibration step (e.g., conformal prediction or quantile calibration) is needed to achieve trustworthy coverage.
