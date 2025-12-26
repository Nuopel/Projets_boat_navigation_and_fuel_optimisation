# MVP-3 Report: Hybrid Physics-ML Models

## Target and Context
Test whether incorporating explicit physics improves predictive accuracy beyond the already-strong ML baseline. Two hybrid strategies are used:
- Residual correction: ML learns the error of the physics model.
- Feature augmentation: physics prediction becomes an extra feature.

## Work Done
- Implemented physics baseline (`src/models/physics_baseline.py`).
- Implemented hybrid models (`src/models/hybrid_model.py`).
- Trained and evaluated via `src/train_hybrid_models.py`.
- Saved plots and metrics in `outputs/hybrid_model/`.

## Results (Validation + Test)
Validation (`outputs/hybrid_model/hybrid_comparison.csv`):
- Ridge: R2 0.9526, RMSE 998.76
- Feature Hybrid: R2 0.9481, RMSE 1,045.18
- Residual Hybrid: R2 0.9471, RMSE 1,055.05
- Physics: R2 0.9077, RMSE 1,393.83

Test (`outputs/hybrid_model/test_comparison.csv`):
- Ridge: R2 0.9468, RMSE 1,186.64, MAPE 23.76%
- Residual Hybrid: R2 0.9436, RMSE 1,222.16, MAPE 12.68%
- Feature Hybrid: R2 0.9375, RMSE 1,286.83, MAPE 12.57%
- Physics: R2 0.9263, RMSE 1,397.56, MAPE 15.98%

## Analysis (Contextual)
The physics baseline alone is substantially weaker than ML, which is expected because it uses a simplified formula and ignores many operational effects. The hybrids also do not beat Ridge on R2/RMSE, indicating that the linear ML baseline already captures most variance in this dataset. However, the residual hybrid reduces MAPE on test by roughly half, which is a useful trade-off if relative error is the priority (e.g., comparing efficiency across ships/routes). In short: physics improves interpretability and relative stability but does not yet improve absolute accuracy.
