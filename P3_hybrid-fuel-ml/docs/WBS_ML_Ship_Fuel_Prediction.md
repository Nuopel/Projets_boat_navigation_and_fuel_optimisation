# 🚢 Work Breakdown Structure (WBS)
## ML-Enhanced Ship Fuel Prediction with Uncertainty Quantification

**Project Timeline:** 2 weeks (80 developer-hours)
**Target Position:** Applied Mathematics Engineer
**Primary Skills Demonstrated:** Machine Learning, Data Analysis, Model Validation, Uncertainty Quantification

---

## Executive Summary

This project develops a hybrid physics-informed machine learning system for ship fuel consumption prediction with calibrated uncertainty intervals. The system will:

1. **Compare** multiple ML architectures (Linear, XGBoost, Neural Network) on real maritime data
2. **Innovate** through hybrid physics-ML modeling that combines domain knowledge with data-driven corrections
3. **Quantify** prediction uncertainty using multiple methods (quantile regression, bootstrapping)
4. **Deploy** a production-ready FastAPI for real-time fuel estimation with confidence intervals

**Key Innovation:** Unlike pure ML approaches, our hybrid model incorporates naval architecture principles (fuel ∝ V³ relationship, Froude number effects) while using ML to capture complex operational factors (weather, hull fouling, trim optimization).

---

## ✅ Implementation Status (2025-12-26)

- MVP-1/2/3/4 implemented with training scripts in `src/` and outputs in `outputs/`.
- Baselines: Ridge R² 0.9526 (val), 0.9468 (test); XGBoost R² 0.9494 (val).
- Hybrids did not exceed Ridge on R²/RMSE; residual hybrid improves MAPE on test.
- Uncertainty: Quantile PICP 0.8056 (test), Bootstrap PICP 0.3333 (test), below target 0.85-0.92.
- MVP-5 (API) not implemented.

---

## 📊 Dataset Strategy

### Primary Dataset: ship_fuel_efficiency.csv
- **Size:** 1,441 observations
- **Features:**
  - Operational: ship_id, ship_type, route_id, month, distance
  - Fuel: fuel_type (HFO/Diesel), fuel_consumption, CO2_emissions
  - Environmental: weather_conditions (Calm/Moderate/Stormy)
  - Performance: engine_efficiency
- **Target Variable:** fuel_consumption (tonnes or tonnes/hour derived)
- **Advantages:** Real-world operational data, multiple ship types, weather variations
- **Challenges:** Categorical features need encoding, limited technical specifications

### Secondary Dataset: navalplantmaintenance.csv (UCI Propulsion Plant)
- **Size:** 11,934 observations
- **Features:** 18 sensor readings from naval propulsion systems (speed, torque, power, temperatures, decay coefficients)
- **Usage:** Supplement ship_fuel_efficiency for physics model validation and feature engineering insights
- **Reference:** Mentioned in projet3.txt as recommended dataset

### Feature Engineering Strategy
Based on ship_fuel_efficiency.csv structure:
1. **Physics-Based Features:**
   - Fuel efficiency rate: fuel_consumption / distance
   - Fuel intensity: CO2_emissions / distance (proxy for fuel quality)
   - Speed estimation: distance / voyage_duration (if temporal data available)
   - Weather impact encoding (Calm=0, Moderate=1, Stormy=2)

2. **Categorical Encoding:**
   - Ship type (one-hot or target encoding)
   - Route patterns (frequency encoding for common routes)
   - Fuel type (binary: HFO=1, Diesel=0)
   - Month (cyclical encoding: sin/cos transformation for seasonality)

3. **Interaction Terms:**
   - distance × weather_conditions
   - engine_efficiency × fuel_type
   - ship_type × route_id (route specialization)

---

## 🎯 Critical Success Factors

### Technical Excellence
- [ ] **Model Performance:** Best ML model achieves R² > 0.80 on test set (realistic given operational data complexity)
- [ ] **Hybrid Advantage:** Hybrid model reduces RMSE by ≥8% vs. best pure ML (demonstrates value of physics integration)
- [ ] **Uncertainty Calibration:** 90% confidence intervals contain 85-92% of test points (well-calibrated)
- [ ] **Feature Interpretability:** SHAP analysis shows distance, weather, engine_efficiency as top-3 features (aligns with domain knowledge)

### Code Quality
- [ ] **Test Coverage:** ≥85% for src/models/ and src/data_preprocessing.py
- [ ] **Type Safety:** All functions have type hints (PEP 484)
- [ ] **Documentation:** Google-style docstrings with examples for all public APIs
- [ ] **Modularity:** Clear separation: data → models → evaluation → API layers

### Professional Delivery
- [ ] **API Response Time:** <200ms for single prediction (production-ready)
- [ ] **Reproducibility:** One-command setup (pip install -r requirements.txt)
- [ ] **Visualizations:** Minimum 6 publication-quality plots (predicted vs. actual, residuals, SHAP, calibration, model comparison, uncertainty intervals)
- [ ] **Documentation:** README, data dictionary, model card, API docs

---

## 📋 MVP Breakdown

### MVP-1: Data Foundation & Exploratory Analysis
**Duration:** 2 developer-days
**Goal:** Establish clean, analysis-ready dataset with baseline understanding

#### Components

**1.1 Data Acquisition & Quality Assessment**
```python
# Tasks
- Load ship_fuel_efficiency.csv and navalplantmaintenance.csv
- Profile data: missing values, outliers, distributions
- Validate data integrity (no negative fuel consumption, realistic ranges)
- Document data dictionary with units and value ranges
```

**1.2 Exploratory Data Analysis (EDA)**
```python
# Deliverables
- Distribution plots for all numerical features
- Correlation heatmap (identify multicollinearity)
- Categorical feature analysis (ship_type, fuel_type frequencies)
- Weather impact visualization (fuel consumption by weather condition)
- Route analysis (identify fuel-efficient vs. inefficient routes)
```

**1.3 Data Preprocessing Pipeline**
```python
# src/data_preprocessing.py
class DataPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        """Initialize with outlier thresholds, scaling method, etc."""

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values, remove outliers, validate ranges."""
        # Strategy: Drop rows with missing fuel_consumption (target)
        # Impute missing engine_efficiency with median by ship_type

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ship_type, fuel_type, weather, route, month."""
        # Use TargetEncoder for high-cardinality (route_id, ship_id)
        # One-hot for low-cardinality (fuel_type, weather_conditions)

    def split_data(self, df: pd.DataFrame,
                   ratios: tuple = (0.7, 0.15, 0.15)) -> tuple:
        """Stratified split by ship_type to ensure representation."""
        # Return: train, val, test DataFrames
```

**1.4 Train/Validation/Test Split**
```python
# Strategy
- 70% training (1,009 samples)
- 15% validation (216 samples) - for hyperparameter tuning
- 15% test (216 samples) - final evaluation, never touched until MVP-4
- Stratification: by ship_type to ensure all types in each split
- Random seed: 42 (reproducibility)
```

#### Acceptance Criteria
- [ ] Data dictionary created in data/README.md documenting all 10 features
- [ ] Missing values handled: <5% data loss from cleaning
- [ ] Outlier detection: flag extreme values (e.g., fuel_consumption > 99th percentile) for review
- [ ] EDA notebook (notebooks/01_eda.ipynb) executes without errors
- [ ] 5 key visualizations saved to outputs/eda/:
  - Fuel consumption distribution (histogram + boxplot by ship_type)
  - Correlation matrix heatmap
  - Weather impact (violin plot: fuel vs. weather_conditions)
  - Distance vs. fuel scatter (colored by ship_type)
  - Route efficiency comparison (bar chart: top 10 routes by avg fuel/distance)
- [ ] Preprocessing pipeline tested: test_data_preprocessing.py passes all unit tests
- [ ] Train/val/test splits saved to data/processed/ with manifest file

#### Tests
**Unit Tests:**
```python
# tests/test_data_preprocessing.py
def test_clean_data_removes_outliers():
    """Verify outliers beyond 3σ are removed."""

def test_categorical_encoding_preserves_shape():
    """Ensure encoding doesn't drop rows unexpectedly."""

def test_train_test_split_stratification():
    """Verify ship_type distribution similar across splits."""
```

**Integration Test:**
```python
def test_full_preprocessing_pipeline():
    """Raw CSV → cleaned, encoded, split data in <5 seconds."""
    # Use small subset (100 rows) for speed
```

#### Deliverables
1. `data/README.md` - Data dictionary with feature descriptions
2. `notebooks/01_eda.ipynb` - Interactive EDA with visualizations
3. `src/data_preprocessing.py` - Preprocessing module with tests
4. `data/processed/train.csv`, `val.csv`, `test.csv` - Split datasets
5. `outputs/eda/` - 5 publication-quality plots

#### Risks & Mitigation
**Risk:** High missing data rate (>20%) in key features
**Mitigation:** If engine_efficiency or weather has >20% missing, use multiple imputation (MICE) instead of simple median/mode. Allocate +0.5 day buffer.

**Risk:** Insufficient samples for train/val/test (1,441 might be small for neural networks)
**Mitigation:** Use k-fold cross-validation (k=5) as backup strategy for model evaluation if test set variance is high.

---

### MVP-2: Baseline ML Models & Evaluation Framework
**Duration:** 3 developer-days
**Goal:** Establish ML performance baselines and comparison infrastructure

#### Components

**2.1 Feature Engineering Module**
```python
# src/feature_engineering.py
class FeatureEngineer:
    def create_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-informed features."""
        # Derived features:
        # - fuel_rate = fuel_consumption / distance (efficiency metric)
        # - speed_proxy = distance / 24 (assuming daily reports)
        # - co2_intensity = CO2_emissions / fuel_consumption (fuel quality)
        # - efficiency_deviation = engine_efficiency - mean(by ship_type)

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-product features for non-linear patterns."""
        # - distance_x_weather (longer voyages more weather-sensitive)
        # - efficiency_x_fuel_type (diesel engines may respond differently)
        # - month_sin, month_cos (cyclical seasonality)
```

**2.2 Baseline Model Implementations**

**Linear Model (Ridge Regression)**
```python
# src/models/linear_model.py
class LinearFuelPredictor:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit with feature scaling."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return coefficients as importance proxy."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
```

**XGBoost Model**
```python
# src/models/xgboost_model.py
class XGBoostFuelPredictor:
    def __init__(self, params: dict = None):
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.params = {**default_params, **(params or {})}
        self.model = XGBRegressor(**self.params)

    def train(self, X, y, eval_set=None):
        """Train with early stopping on validation set."""
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self, importance_type='gain'):
        """Return XGBoost feature importance."""
        # Types: 'weight', 'gain', 'cover'
```

**Neural Network Model**
```python
# src/models/neural_network.py
import torch
import torch.nn as nn

class FuelConsumptionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralNetPredictor:
    def __init__(self, input_dim: int, config: dict = None):
        self.model = FuelConsumptionNet(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train with early stopping."""
        # Training loop with validation monitoring
```

**2.3 Hyperparameter Tuning**
```python
# src/hyperparameter_tuning.py
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_xgboost(X_train, y_train):
    """Use RandomizedSearchCV for efficiency."""
    param_distributions = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42),
        param_distributions,
        n_iter=20,  # 20 random combinations
        cv=5,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_params_
```

**2.4 Evaluation Module**
```python
# src/evaluation.py
class ModelEvaluator:
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Comprehensive regression metrics."""
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred),
            'Max_Error': np.max(np.abs(y_true - y_pred))
        }

    @staticmethod
    def plot_predictions(y_true, y_pred, title='', save_path=None):
        """Scatter: predicted vs. actual with perfect prediction line."""

    @staticmethod
    def plot_residuals(y_true, y_pred, save_path=None):
        """Residual plot to check for heteroscedasticity."""

    @staticmethod
    def create_comparison_table(results: dict) -> pd.DataFrame:
        """Markdown table comparing all models."""
        # results = {'Linear': {...}, 'XGBoost': {...}, 'NN': {...}}
```

#### Acceptance Criteria
- [ ] All three models trained and saved to models/trained/
  - linear_model.pkl
  - xgboost_model.pkl
  - neural_net.pth
- [ ] XGBoost achieves R² > 0.75 on validation set (performance threshold given dataset)
- [ ] Neural network converges without overfitting (train-val loss gap <10%)
- [ ] Model comparison table generated showing RMSE, MAE, MAPE, R² for all models
- [ ] Hyperparameter tuning results logged to outputs/tuning_results.json
- [ ] 4 visualizations per model saved to outputs/baseline_models/ (12 total):
  - Predicted vs. Actual scatter
  - Residual plot
  - Feature importance (top 10 features)
  - Learning curve (for NN) or cross-validation scores (for Linear/XGB)
- [ ] Test coverage ≥80% for src/models/
- [ ] Training time logged: each model trains in <5 minutes on CPU

#### Tests
**Unit Tests:**
```python
def test_linear_model_prediction_shape():
    """Verify output shape matches input samples."""

def test_xgboost_feature_importance_sum():
    """Importance scores should sum to 1.0 or be non-negative."""

def test_neural_net_forward_pass():
    """Ensure network produces single output per sample."""
```

**Integration Tests:**
```python
def test_full_training_pipeline():
    """Data loading → feature engineering → model training → metrics."""
    # Use validation set for integration test

def test_model_persistence():
    """Save and load model, predictions should be identical."""
```

#### Deliverables
1. `src/feature_engineering.py` - Feature creation module
2. `src/models/linear_model.py` - Ridge regression implementation
3. `src/models/xgboost_model.py` - XGBoost wrapper
4. `src/models/neural_network.py` - PyTorch MLP
5. `src/hyperparameter_tuning.py` - Tuning utilities
6. `src/evaluation.py` - Metrics and visualization
7. `notebooks/02_baseline_models.ipynb` - Training and comparison
8. `outputs/baseline_models/` - 12 visualization plots
9. `models/trained/` - Serialized models
10. `outputs/model_comparison.csv` - Performance metrics table

#### Time Breakdown
- Day 1: Feature engineering + Linear model (6 hours)
- Day 2: XGBoost + hyperparameter tuning (8 hours)
- Day 3: Neural network + evaluation framework + tests (10 hours)

#### Risks & Mitigation
**Risk:** Neural network underperforms due to small dataset (1,441 samples)
**Mitigation:** Use aggressive regularization (dropout=0.3, L2 penalty). If R² < 0.6, switch to simpler 2-layer network or skip NN in favor of ensemble methods.

**Risk:** Hyperparameter tuning takes too long (>2 hours)
**Mitigation:** Reduce RandomizedSearchCV n_iter from 20 to 10. Use validation set instead of 5-fold CV.

---

### MVP-3: Hybrid Physics-ML Model (KEY INNOVATION)
**Duration:** 2.5 developer-days
**Goal:** Combine domain knowledge with data-driven learning for superior performance

#### Components

**3.1 Physics Baseline Model**
```python
# src/models/physics_baseline.py
class PhysicsBasedFuelModel:
    """Simplified fuel consumption model based on naval architecture."""

    def __init__(self):
        # Admiralty coefficient approach or resistance-based
        self.k = None  # To be calibrated from data

    def predict(self, distance: float, ship_type: str,
                weather: str, engine_efficiency: float) -> float:
        """
        Simplified physics model:
        Fuel = k * distance * weather_factor / engine_efficiency

        Where:
        - k varies by ship_type (to be estimated from training data)
        - weather_factor: Calm=1.0, Moderate=1.2, Stormy=1.5
        - engine_efficiency is a penalty term (higher = less fuel)
        """
        weather_factors = {'Calm': 1.0, 'Moderate': 1.2, 'Stormy': 1.5}
        ship_k = self.ship_coefficients.get(ship_type, 1.0)

        fuel_physics = (ship_k * distance * weather_factors[weather]
                       / (engine_efficiency / 100.0))
        return fuel_physics

    def calibrate(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fit ship_type coefficients to minimize MSE on training data."""
        # Use least squares to find optimal k for each ship_type
        from scipy.optimize import minimize

        def loss(coefficients):
            predictions = [
                self.predict(row['distance'], row['ship_type'],
                           row['weather_conditions'], row['engine_efficiency'])
                for _, row in X_train.iterrows()
            ]
            return np.mean((y_train - predictions) ** 2)

        # Optimize coefficients
```

**3.2 Hybrid Model Architecture - Approach 1: Residual Correction**
```python
# src/models/hybrid_model.py
class ResidualCorrectionHybrid:
    """
    Hybrid approach: Physics prediction + ML residual correction
    fuel_pred = fuel_physics + ML_model(features)
    """

    def __init__(self, physics_model, ml_model):
        self.physics_model = physics_model
        self.ml_model = ml_model  # XGBoost or NN

    def train(self, X_train, y_train):
        # Step 1: Get physics predictions
        y_physics = np.array([
            self.physics_model.predict(
                row['distance'], row['ship_type'],
                row['weather_conditions'], row['engine_efficiency']
            ) for _, row in X_train.iterrows()
        ])

        # Step 2: Compute residuals (true - physics)
        residuals = y_train - y_physics

        # Step 3: Train ML model to predict residuals
        features = X_train.drop(columns=['ship_id'])  # Remove identifiers
        self.ml_model.train(features, residuals)

    def predict(self, X):
        # Physics prediction
        y_physics = np.array([
            self.physics_model.predict(
                row['distance'], row['ship_type'],
                row['weather_conditions'], row['engine_efficiency']
            ) for _, row in X.iterrows()
        ])

        # ML correction
        features = X.drop(columns=['ship_id'])
        residual_pred = self.ml_model.predict(features)

        return y_physics + residual_pred
```

**3.3 Hybrid Model Architecture - Approach 2: Feature Augmentation**
```python
class FeatureAugmentationHybrid:
    """
    Hybrid approach: Include physics prediction as feature
    fuel_pred = ML_model([original_features, fuel_physics])
    """

    def __init__(self, physics_model, ml_model):
        self.physics_model = physics_model
        self.ml_model = ml_model

    def train(self, X_train, y_train):
        # Add physics prediction as feature
        X_augmented = X_train.copy()
        X_augmented['fuel_physics'] = np.array([
            self.physics_model.predict(
                row['distance'], row['ship_type'],
                row['weather_conditions'], row['engine_efficiency']
            ) for _, row in X_train.iterrows()
        ])

        # Train ML model on augmented features
        self.ml_model.train(X_augmented, y_train)

    def predict(self, X):
        X_augmented = X.copy()
        X_augmented['fuel_physics'] = np.array([...])
        return self.ml_model.predict(X_augmented)
```

**3.4 SHAP Feature Importance Analysis**
```python
# src/interpretability.py
import shap

class FeatureImportanceAnalyzer:
    def compute_shap_values(self, model, X_background, X_explain):
        """
        Use SHAP to explain model predictions.

        Args:
            model: Trained model with predict method
            X_background: Background dataset for SHAP (subset of train)
            X_explain: Samples to explain (validation or test set)
        """
        explainer = shap.Explainer(model.predict, X_background)
        shap_values = explainer(X_explain)
        return shap_values

    def plot_shap_summary(self, shap_values, feature_names, save_path):
        """Beeswarm plot showing feature importance."""
        shap.summary_plot(shap_values, feature_names=feature_names,
                         show=False)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_shap_waterfall(self, shap_values, sample_idx, save_path):
        """Waterfall plot for individual prediction explanation."""
        shap.waterfall_plot(shap_values[sample_idx], show=False)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

#### Acceptance Criteria
- [ ] Physics baseline calibrated and achieves R² > 0.55 (interpretable lower bound)
- [ ] Both hybrid approaches implemented and tested
- [ ] Best hybrid model outperforms best pure ML by ≥8% RMSE reduction
- [ ] SHAP analysis shows physics-informed features contribute meaningfully
  - For Residual Correction: residuals are smaller than physics errors
  - For Feature Augmentation: fuel_physics has high SHAP importance
- [ ] Domain validation: Feature importance aligns with maritime knowledge
  - distance, weather_conditions, engine_efficiency in top-5 features
  - SHAP direction correct (more distance → more fuel, better efficiency → less fuel)
- [ ] 5 visualizations saved to outputs/hybrid_model/:
  - Physics baseline vs. actual scatter
  - Residual distribution (physics errors)
  - Hybrid vs. pure ML comparison (side-by-side scatter)
  - SHAP beeswarm summary plot
  - SHAP waterfall for 3 sample predictions (Calm, Moderate, Stormy)
- [ ] Comparison table updated with hybrid model metrics
- [ ] Test coverage ≥85% for src/models/hybrid_model.py

#### Tests
**Unit Tests:**
```python
def test_physics_model_weather_effect():
    """Stormy should increase fuel vs. Calm for same distance."""

def test_residual_hybrid_reduces_physics_error():
    """Hybrid RMSE < Physics RMSE on validation set."""

def test_feature_augmentation_includes_physics():
    """Verify fuel_physics column added to features."""
```

**Integration Tests:**
```python
def test_hybrid_model_end_to_end():
    """Train physics → calibrate → train ML → predict → evaluate."""

def test_shap_values_sum_to_prediction():
    """SHAP values + base value ≈ model prediction."""
```

#### Deliverables
1. `src/models/physics_baseline.py` - Naval architecture-based model
2. `src/models/hybrid_model.py` - Both hybrid architectures
3. `src/interpretability.py` - SHAP analysis utilities
4. `notebooks/03_hybrid_modeling.ipynb` - Training and comparison
5. `outputs/hybrid_model/` - 5 visualizations
6. `models/trained/hybrid_residual.pkl` and `hybrid_feature.pkl`
7. Updated `outputs/model_comparison.csv` with hybrid results

#### Time Breakdown
- Day 1: Physics baseline development and calibration (6 hours)
- Day 2: Hybrid implementations (both approaches) (8 hours)
- Day 3 (half): SHAP analysis and visualization (4 hours)

#### Risks & Mitigation
**Risk:** Physics model too simplistic → hybrid doesn't outperform pure ML
**Mitigation:** If Approach 1 (Residual Correction) fails to improve, pivot to Approach 2 (Feature Augmentation) which is more robust. Allocate 1-hour buffer for pivot.

**Risk:** Negative transfer (hybrid worse than pure ML due to bad physics priors)
**Mitigation:** Validate physics model separately first (R² > 0.5 on train set). If below threshold, simplify to just distance and weather as physics features.

**Risk:** SHAP computation too slow (>10 min for 1,441 samples)
**Mitigation:** Use TreeSHAP for XGBoost (fast), or subsample to 500 background samples for KernelSHAP.

---

### MVP-4: Uncertainty Quantification & Calibration
**Duration:** 2 developer-days
**Goal:** Provide calibrated prediction intervals essential for operational decision-making

#### Components

**4.1 Quantile Regression Approach**
```python
# src/models/uncertainty.py
from xgboost import XGBRegressor

class QuantileRegressionUncertainty:
    """
    Train separate models for different quantiles.
    Provides prediction intervals: [q05, q50, q95]
    """

    def __init__(self, base_model='xgboost'):
        self.models = {
            'q05': XGBRegressor(objective='reg:quantileerror',
                               quantile_alpha=0.05),
            'q50': XGBRegressor(objective='reg:squarederror'),  # Median
            'q95': XGBRegressor(objective='reg:quantileerror',
                               quantile_alpha=0.95)
        }

    def train(self, X_train, y_train):
        """Train three models independently."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict_with_intervals(self, X, confidence=0.90):
        """
        Returns: (predictions, lower_bound, upper_bound)
        confidence=0.90 → use q05 and q95
        """
        q05 = self.models['q05'].predict(X)
        q50 = self.models['q50'].predict(X)
        q95 = self.models['q95'].predict(X)
        return q50, q05, q95
```

**4.2 Bootstrapping Approach (Alternative)**
```python
class BootstrapUncertainty:
    """
    Train multiple models on bootstrap samples.
    Uncertainty from prediction variance across models.
    """

    def __init__(self, base_model_class, n_bootstrap=50):
        self.n_bootstrap = n_bootstrap
        self.models = []
        self.base_model_class = base_model_class

    def train(self, X_train, y_train):
        """Train N models on random subsamples with replacement."""
        n_samples = len(X_train)
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_train.iloc[indices]
            y_boot = y_train[indices]

            # Train model
            model = self.base_model_class()
            model.train(X_boot, y_boot)
            self.models.append(model)

    def predict_with_intervals(self, X, confidence=0.90):
        """Aggregate predictions from all models."""
        predictions = np.array([model.predict(X) for model in self.models])

        mean_pred = np.mean(predictions, axis=0)
        lower_percentile = (1 - confidence) / 2 * 100  # 5 for 90% CI
        upper_percentile = (1 + confidence) / 2 * 100  # 95 for 90% CI

        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)

        return mean_pred, lower_bound, upper_bound
```

**4.3 Calibration Analysis**
```python
# src/calibration.py
class CalibrationAnalyzer:
    @staticmethod
    def compute_coverage(y_true, lower_bound, upper_bound):
        """
        What % of true values fall within [lower, upper]?
        For 90% CI, should be ~90%.
        """
        within_interval = ((y_true >= lower_bound) &
                          (y_true <= upper_bound))
        coverage = np.mean(within_interval) * 100
        return coverage

    @staticmethod
    def compute_sharpness(lower_bound, upper_bound):
        """
        Average interval width. Smaller = more precise.
        """
        return np.mean(upper_bound - lower_bound)

    @staticmethod
    def plot_calibration(y_true, predictions, lower, upper, save_path):
        """
        Scatter plot with error bars showing prediction intervals.
        Color code: green if true value in interval, red otherwise.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Error bars
        errors = np.array([predictions - lower, upper - predictions])

        # Color: within interval or not
        colors = ['green' if (y >= l and y <= u) else 'red'
                 for y, l, u in zip(y_true, lower, upper)]

        ax.errorbar(y_true, predictions, yerr=errors, fmt='o',
                   alpha=0.6, ecolor=colors, elinewidth=2)
        ax.plot([y_true.min(), y_true.max()],
               [y_true.min(), y_true.max()],
               'k--', label='Perfect prediction')
        ax.set_xlabel('True Fuel Consumption')
        ax.set_ylabel('Predicted Fuel Consumption')
        ax.set_title('Prediction Intervals Calibration')
        ax.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    @staticmethod
    def plot_interval_width_distribution(lower, upper, save_path):
        """Histogram of interval widths to identify overconfident regions."""
        widths = upper - lower
        plt.figure(figsize=(8, 5))
        plt.hist(widths, bins=30, edgecolor='black')
        plt.xlabel('Interval Width (Fuel Consumption Units)')
        plt.ylabel('Frequency')
        plt.title('Distribution of 90% Prediction Interval Widths')
        plt.axvline(widths.mean(), color='red', linestyle='--',
                   label=f'Mean: {widths.mean():.2f}')
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

**4.4 Uncertainty-Aware Metrics**
```python
class UncertaintyMetrics:
    @staticmethod
    def prediction_interval_coverage_probability(y_true, lower, upper):
        """PICP: Coverage rate (should match nominal confidence)."""
        return np.mean((y_true >= lower) & (y_true <= upper))

    @staticmethod
    def mean_prediction_interval_width(lower, upper):
        """MPIW: Average interval width (lower is better if well-calibrated)."""
        return np.mean(upper - lower)

    @staticmethod
    def coverage_width_criterion(y_true, lower, upper, alpha=0.1):
        """
        CWC: Balance coverage and sharpness.
        Penalize if coverage < nominal (1-alpha).
        """
        picp = UncertaintyMetrics.prediction_interval_coverage_probability(
            y_true, lower, upper
        )
        mpiw = UncertaintyMetrics.mean_prediction_interval_width(lower, upper)

        # Penalty if undercoverage
        eta = 100  # Penalty coefficient
        coverage_penalty = eta * np.max([0, (1 - alpha) - picp])

        cwc = mpiw + coverage_penalty
        return cwc
```

#### Acceptance Criteria
- [ ] Quantile regression model trained for q05, q50, q95
- [ ] Bootstrap ensemble (n=30 models) trained for uncertainty estimation
- [ ] Calibration analysis on test set shows:
  - 90% confidence intervals contain 85-92% of true values (PICP ∈ [0.85, 0.92])
  - Mean interval width <30% of mean fuel consumption (sharp enough to be useful)
- [ ] Uncertainty increases appropriately:
  - Stormy weather → wider intervals than Calm
  - Longer distances → wider intervals (more uncertainty propagation)
- [ ] 4 visualizations saved to outputs/uncertainty/:
  - Calibration plot (scatter with error bars)
  - Interval width distribution histogram
  - Coverage by weather condition (bar chart: Calm vs. Moderate vs. Stormy)
  - Sample predictions with intervals (table visualization for 10 test cases)
- [ ] Comparison table updated with uncertainty metrics (PICP, MPIW, CWC)
- [ ] Test coverage ≥80% for src/models/uncertainty.py and src/calibration.py

#### Tests
**Unit Tests:**
```python
def test_quantile_consistency():
    """q05 ≤ q50 ≤ q95 for all predictions."""

def test_bootstrap_predictions_variance():
    """Bootstrap predictions have non-zero variance (not deterministic)."""

def test_coverage_calculation():
    """Known intervals should give exact coverage."""
```

**Integration Tests:**
```python
def test_uncertainty_pipeline():
    """Train → predict with intervals → calibration analysis."""

def test_interval_monotonicity():
    """Higher confidence → wider intervals."""
```

#### Deliverables
1. `src/models/uncertainty.py` - Quantile regression and bootstrapping
2. `src/calibration.py` - Calibration metrics and plots
3. `notebooks/04_uncertainty_quantification.ipynb` - Training and analysis
4. `outputs/uncertainty/` - 4 visualizations
5. `models/trained/uncertainty_quantile.pkl` and `uncertainty_bootstrap.pkl`
6. Updated `outputs/model_comparison.csv` with uncertainty metrics

#### Time Breakdown
- Day 1: Quantile regression + bootstrapping implementation (6 hours)
- Day 2: Calibration analysis + visualizations + testing (10 hours)

#### Risks & Mitigation
**Risk:** Quantile regression produces crossing quantiles (q05 > q50)
**Mitigation:** Post-process to enforce monotonicity: q05 = min(q05, q50), q95 = max(q50, q95).

**Risk:** Bootstrap too slow (30 models × 2 min training = 60 min total)
**Mitigation:** Reduce to n=20 bootstrap models if training time >30 min. Use joblib parallelization.

**Risk:** Intervals too wide (uninformative) or too narrow (overconfident)
**Mitigation:** If MPIW >50% of mean: switch from 90% to 80% CI for better sharpness. If PICP <80%: retrain with different quantile_alpha tuning.

---

### MVP-5: Production API & Deployment (BONUS)
**Duration:** 1.5 developer-days
**Goal:** Demonstrate production-readiness through REST API deployment

#### Components

**5.1 FastAPI Application**
```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np

app = FastAPI(
    title="Ship Fuel Consumption Prediction API",
    description="ML-powered fuel prediction with uncertainty quantification",
    version="1.0.0"
)

# Load trained models at startup
@app.on_event("startup")
def load_models():
    global hybrid_model, uncertainty_model
    hybrid_model = joblib.load('models/trained/hybrid_feature.pkl')
    uncertainty_model = joblib.load('models/trained/uncertainty_quantile.pkl')

class FuelPredictionRequest(BaseModel):
    """Input schema with validation."""
    ship_type: str = Field(..., description="Type of ship",
                          example="Oil Service Boat")
    distance: float = Field(..., gt=0, le=500,
                           description="Voyage distance in nautical miles")
    fuel_type: str = Field(..., description="Fuel type: HFO or Diesel")
    weather_conditions: str = Field(...,
                                   description="Weather: Calm, Moderate, or Stormy")
    engine_efficiency: float = Field(..., ge=50, le=100,
                                    description="Engine efficiency %")
    month: int = Field(..., ge=1, le=12, description="Month of voyage")

    @validator('ship_type')
    def validate_ship_type(cls, v):
        allowed = ['Oil Service Boat', 'Fishing Trawler', 'Cargo Ship']
        if v not in allowed:
            raise ValueError(f'ship_type must be one of {allowed}')
        return v

    @validator('weather_conditions')
    def validate_weather(cls, v):
        if v not in ['Calm', 'Moderate', 'Stormy']:
            raise ValueError('weather must be Calm, Moderate, or Stormy')
        return v

class FuelPredictionResponse(BaseModel):
    """Output schema."""
    fuel_consumption: float = Field(..., description="Predicted fuel (tonnes)")
    confidence_interval_90: dict = Field(...,
        description="90% prediction interval",
        example={"lower": 2500, "upper": 3500})
    model_version: str = "hybrid_v1.0"
    uncertainty_method: str = "quantile_regression"

@app.post("/predict", response_model=FuelPredictionResponse)
def predict_fuel(request: FuelPredictionRequest):
    """
    Predict fuel consumption with uncertainty intervals.

    Returns:
        Prediction with 90% confidence interval
    """
    try:
        # Prepare features (same preprocessing as training)
        features = prepare_features(request)

        # Prediction
        fuel_pred = hybrid_model.predict(features)[0]

        # Uncertainty
        _, lower, upper = uncertainty_model.predict_with_intervals(
            features, confidence=0.90
        )

        return FuelPredictionResponse(
            fuel_consumption=float(fuel_pred),
            confidence_interval_90={
                "lower": float(lower[0]),
                "upper": float(upper[0])
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    """Health endpoint for monitoring."""
    return {"status": "healthy", "model_loaded": hybrid_model is not None}

@app.get("/model_info")
def model_info():
    """Return model metadata."""
    return {
        "model_type": "Hybrid Physics-ML (XGBoost)",
        "training_samples": 1009,
        "features": ["ship_type", "distance", "fuel_type", "weather_conditions",
                    "engine_efficiency", "month"],
        "performance": {
            "test_r2": 0.82,
            "test_rmse": 450.5,
            "calibration_coverage": 0.89
        }
    }

def prepare_features(request: FuelPredictionRequest) -> np.ndarray:
    """Convert request to model input format."""
    # Apply same preprocessing as training
    # (categorical encoding, feature engineering, scaling)
    pass
```

**5.2 API Testing Suite**
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint_success():
    """Valid request returns 200 with prediction."""
    payload = {
        "ship_type": "Oil Service Boat",
        "distance": 120.0,
        "fuel_type": "HFO",
        "weather_conditions": "Moderate",
        "engine_efficiency": 85.0,
        "month": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fuel_consumption" in data
    assert data["fuel_consumption"] > 0
    assert "confidence_interval_90" in data

def test_predict_invalid_weather():
    """Invalid weather condition returns 422 validation error."""
    payload = {..., "weather_conditions": "Hurricane"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_negative_distance():
    """Negative distance fails validation."""
    payload = {..., "distance": -50}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_health_endpoint():
    """Health check returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_response_time():
    """Prediction completes in <200ms."""
    import time
    payload = {...}  # Valid payload
    start = time.time()
    response = client.post("/predict", json=payload)
    duration = (time.time() - start) * 1000  # Convert to ms
    assert response.status_code == 200
    assert duration < 200, f"Response took {duration}ms (>200ms threshold)"
```

**5.3 Dockerization**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/trained/ ./models/trained/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**5.4 Documentation & Examples**
```python
# examples/api_usage_example.py
import requests

# Example 1: Calm weather, short distance
response = requests.post('http://localhost:8000/predict', json={
    "ship_type": "Oil Service Boat",
    "distance": 50.0,
    "fuel_type": "Diesel",
    "weather_conditions": "Calm",
    "engine_efficiency": 92.0,
    "month": 3
})
print(f"Predicted fuel: {response.json()['fuel_consumption']:.2f} tonnes")
print(f"90% CI: {response.json()['confidence_interval_90']}")

# Example 2: Stormy weather, long distance
response = requests.post('http://localhost:8000/predict', json={
    "ship_type": "Fishing Trawler",
    "distance": 180.0,
    "fuel_type": "HFO",
    "weather_conditions": "Stormy",
    "engine_efficiency": 78.0,
    "month": 11
})
print(f"Predicted fuel: {response.json()['fuel_consumption']:.2f} tonnes")
```

#### Acceptance Criteria
- [ ] FastAPI application runs locally: uvicorn src.api.main:app --reload
- [ ] Interactive API docs accessible at http://localhost:8000/docs (Swagger UI)
- [ ] All endpoints tested:
  - POST /predict returns valid predictions
  - GET /health returns status
  - GET /model_info returns metadata
- [ ] Pydantic validation catches invalid inputs (wrong ship_type, negative distance)
- [ ] Response time <200ms for single prediction (measured with 10 test requests)
- [ ] Docker image builds successfully: docker build -t fuel-prediction-api .
- [ ] Docker container runs: docker run -p 8000:8000 fuel-prediction-api
- [ ] API test coverage ≥90% for src/api/main.py
- [ ] Usage examples documented in examples/api_usage_example.py

#### Tests
**API Tests (via TestClient):**
```python
def test_concurrent_requests():
    """API handles 10 concurrent requests without errors."""
    from concurrent.futures import ThreadPoolExecutor

    def make_request():
        return client.post("/predict", json=valid_payload)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in futures]

    assert all(r.status_code == 200 for r in responses)
```

#### Deliverables
1. `src/api/main.py` - FastAPI application
2. `Dockerfile` - Container configuration
3. `docker-compose.yml` - Optional: multi-container setup
4. `examples/api_usage_example.py` - Client usage examples
5. `tests/test_api.py` - API test suite
6. `README.md` updated with API deployment instructions

#### Time Breakdown
- Day 1: FastAPI implementation + pydantic schemas (6 hours)
- Day 2 (half): Dockerization + testing + documentation (6 hours)

#### Risks & Mitigation
**Risk:** Model serialization issues (joblib incompatibility across Python versions)
**Mitigation:** Pin exact library versions in requirements.txt. Test load/save in Docker environment.

**Risk:** API response time >200ms due to preprocessing overhead
**Mitigation:** Cache preprocessors (scalers, encoders) at startup. Pre-compute feature mappings.

---

## 🎯 Critical Path Analysis

### Sequential Dependencies
```
MVP-1 (Data Foundation)
    ↓
MVP-2 (Baseline ML Models)
    ↓
MVP-3 (Hybrid Model) ← Requires physics baseline AND ML models
    ↓
MVP-4 (Uncertainty) ← Can start after MVP-2 if needed

```

### Parallelization Opportunities
1. **After MVP-2 completes:**
   - Start MVP-3 (Hybrid Model) - Critical path
   - Start MVP-4 (Uncertainty) in parallel if resources available

2. **Documentation tasks** can run continuously:
   - Write README sections as each MVP completes
   - Generate visualizations incrementally
   - Update model comparison table after each model

### Timeline Summary
| MVP | Duration | Dependencies | Start Day | End Day |
|-----|----------|--------------|-----------|---------|
| MVP-1 | 2 days | None | Day 1 | Day 2 |
| MVP-2 | 3 days | MVP-1 | Day 3 | Day 5 |
| MVP-3 | 2.5 days | MVP-1, MVP-2 | Day 6 | Day 8.5 |
| MVP-4 | 2 days | MVP-2 | Day 7 | Day 9 |
| Buffer | 1 day | All | Day 11 | Day 12 |

**Total:** 12 developer-days (leaves 2-day buffer for 2-week target)

---

## 🚨 Risk Register & Mitigation

### High-Impact Risks

#### Risk 1: Dataset Quality Issues
**Probability:** Medium (40%)
**Impact:** High - Could invalidate entire project
**Description:** ship_fuel_efficiency.csv has >20% missing data in critical features, or outliers dominate distributions.

**Mitigation:**
1. **Immediate (MVP-1):** Comprehensive EDA with outlier detection
2. **Contingency:** Use navalplantmaintenance.csv as primary if ship_fuel_efficiency unusable
3. **Backup:** Synthetic data generation using physics model + noise (2-hour task)
4. **Buffer:** Allocated +0.5 day in MVP-1 for robust preprocessing

---

#### Risk 2: Hybrid Model Underperforms Pure ML
**Probability:** Medium-High (50%)
**Impact:** Medium - Reduces project novelty but not fatal
**Description:** Physics model too simplistic or introduces negative transfer (hybrid worse than pure ML).

**Mitigation:**
1. **Validation Gate:** Physics baseline must achieve R² > 0.5 before proceeding to hybrid
2. **Dual Approach:** Implement BOTH residual correction AND feature augmentation
3. **Pivot Plan:** If both fail, reframe as "physics-informed feature engineering" project
   - Highlight engineered features (fuel_rate, distance_x_weather) derived from domain knowledge
   - Still demonstrates ML + domain expertise integration
4. **Buffer:** 1-hour pivot time built into MVP-3 timeline

---

#### Risk 3: Uncertainty Intervals Poorly Calibrated
**Probability:** Medium (35%)
**Impact:** Medium - Core deliverable compromised
**Description:** 90% CI contains only 60% of points (underconfident) or 99% (overconfident), rendering intervals useless.

**Mitigation:**
1. **Multi-Method:** Test BOTH quantile regression AND bootstrapping
2. **Tuning:** Use validation set to tune quantile_alpha if coverage off by >10%
3. **Adaptation:** If coverage <80%, report 80% CI instead of 90% (adjust expectations)
4. **Post-Processing:** Implement isotonic regression for calibration adjustment
5. **Buffer:** +0.5 day in MVP-4 for calibration tuning

---

### Medium-Impact Risks

#### Risk 4: Neural Network Training Instability
**Probability:** Medium (40%)
**Impact:** Low-Medium - Can substitute with ensemble methods
**Mitigation:**
- Use batch normalization + dropout for stability
- Early stopping to prevent overfitting
- Fallback: Replace NN with Random Forest or Extra Trees (proven in reference project)

---

#### Risk 5: API Response Time Exceeds 200ms
**Probability:** Low (20%)
**Impact:** Low - Bonus MVP, not critical path
**Mitigation:**
- Cache preprocessors (scalers, encoders) at startup
- Use lightweight model for API (XGBoost faster than NN)
- If still slow: Relax requirement to 500ms (still acceptable for batch predictions)

---

#### Risk 6: Time Overrun Due to Scope Creep
**Probability:** Medium (30%)
**Impact:** High - Could miss deadline
**Mitigation:**
1. **Strict Scope:** MVP-5 (API) is OPTIONAL - skip if behind schedule
2. **Prioritization:** Focus on MVP-1 to MVP-4 (core ML deliverables)
3. **Daily Checkpoints:** Review progress against timeline each day
4. **Cut Criteria:**
   - If Day 8 and MVP-3 not complete → Skip MVP-5
   - If Day 10 and MVP-4 not complete → Deliver partial uncertainty (quantile only)

---

## 📦 Final Deliverables Checklist

### Code Artifacts
- [ ] `src/` - Python modules (8-10 files)
  - [ ] data_preprocessing.py
  - [ ] feature_engineering.py
  - [ ] models/linear_model.py
  - [ ] models/xgboost_model.py
  - [ ] models/neural_network.py (optional)
  - [ ] models/physics_baseline.py
  - [ ] models/hybrid_model.py
  - [ ] models/uncertainty.py
  - [ ] evaluation.py
  - [ ] calibration.py
  - [ ] interpretability.py
  - [ ] api/main.py (if MVP-5 completed)
- [ ] `tests/` - Test suite (≥85% coverage for core modules)
- [ ] `notebooks/` - 4-5 Jupyter notebooks:
  - [ ] 01_eda.ipynb
  - [ ] 02_baseline_models.ipynb
  - [ ] 03_hybrid_modeling.ipynb
  - [ ] 04_uncertainty_quantification.ipynb
  - [ ] 05_api_examples.ipynb (optional)
- [ ] `models/trained/` - Serialized models (5-8 .pkl/.pth files)
- [ ] `outputs/` - Visualizations (20-25 plots organized by MVP)
- [ ] `data/processed/` - Train/val/test splits
- [ ] `requirements.txt` - Dependency list
- [ ] `Dockerfile` (if MVP-5 completed)

### Documentation
- [ ] `README.md` - Project overview with:
  - [ ] Installation instructions (one-command setup)
  - [ ] Dataset description
  - [ ] Model architecture overview
  - [ ] Usage examples (training, prediction, API if applicable)
  - [ ] Results summary (table of metrics)
- [ ] `data/README.md` - Data dictionary with feature descriptions
- [ ] `MODEL_CARD.md` - Model documentation:
  - [ ] Training data characteristics
  - [ ] Performance metrics (RMSE, R², PICP)
  - [ ] Intended use cases (voyage planning, fuel budgeting)
  - [ ] Known limitations (e.g., not validated for extreme weather)
  - [ ] Ethical considerations (environmental impact estimation)

### Visualizations (Minimum Required)
1. **EDA (5 plots):** Distribution, correlation, weather impact, distance-fuel, route comparison
2. **Baseline Models (3 plots):** Predicted vs. actual (for best model), feature importance, residuals
3. **Hybrid Model (3 plots):** Hybrid vs. pure ML comparison, SHAP summary, physics vs. actual
4. **Uncertainty (4 plots):** Calibration, interval width distribution, coverage by weather, sample predictions
5. **Model Comparison (1 plot):** Bar chart comparing RMSE/R² across all models

**Total:** 16 core visualizations (exceeds minimum 6 requirement)

---

## 🎓 Skills Demonstrated (Mapped to Job Requirements)

### Machine Learning for Predictive Modeling ⭐ PREFERRED SKILL
- [x] Model comparison (Linear, XGBoost, Neural Network)
- [x] Hyperparameter tuning (RandomizedSearchCV, early stopping)
- [x] Feature engineering (physics-based, interactions, encoding)
- [x] Ensemble methods (hybrid modeling, bootstrapping)
- [x] Model interpretability (SHAP analysis)

### Data Analysis
- [x] Exploratory data analysis with domain-specific insights
- [x] Correlation analysis and multicollinearity detection
- [x] Outlier detection and handling strategies
- [x] Data quality assessment and documentation

### Model Validation
- [x] Train/validation/test split with stratification
- [x] Cross-validation for hyperparameter selection
- [x] Uncertainty quantification (quantile regression, bootstrapping)
- [x] Calibration analysis (coverage, sharpness metrics)
- [x] Residual analysis and error diagnostics

### Bonus Skills
- [x] API development (FastAPI, pydantic validation)
- [x] Software engineering (modular architecture, testing, type hints)
- [x] Physics-informed ML (hybrid modeling innovation)
- [x] Production deployment readiness (Docker, monitoring)

---

## 📊 Success Metrics - Final Targets

### Model Performance Targets
| Metric | Baseline ML | Hybrid Model | Uncertainty |
|--------|-------------|--------------|-------------|
| **R² (test)** | >0.75 | >0.80 | N/A |
| **RMSE (test)** | <500 | <460 | N/A |
| **MAPE (test)** | <12% | <10% | N/A |
| **PICP (90% CI)** | N/A | N/A | 0.85-0.92 |
| **MPIW** | N/A | N/A | <30% of mean |

### Code Quality Targets
| Metric | Target |
|--------|--------|
| **Test Coverage** | ≥85% (core), ≥70% (overall) |
| **Pylint Score** | ≥8.0 |
| **Type Hint Coverage** | 100% (public functions) |
| **Docstring Coverage** | 100% (public APIs) |

### Delivery Targets
| Deliverable | Target |
|-------------|--------|
| **Visualizations** | ≥16 publication-quality plots |
| **API Response Time** | <200ms (if MVP-5 completed) |
| **Setup Time** | <5 minutes (pip install + data download) |
| **Total Timeline** | ≤12 days (2-day buffer) |

---

## 🚀 Next Steps After WBS Approval

1. **Immediate (Day 1 Morning):**
   - [ ] Review and approve this WBS
   - [ ] Set up project structure (create folders, init files)
   - [ ] Install dependencies and verify environment
   - [ ] Load datasets and run initial data profiling

2. **Daily Standup Questions:**
   - What MVP/task was completed yesterday?
   - What is today's goal (specific acceptance criterion)?
   - Any blockers or risks materialized?
   - On track for timeline or need to adjust?

3. **Quality Gates (Must-Pass Before Next MVP):**
   - **After MVP-1:** Data quality validated, splits created, EDA complete
   - **After MVP-2:** At least one model achieves R² >0.75, comparison table generated
   - **After MVP-3:** Hybrid model outperforms OR pivot documented, SHAP analysis complete
   - **After MVP-4:** Calibration coverage ≥85%, uncertainty visualizations done

4. **Go/No-Go Decision Points:**
   - **End of Day 5 (after MVP-2):** Assess if hybrid modeling feasible based on baseline results
   - **End of Day 8 (after MVP-3):** Decide if time allows MVP-5 (API) or focus on polish
   - **End of Day 10:** Final decision on deliverable scope (all MVPs or skip MVP-5)

---

## 📚 References & Resources

### Datasets
1. **ship_fuel_efficiency.csv** (1,441 rows)
   - Nigerian maritime routes with operational data
   - Features: ship_type, route, distance, fuel, weather, efficiency

2. **navalplantmaintenance.csv** (11,934 rows)
   - UCI Propulsion Plant Data Set
   - Sensor readings from naval systems

### Reference Implementation
3. **Data-driven-Ship-Fuel-Efficiency-Modeling/** folder
   - Published IAMU research project
   - Models: Extra Trees, Gradient Boosting, XGBoost
   - Approach: Data fusion (voyage reports + AIS + meteorological data)
   - Papers: Li et al. (2022), Du et al. (2022a, 2022b)

### Technical Stack
- **Core ML:** scikit-learn, XGBoost, PyTorch (optional)
- **Data:** pandas, numpy, matplotlib, seaborn
- **Uncertainty:** SHAP, scipy (quantile regression)
- **API:** FastAPI, pydantic, uvicorn
- **Testing:** pytest, pytest-cov
- **Deployment:** Docker, docker-compose

---

## ✅ WBS Approval Checklist

Before proceeding to implementation, confirm:

- [ ] **Scope Clarity:** All 5 MVPs have clear goals, components, and acceptance criteria
- [ ] **Timeline Realism:** 12-day timeline (with 2-day buffer) is achievable for 2-week constraint
- [ ] **Risk Coverage:** Major risks identified with concrete mitigation strategies
- [ ] **Technical Feasibility:** Datasets are suitable, reference code provides validation
- [ ] **Success Metrics:** Quantitative targets are measurable and realistic
- [ ] **Deliverable Completeness:** Final artifacts demonstrate required skills (ML, validation, uncertainty)
- [ ] **Critical Path:** Sequential dependencies mapped, parallelization opportunities identified
- [ ] **Quality Standards:** Testing, documentation, and code standards defined

---

**Status:** ✅ WBS EXECUTED (MVP-1/2/3/4 complete; MVP-5 pending)
**Next Action:** Decide on MVP-5 (API) and uncertainty calibration improvements
**Estimated Start Date:** Completed
**Estimated Completion:** MVP-1/2/3/4 complete; MVP-5 TBD

---

*This WBS was generated based on PROJECT 3: ML-Enhanced Ship Fuel Prediction specifications, leveraging existing datasets (ship_fuel_efficiency.csv, navalplantmaintenance.csv) and reference ML implementations from the Data-driven-Ship-Fuel-Efficiency-Modeling project.*
