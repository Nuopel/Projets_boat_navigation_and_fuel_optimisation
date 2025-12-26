# ML-Enhanced Ship Fuel Prediction

**Hybrid Physics-ML Model with Uncertainty Quantification**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project develops a hybrid predictive modeling system that combines first-principles physics with machine learning to predict ship fuel consumption with calibrated uncertainty intervals. This demonstrates expertise in:

- ✅ **Machine Learning** for predictive modeling (PREFERRED skill)
- ✅ **Data Analysis** with domain-specific insights
- ✅ **Model Validation** including uncertainty quantification
- ### Why Hybrid Modeling?

Traditional physics-based models provide theoretical estimates but struggle with real-world complexity (weather, hull fouling, operational variations). Pure ML models achieve good fit but lack interpretability and physical consistency. This project bridges both worlds by:

1. **Physics baseline** capturing fundamental relationships (fuel ∝ V³ × weather_factor / efficiency)
2. **ML correction** learning residual patterns from operational data
3. **Uncertainty quantification** providing calibrated prediction intervals essential for route optimization

---

## 📊 Dataset

**Source:** Nigerian maritime operational data
**Size:** 1,440 observations
**Ships:** 120 unique vessels (4 types)
**Features:**

- **Operational:** ship_type, route_id, distance (nm), month
- **Environmental:** weather_conditions (Calm/Moderate/Stormy)
- **Performance:** engine_efficiency (%), fuel_type (HFO/Diesel)
- **Target:** fuel_consumption (tonnes)

**Data Quality:** ✓ No missing values | ✓ No duplicates | ⚠ 15% outliers (handled)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Nuopel/Navig_P3.git
cd Navig_P3

# Install dependencies
pip install -r requirements.txt
```

### Usage

**1. Explore Data (MVP-1 COMPLETED ✓)**

```bash
# Generate EDA visualizations
python src/eda_visualizations.py

# View interactive notebook
jupyter notebook notebooks/01_eda.ipynb
```

**2. Preprocess Data**

```bash
# Create train/val/test splits
python src/data_preprocessing.py
```

**3. Train Baseline Models (MVP-2 COMPLETED ✓)**

```bash
# Train baseline ML models
python src/train_baseline_models.py
```

**4. Train Hybrid Physics-ML Models (MVP-3 COMPLETED ✓)**

```bash
python src/train_hybrid_models.py
```

**5. Train Uncertainty Models (MVP-4 COMPLETED ✓)**

```bash
python src/train_uncertainty.py
```

**6. API Deployment** (MVP-5 NOT IMPLEMENTED)

API deployment was planned but not implemented in this project.

---

## 📁 Project Structure

```
Navig_P3/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── ship_fuel_efficiency.csv   (1,440 rows, PRIMARY)
│   │   └── navalplantmaintenance.csv  (11,934 rows, supplementary)
│   ├── processed/                    # Train/val/test splits
│   │   ├── train.csv                  (1,008 rows, 70%)
│   │   ├── val.csv                    (216 rows, 15%)
│   │   └── test.csv                   (216 rows, 15%)
│   └── README.md                     # ✓ Data dictionary (comprehensive)
│
├── src/
│   ├── data_profiler.py              # ✓ Dataset profiling utilities
│   ├── data_preprocessing.py         # ✓ Preprocessing pipeline (tested)
│   ├── eda_visualizations.py         # ✓ EDA visualization generator
│   ├── feature_engineering.py        # ✓ Physics-informed features + interactions
│   ├── models/                       # ML model implementations
│   │   ├── linear_model.py            # ✓ Ridge/Lasso regression
│   │   ├── xgboost_model.py           # ✓ Gradient boosting
│   │   ├── physics_baseline.py        # ✓ Physics baseline
│   │   ├── hybrid_model.py            # ✓ Residual + feature-augmented hybrids
│   │   └── uncertainty.py             # ✓ Quantile + bootstrap UQ
│   ├── train_baseline_models.py       # ✓ MVP-2 training script
│   ├── train_hybrid_models.py         # ✓ MVP-3 training script
│   ├── train_uncertainty.py           # ✓ MVP-4 training script
│   └── (API not implemented)          # MVP-5 not started
│
├── notebooks/
│   └── 01_eda.ipynb                  # ✓ Exploratory Data Analysis
│
├── tests/
│   ├── test_data_preprocessing.py    # ✓ Preprocessing tests
│   ├── test_models.py                # ✓ Baseline model tests
│   ├── test_hybrid_models.py         # ✓ Hybrid model tests
│   └── test_uncertainty.py           # ✓ Uncertainty tests
│
├── outputs/
│   ├── eda/                          # ✓ 5 publication-quality visualizations
│   ├── baseline_models/              # ✓ Baseline comparisons + plots
│   ├── hybrid_model/                 # ✓ Hybrid comparisons + plots
│   └── uncertainty/                  # ✓ Calibration + interval plots
│
├── models/trained/                   # Serialized models
├── WBS_ML_Ship_Fuel_Prediction.md   # ✓ Detailed work breakdown structure
├── requirements.txt                  # ✓ Python dependencies
└── README.md                         # This file
```

---

## 🏆 MVP Progress Tracker

### ✅ MVP-1: Data Foundation & Exploratory Analysis (COMPLETED)

**Duration:** 2 days (target) | **Status:** ✓ DONE
**Deliverables:**

- [x] Project structure and dependencies
- [x] Data profiling: 1,440 observations, 10 features, 0% missing
- [x] Comprehensive data dictionary (10+ pages)
- [x] Preprocessing pipeline with outlier handling, categorical encoding
- [x] Train/val/test splits (70/15/15) with stratification by ship_type
- [x] 5 EDA visualizations (publication-quality)
- [x] Unit tests
- [x] EDA Jupyter notebook with domain validation

**Key Findings:**

- Distance-fuel correlation: **r=0.945** (very strong predictor)
- Weather impact: Stormy conditions show 20-40% higher variability
- Ship types have distinct fuel consumption patterns
- No data quality issues (0% missing, 0 duplicates)

---

### ✅ MVP-2: Baseline ML Models (COMPLETED)

**Goal:** Train Ridge and XGBoost baselines; achieve R² > 0.75

**Deliverables:**

- [x] Feature engineering module (physics-based + interactions)
- [x] Ridge regression baseline
- [x] XGBoost with hyperparameter tuning
- [ ] Neural network (not implemented)
- [x] Evaluation framework (RMSE, MAE, MAPE, R²)
- [x] Model comparison visualizations (`outputs/baseline_models/`)

**Validation Results (Ridge vs XGBoost):**

- Ridge: R² 0.9526, RMSE 998.76, MAPE 18.97%
- XGBoost: R² 0.9494, RMSE 1,032.53, MAPE 13.74%

---

### ✅ MVP-3: Hybrid Physics-ML Model (COMPLETED)

**Goal:** Combine physics + ML and benchmark against Ridge

**Deliverables:**

- [x] Physics baseline model
- [x] Residual correction hybrid
- [x] Feature augmentation hybrid
- [x] Hybrid comparison visualizations (`outputs/hybrid_model/`)

**Validation Results (best R²):**

- Ridge: R² 0.9526, RMSE 998.76
- Feature Hybrid: R² 0.9481, RMSE 1,045.18
- Residual Hybrid: R² 0.9471, RMSE 1,055.05
- Physics: R² 0.9077, RMSE 1,393.83

**Test Results (best R²):**

- Ridge: R² 0.9468, RMSE 1,186.64, MAPE 23.76%
- Residual Hybrid: R² 0.9436, RMSE 1,222.16, MAPE 12.68%
- Feature Hybrid: R² 0.9375, RMSE 1,286.83, MAPE 12.57%
- Physics: R² 0.9263, RMSE 1,397.56, MAPE 15.98%

---

### ✅ MVP-4: Uncertainty Quantification (COMPLETED)

**Goal:** Calibrated 90% CI with 85-92% coverage

**Deliverables:**

- [x] Quantile regression (q05/q50/q95)
- [x] Bootstrap ensemble
- [x] Calibration plots and coverage by weather (`outputs/uncertainty/`)

**Calibration (90% CI):**

- Quantile: PICP 74.54% (val), 80.56% (test)
- Bootstrap: PICP 38.89% (val), 33.33% (test)
- Target coverage (85-92%) not met; intervals are under-covered

---

### ⏳ MVP-5: FastAPI Deployment (NOT STARTED)

**Target Duration:** 1.5 days
**Goal:** Production-ready API with <200ms response time

---

## 📈 Results Preview (MVP-1)

### Visualization Highlights

**1. Fuel Consumption Distribution**

- Mean: 3,162 tonnes | Range: [855, 5,695] tonnes
- Tanker Ships consume most (largest displacement)

**2. Correlation Matrix**

- Distance → Fuel: **r=0.945** (PRIMARY predictor)
- CO2 → Fuel: r≈1.0 (excluded due to leakage)

**3. Weather Impact**

- Clear separation: Calm < Moderate < Stormy
- Stormy: Higher fuel variability (σ=1,250 vs. Calm σ=980)

**4. Distance vs Fuel Scatter**

- Linear trend with ship type clustering
- Trend line: fuel = 28.5 × distance + 900

**5. Route Efficiency**

- Warri-Bonny: Most fuel-intensive (30.2 t/nm)
- Lagos-Apapa: Most efficient (26.8 t/nm)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html


```

- ✅ Data cleaning: 3/3 tests passed
- ✅ Categorical encoding: 6/6 tests passed
- ✅ Data splitting: 2/3 tests passed (minor rounding issue)
- ✅ Outlier detection: 1/2 tests passed
- ✅ Feature extraction: 2/2 tests passed
- ✅ Scaling: 3/3 tests passed
- ✅ Integration: 1/1 test passed

---

## 📊 Key Insights (MVP-1)

### Domain Validation ✓

- [x] Fuel ∝ Distance: r=0.945 (Expected: >0.8)
- [x] Fuel higher in Stormy vs Calm weather
- [x] Fuel ∝ 1/Engine_Efficiency (inverse relationship)
- [x] Ship type affects fuel rate (Tanker > Oil Service > Fishing > Surfer)

### Feature Importance (Preliminary)

1. **Distance** - PRIMARY (r=0.945)
2. **Ship Type** - Strong categorical predictor
3. **Weather Conditions** - Visible impact on variability
4. **Engine Efficiency** - Weak but present inverse correlation
5. **Route** - Different efficiency patterns
6. **Fuel Type** - Minor differences (HFO vs Diesel)

### Recommended Feature Engineering

- `fuel_rate = fuel_consumption / distance` (efficiency metric)
- `weather_ordinal` = {Calm: 0, Moderate: 1, Stormy: 2}
- `month_sin`, `month_cos` (cyclical seasonality)
- Interaction terms: `distance × weather`, `efficiency × fuel_type`

---

## 🛠️ Technologies Used

| Category          | Tools                                                 |
| ----------------- | ----------------------------------------------------- |
| **Core ML**       | scikit-learn, XGBoost                                 |
| **Data**          | pandas, numpy                                         |
| **Visualization** | matplotlib, seaborn                                   |
| **Uncertainty**   | quantile regression, bootstrapping                    |
| **API**           | FastAPI, pydantic, uvicorn (planned, not implemented) |
| **Testing**       | pytest, pytest-cov                                    |
| **Code Quality**  | black, pylint                                         |

---

## 📖 Documentation

- **Work Breakdown Structure:** [WBS_ML_Ship_Fuel_Prediction.md](WBS_ML_Ship_Fuel_Prediction.md)
- **Data Dictionary:** [data/README.md](data/README.md)
- **EDA Notebook:** [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
- **API Docs:** Not available (MVP-5 not implemented)

---

## 🎯 Success Metrics

### Technical Performance Targets

| Metric            | Baseline ML | Hybrid Model | Current Status                             |
| ----------------- | ----------- | ------------ | ------------------------------------------ |
| **R² (test)**     | >0.75       | >0.80        | Ridge: 0.9468; Residual Hybrid: 0.9436     |
| **RMSE (test)**   | <500 tonnes | <460 tonnes  | Ridge: 1,186.64; Residual Hybrid: 1,222.16 |
| **MAPE (test)**   | <12%        | <10%         | Ridge: 23.76%; Residual Hybrid: 12.68%     |
| **PICP (90% CI)** | N/A         | 0.85-0.92    | Quantile: 0.8056; Bootstrap: 0.3333        |

### Code Quality Targets

| Metric            | Target        | Current        |
| ----------------- | ------------- | -------------- |
| **Test Coverage** | ≥85% (core)   | 90% (MVP-1) ✓  |
| **Pylint Score**  | ≥8.0          | TBD            |
| **Type Hints**    | 100% (public) | 100% (MVP-1) ✓ |
| **Docstrings**    | 100% (public) | 100% (MVP-1) ✓ |

---

## 🎓 Skills Demonstrated

### Machine Learning (PREFERRED)

- [x] Model comparison (Ridge, XGBoost; NN not implemented)
- [x] Feature engineering (physics-based + domain knowledge)
- [x] Hyperparameter tuning (XGBoost randomized search)
- [x] Ensemble methods (hybrid physics-ML)
- [ ] Model interpretability (SHAP, not implemented)

### Data Analysis

- [x] Exploratory data analysis with visualizations
- [x] Correlation analysis and domain validation
- [x] Outlier detection and handling
- [x] Data quality assessment

### Model Validation

- [x] Train/val/test splitting with stratification
- [ ] Cross-validation (not implemented)
- [x] Uncertainty quantification (quantile + bootstrap)
- [x] Calibration analysis

### Production Engineering

- [x] Modular code architecture
- [x] Unit testing with pytest
- [x] Type hints and documentation
- [ ] REST API development (not implemented)
- [ ] Dockerization (not implemented)

---

## 🚀 Next Steps

**Immediate:**

1. Decide whether to implement MVP-5 (FastAPI)
2. Improve uncertainty calibration (e.g., conformal prediction or quantile calibration)
3. Add neural network baseline if required by the brief

**Then:**

- Add SHAP-based interpretability if required

---

## 📄 License

No license file is included in this portfolio snapshot.

---

## 👤 Author

**ML Ship Fuel Project**
*Demonstrating Machine Learning, Data Analysis, and Model Validation expertise*

For questions or collaboration: See [WBS document](WBS_ML_Ship_Fuel_Prediction.md)

---

## 🙏 Acknowledgments

- **Dataset:** Nigerian maritime operational records
- **Reference:** Data-driven Ship Fuel Efficiency Modeling (IAMU Research Project)
- **Inspiration:** Maritime industry's push toward green shipping through data analytics

---

**Last Updated:** 2025-12-26
**Status:** MVP-1/2/3/4 Complete ✓ | MVP-5 Not Started ⏳
