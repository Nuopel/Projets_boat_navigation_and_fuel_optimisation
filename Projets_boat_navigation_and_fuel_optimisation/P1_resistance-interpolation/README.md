# Ship Performance Surface Interpolation

Advanced interpolation methods for predicting hydrodynamic resistance from sparse experimental data.

## Project Overview

This project compares three interpolation techniques (Kriging, Radial Basis Functions, and Splines) for ship performance prediction using the UCI Yacht Hydrodynamics dataset. The work demonstrates skills essential for maritime data analysis and performance optimization.

### Business Context

Sea trials and towing tank experiments are expensive. Engineers need to interpolate between limited measurement points to predict performance across the entire operational domain. This project evaluates which interpolation methods work best for sparse hydrodynamic data.

## Features

- **Three Interpolation Methods:**
  
  - Kriging (Gaussian Process Regression) with uncertainty quantification
  - Radial Basis Functions (RBF) for irregular sampling
  - Bivariate Splines for fast computation

- **Comprehensive Benchmarking:**
  
  - Accuracy metrics (RMSE, MAE, R²)
  - Convergence analysis (sparse to dense data)
  - Computational performance profiling
  - Robustness to experimental noise

- **Visualization Suite:**
  
  - 3D surface plots
  - Error heatmaps
  - Convergence curves
  - Uncertainty maps (Kriging)

## Installation

### Requirements

- Python 3.10 or higher
- See `requirements.txt` for dependencies

### Setup

```bash
# Clone the repository
git clone https://github.com/Nuopel/Navig_P1.git
cd Navig_P1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from src.data.loader import YachtDataLoader
from src.data.preprocessor import DataPreprocessor
from src.interpolators.rbf import RBFInterpolator

# Load and preprocess data
loader = YachtDataLoader('yacht_hydro.xls')
df = loader.load()

preprocessor = DataPreprocessor()
df_vt = preprocessor.create_vt_surface(df, aggregate_duplicates=True)
# Aggregate duplicate (V, T) points to keep the 2D surface well-posed.
X_train, X_test, y_train, y_test = preprocessor.train_test_split(df_vt)

# Train interpolator
interpolator = RBFInterpolator(kernel='thin_plate_spline')
interpolator.fit(X_train.values, y_train.values)

# Predict
predictions = interpolator.predict(X_test.values)
```

## Project Structure

```
navig_p1/
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── interpolators/      # Interpolation methods (RBF, Splines, Kriging)
│   ├── evaluation/         # Metrics and benchmarking
│   └── visualization/      # Plotting utilities
├── tests/                  # Unit and integration tests
├── notebooks/              # Analysis notebooks
├── data/                   # Dataset storage
├── results/                # Benchmark outputs
├── figures/                # Generated visualizations
├── requirements.txt
└── README.md
```

## 📚 Documentation & Results

### Comprehensive Documentation

**Index:** [`docs/INDEX.md`](docs/INDEX.md) - Complete documentation index with links to all MVPs

**MVP-1 Results:** [`docs/MVP1_RESULTS.md`](docs/MVP1_RESULTS.md)

- UCI Yacht dataset analysis (308 samples)
- Velocity/Draft proxy derivation results
- Feature correlation analysis
- Train/test split visualization

**MVP-2 Results:** [`docs/MVP2_RESULTS.md`](docs/MVP2_RESULTS.md)

- Synthetic resistance surface model
- Sampling strategy comparison (Random vs Latin Hypercube vs Structured)
- Noise injection analysis (SNR validation)
- Complete metrics suite validation

### Visualizations

**Generate figures:**

```bash
cd scripts
python generate_mvp_visualizations.py
```

**Available figures** (saved to `figures/` directory):

**MVP-1:**

- `mvp1_yacht_3d_scatter.png` - 3D scatter plot of yacht resistance data
- `mvp1_train_test_split.png` - Train/test split visualization
- `mvp1_correlation_matrix.png` - Feature correlation heatmap

**MVP-2:**

- `mvp2_synthetic_surface.png` - Synthetic resistance surface (ground truth)
- `mvp2_sampling_strategies.png` - Comparison of 3 sampling methods
- `mvp2_noise_impact.png` - Effect of different SNR levels
- `mvp2_coverage_comparison.png` - Spatial coverage analysis

All figures are publication-quality (300 DPI).

### Reproducing Results

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run tests:** `pytest tests/ -v` 
3. **Check coverage:** `pytest --cov=src tests/`  (expect: 95%)
4. **Generate visualizations:** `cd scripts && python generate_mvp_visualizations.py`
5. **View documentation:** Open `docs/INDEX.md` for complete results

## Usage

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data_loader.py
```

### Exploratory Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### Benchmarking

```bash
# Run benchmark suite (coming in MVP-4)
python -m src.evaluation.benchmarker
```

## Dataset

**Source:** UCI Machine Learning Repository - Yacht Hydrodynamics Dataset
**Samples:** 308 experimental measurements
**Features:** 6 dimensionless hydrodynamic parameters
**Target:** Residuary resistance per unit displacement

**Proxy Derivation:**

- Velocity: Derived from Froude number using `V = Fr × √(g × L)`
- Draft: Derived from beam-draught ratio using `T = Beam / (B/T)`

## Development Status

- ✅ **MVP-1:** Data foundation and infrastructure 
- ✅ **MVP-2:** Synthetic data generation & metrics
- ✅ **MVP-3:** Interpolation methods 
  - RBF Interpolator: 7 kernels, auto-epsilon, smoothing
  - Spline Interpolator: configurable degrees, grid optimization
  - Kriging Interpolator: uncertainty quantification, 3 kernels
- ✅ **MVP-4:** Benchmarking and comparative analysis (NEXT)
- ✅ **MVP-5:** Visualization and final documentation

## Key Results

Mean metrics at `n_train=100` (5 trials), unnormalized aggregated (V, T) surface.
Times are in seconds.

| Method  | RMSE   | MAE   | R²      | Train Time | Predict Time |
| ------- | ------ | ----- | ------- | ---------- | ------------ |
| Kriging | 1.16   | 0.63  | 0.993   | 0.311      | 0.00026      |
| RBF     | 4.03   | 1.47  | 0.865   | 0.092      | 0.00013      |
| Splines | 108804 | 21773 | -3.57e8 | 0.0010     | 0.00002      |

## Contributing

This is a demonstration project for an Applied Mathematics Engineer position. For questions or suggestions, please open an issue.

## License

This project is provided as-is for portfolio and educational purposes.

## References

### Interpolation Methods

- Kriging: Matheron (1963) - Principles of geostatistics
- RBF: Buhmann (2003) - Radial Basis Functions: Theory and Implementations
- Splines: De Boor (1978) - A Practical Guide to Splines

### Dataset

- UCI Yacht Hydrodynamics: Gerritsma et al. (1981) - Delft yacht series
- Available at: https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics

## Contact

For inquiries about this project, please contact via GitHub issues.
