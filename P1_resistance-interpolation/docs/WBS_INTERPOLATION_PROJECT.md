# 📊 Work Breakdown Structure (WBS)
# Ship Performance Surface Interpolation Project

**Project Duration:** 10-12 developer-days
**Methodology:** Incremental MVPs with continuous testing
**Quality Gates:** Each MVP must pass all tests before proceeding

---

## 🎯 Project Overview

**Goal:** Implement and compare three interpolation methods (Kriging, RBF, Splines) for predicting hydrodynamic resistance from sparse experimental data.

**Success Criteria:**
- ✅ All 3 interpolators functional with unified API
- ✅ RMSE < 0.005 on UCI test set (best method)
- ✅ Test coverage >80%
- ✅ Complete comparative analysis with visualizations
- ✅ Technical documentation with recommendations

---

## 📋 MVP Breakdown

### Critical Path: MVP-1 → MVP-2 → MVP-3 → MVP-4 → MVP-5

```
MVP-1 (Foundation) ──┐
                     ├──> MVP-3 (Interpolators) ──> MVP-4 (Benchmarking) ──> MVP-5 (Viz & Docs)
MVP-2 (Synthetic)  ──┘
```

**Parallel Opportunities:**
- MVP-1 and MVP-2 can partially overlap (synthetic data doesn't require UCI loaded)
- Within MVP-3: RBF, Splines, Kriging can be developed sequentially
- MVP-5 visualization code can start during MVP-4

---

## 🏗️ MVP-1: Data Foundation & Infrastructure

### Goal
Establish data pipeline for UCI Yacht Hydrodynamics dataset with proxy derivation (V, T) and create project scaffolding with testing infrastructure.

### Components

#### 1.1 Project Structure
```
navig_p1/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # YachtDataLoader class
│   │   ├── preprocessor.py    # DataPreprocessor class
│   │   └── synthetic.py       # (MVP-2)
│   ├── interpolators/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseInterpolator abstract class
│   │   ├── rbf.py             # (MVP-3)
│   │   ├── spline.py          # (MVP-3)
│   │   └── kriging.py         # (MVP-3)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # RMSE, MAE, R² calculators
│   │   └── benchmarker.py     # (MVP-4)
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py         # (MVP-5)
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── ...
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

#### 1.2 YachtDataLoader Class
**Responsibility:** Load UCI dataset from Excel, handle column naming, basic validation

**Key Methods:**
```python
class YachtDataLoader:
    def __init__(self, filepath: str)
    def load(self) -> pd.DataFrame
    def get_feature_names(self) -> List[str]
    def get_target_name(self) -> str
    def validate(self) -> bool  # Check 308 samples, 7 columns
```

#### 1.3 DataPreprocessor Class
**Responsibility:** Derive (V, T) proxies from Froude number and B/T ratio

**Key Methods:**
```python
class DataPreprocessor:
    def __init__(self, reference_length: float = 10.0, reference_beam: float = 3.0)
    def derive_velocity(self, froude: np.ndarray) -> np.ndarray
    def derive_draft(self, beam_draft_ratio: np.ndarray) -> np.ndarray
    def create_vt_surface(self, df: pd.DataFrame) -> pd.DataFrame
    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[...]
```

**Derivation Formulas:**
- **Velocity:** V = Froude × √(g × L) with g=9.81 m/s², L=10m → V in knots conversion
- **Draft:** T = Beam_ref / (B/T ratio) with Beam_ref = 3m

#### 1.4 Exploratory Analysis Notebook
**Content:**
- Load UCI dataset and display first rows
- Statistical summary (min, max, mean, std for all features)
- Distribution plots for Froude, B/T ratio, Resistance
- Derived (V, T) domain visualization
- Correlation matrix heatmap
- 3D scatter plot of R(V, T) on training data

### Acceptance Criteria
- ✅ UCI dataset loads without errors (308 samples, 7 columns)
- ✅ Proxy derivation produces sensible ranges:
  - V ∈ [10, 25] knots (approximate)
  - T ∈ [6, 10] meters (approximate)
- ✅ Train/test split: 246 train, 62 test (80/20 with seed=42)
- ✅ EDA notebook runs end-to-end without errors
- ✅ Project structure created with all __init__.py files
- ✅ requirements.txt includes: pandas, numpy, scikit-learn, scipy, matplotlib, pytest, openpyxl

### Testing Requirements

**tests/test_data_loader.py** (5 tests)
1. `test_load_yacht_data_shape()` - Verify 308×7 dimensions
2. `test_load_yacht_data_columns()` - Check expected column names
3. `test_load_yacht_data_no_nulls()` - Ensure no missing values
4. `test_load_yacht_data_target_positive()` - Resistance > 0
5. `test_load_invalid_filepath()` - Handle FileNotFoundError gracefully

**tests/test_preprocessor.py** (6 tests)
1. `test_derive_velocity_froude_range()` - V increases with Froude
2. `test_derive_draft_ratio_range()` - T decreases with B/T ratio
3. `test_create_vt_surface_columns()` - Output has V, T, R columns
4. `test_train_test_split_sizes()` - 80/20 split correct
5. `test_train_test_split_reproducibility()` - Same seed → same split
6. `test_train_test_no_overlap()` - No indices in both sets

### Dependencies
- None (foundation MVP)

### Risk Mitigation
- **Risk:** Excel format issues with pandas
  **Mitigation:** Use openpyxl engine explicitly, validate with small test file first
- **Risk:** Proxy derivation produces unrealistic values
  **Mitigation:** Add validation ranges, log warnings if outside expected bounds

### Estimated Time: 2 days

**Breakdown:**
- Project structure setup: 0.5 day
- YachtDataLoader + tests: 0.5 day
- DataPreprocessor + tests: 0.75 day
- EDA notebook: 0.25 day

---

## 🎲 MVP-2: Synthetic Data Generation & Validation Framework

### Goal
Create controlled synthetic resistance surface with known ground truth for objective validation of interpolation methods before applying to real UCI data.

### Components

#### 2.1 SyntheticSurfaceGenerator Class
**Responsibility:** Generate physically realistic resistance surface with sparse sampling and noise injection

**Key Methods:**
```python
class SyntheticSurfaceGenerator:
    def __init__(self, v_range: Tuple[float, float] = (10, 25),
                 t_range: Tuple[float, float] = (6, 10),
                 noise_std: float = 0.01)

    def resistance_function(self, V: np.ndarray, T: np.ndarray) -> np.ndarray
        """Generate true resistance surface using physically inspired formula.

        Example: R = a*V^2 + b*T^(-1) + c*V*T + d + gaussian_noise
        Coefficients chosen to mimic yacht resistance behavior.
        """

    def create_dense_grid(self, grid_size: int = 100) -> Dict[str, np.ndarray]
        """Ground truth on fine grid for error computation."""

    def sample_sparse(self, n_samples: int, strategy: str = 'random') -> pd.DataFrame
        """Sample training points: random, latin_hypercube, or structured."""

    def add_noise(self, R: np.ndarray, snr_db: float = 20) -> np.ndarray
        """Add Gaussian noise with specified Signal-to-Noise Ratio."""
```

**Surface Design:**
```
R(V, T) = 0.05*V² - 2.0*T + 0.01*V*T + 15.0 + ε
where ε ~ N(0, σ²)

Physical Interpretation:
- V² term: Velocity-dependent resistance (dominant at high speed)
- T⁻¹ term: Draft influences wetted surface
- V*T term: Interaction effect
- Constant: Baseline resistance
```

#### 2.2 MetricsCalculator Class
**Responsibility:** Compute standardized error metrics between predictions and ground truth

**Key Methods:**
```python
class MetricsCalculator:
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float

    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]
        """Return dictionary with all metrics."""
```

#### 2.3 Validation Notebook
**Content:**
- Generate synthetic surface and visualize ground truth 3D plot
- Sample sparse training sets (10, 30, 50 points)
- Add noise at different SNR levels (10dB, 20dB, noise-free)
- Visualize sampling strategies comparison
- Save synthetic datasets for use in MVP-3/4

### Acceptance Criteria
- ✅ Synthetic surface is smooth and physically plausible (no discontinuities)
- ✅ Ground truth grid (100×100) computed in <1 second
- ✅ Sparse sampling produces expected number of unique points
- ✅ Latin Hypercube sampling has better space-filling than random
- ✅ Noise addition produces specified SNR (within 0.5 dB tolerance)
- ✅ MetricsCalculator returns consistent results with sklearn equivalents

### Testing Requirements

**tests/test_synthetic.py** (7 tests)
1. `test_resistance_function_shape()` - Output matches input shape
2. `test_resistance_function_velocity_monotonic()` - R increases with V (T fixed)
3. `test_dense_grid_size()` - 100×100 grid has 10,000 points
4. `test_sparse_sampling_count()` - Requested n_samples returned
5. `test_sparse_sampling_unique()` - No duplicate points
6. `test_add_noise_snr()` - Achieved SNR matches request ±0.5dB
7. `test_synthetic_deterministic()` - Same seed → same surface

**tests/test_metrics.py** (6 tests)
1. `test_rmse_zero_error()` - RMSE=0 when y_true=y_pred
2. `test_mae_known_values()` - MAE=[1,2,3] vs [2,3,4] = 1.0
3. `test_r2_perfect_prediction()` - R²=1.0 when perfect
4. `test_max_error_identifies_worst()` - Correct worst-case value
5. `test_metrics_nan_handling()` - Raise error on NaN inputs
6. `test_compute_all_metrics_keys()` - Returns rmse, mae, r2, max_error

### Dependencies
- MVP-1 (for project structure and MetricsCalculator location)

### Risk Mitigation
- **Risk:** Synthetic surface too simple or unrealistic
  **Mitigation:** Include nonlinear terms and validate visually, compare to UCI data characteristics
- **Risk:** Sampling strategies produce poor coverage
  **Mitigation:** Implement multiple strategies, visualize coverage, use scipy.stats.qmc for LHS

### Estimated Time: 1.5 days

**Breakdown:**
- SyntheticSurfaceGenerator + tests: 0.75 day
- MetricsCalculator + tests: 0.5 day
- Validation notebook: 0.25 day

---

## 🧮 MVP-3: Interpolation Methods Implementation

### Goal
Implement three interpolation methods with unified API, each validated on synthetic data to ensure correctness before benchmarking.

### Architecture: Unified API

#### 3.1 BaseInterpolator (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseInterpolator(ABC):
    """Abstract base class for all interpolation methods.

    Enforces consistent API for training, prediction, and metadata.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.train_time = None
        self.predict_time = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseInterpolator':
        """Train interpolator on (X, y) pairs.

        Args:
            X: (n_samples, 2) array of (V, T) coordinates
            y: (n_samples,) array of resistance values

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict resistance at query points.

        Args:
            X: (n_queries, 2) array of (V, T) coordinates

        Returns:
            (n_queries,) array of predicted resistance
        """
        pass

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
        """Convenience method for train + predict."""
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def get_metadata(self) -> dict:
        """Return training/prediction times and hyperparameters."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'train_time': self.train_time,
            'predict_time': self.predict_time
        }
```

### Component Breakdown

#### 3.2 RBF Interpolator (Implement FIRST)
**Priority:** Highest - Most straightforward, good baseline

**Implementation:**
```python
from scipy.interpolate import RBFInterpolator

class RBFInterpolator(BaseInterpolator):
    """Radial Basis Function interpolation.

    Advantages:
    - Excellent for irregularly spaced points
    - No grid requirement
    - Fast prediction

    Hyperparameters:
    - kernel: 'thin_plate_spline' (default), 'multiquadric', 'gaussian'
    - smoothing: regularization parameter (default=0 for exact interpolation)
    """

    def __init__(self, kernel: str = 'thin_plate_spline', smoothing: float = 0.0):
        super().__init__(name='RBF')
        self.kernel = kernel
        self.smoothing = smoothing
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RBFInterpolator':
        import time
        start = time.time()

        self._model = RBFInterpolator(
            X, y,
            kernel=self.kernel,
            smoothing=self.smoothing
        )

        self.train_time = time.time() - start
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict()")

        import time
        start = time.time()
        predictions = self._model(X)
        self.predict_time = time.time() - start

        return predictions
```

**Testing:** tests/test_rbf_interpolator.py (8 tests)
1. `test_rbf_fit_sets_flag()` - is_fitted becomes True
2. `test_rbf_predict_before_fit_raises()` - Error when not fitted
3. `test_rbf_perfect_interpolation_noiseless()` - Exact fit on training points (smoothing=0)
4. `test_rbf_shape_consistency()` - Output shape matches input
5. `test_rbf_different_kernels()` - thin_plate, multiquadric, gaussian all work
6. `test_rbf_smoothing_reduces_overfitting()` - smoothing>0 has higher train error
7. `test_rbf_timing_metadata()` - train_time and predict_time recorded
8. `test_rbf_synthetic_validation()` - RMSE < 0.05 on 50-point synthetic

**Estimated Time:** 1 day

---

#### 3.3 Spline Interpolator (Implement SECOND)
**Priority:** High - Fast, well-established

**Implementation:**
```python
from scipy.interpolate import SmoothBivariateSpline

class SplineInterpolator(BaseInterpolator):
    """Bivariate spline interpolation.

    Advantages:
    - Very fast computation
    - Smoothing parameter controls overfitting
    - Mathematically well-founded

    Limitations:
    - Prefers regular grids (but works with irregular)

    Hyperparameters:
    - smoothing: s parameter (default=0 for interpolation, >0 for smoothing)
    - kx, ky: spline degrees in x and y (default=3 for cubic)
    """

    def __init__(self, smoothing: float = 0.0, kx: int = 3, ky: int = 3):
        super().__init__(name='Spline')
        self.smoothing = smoothing
        self.kx = kx
        self.ky = ky
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SplineInterpolator':
        import time
        start = time.time()

        # SmoothBivariateSpline expects separate x, y arrays
        x_coords = X[:, 0]
        y_coords = X[:, 1]

        self._model = SmoothBivariateSpline(
            x_coords, y_coords, y,
            s=self.smoothing,
            kx=self.kx, ky=self.ky
        )

        self.train_time = time.time() - start
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict()")

        import time
        start = time.time()

        # Predict point-by-point (ev method)
        x_query = X[:, 0]
        y_query = X[:, 1]
        predictions = self._model.ev(x_query, y_query)

        self.predict_time = time.time() - start

        return predictions
```

**Testing:** tests/test_spline_interpolator.py (8 tests)
1. `test_spline_fit_sets_flag()`
2. `test_spline_predict_before_fit_raises()`
3. `test_spline_perfect_interpolation_noiseless()`
4. `test_spline_shape_consistency()`
5. `test_spline_smoothing_effect()` - s>0 produces smoother surface
6. `test_spline_speed_benchmark()` - Faster than RBF on same data
7. `test_spline_extrapolation_warning()` - Document behavior outside domain
8. `test_spline_synthetic_validation()` - RMSE < 0.05 on 50-point synthetic

**Estimated Time:** 1 day

---

#### 3.4 Kriging Interpolator (Implement THIRD)
**Priority:** Medium-High - Most complex, but provides uncertainty quantification

**Implementation:**
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

class KrigingInterpolator(BaseInterpolator):
    """Gaussian Process Regression / Kriging.

    Advantages:
    - Provides uncertainty estimates (standard deviation)
    - Excellent for very sparse data
    - Statistically rigorous

    Limitations:
    - O(n³) complexity - slow for n>500
    - Requires kernel selection

    Hyperparameters:
    - kernel: RBF (default), Matern, or custom
    - alpha: noise regularization (default=1e-6)
    - n_restarts_optimizer: kernel hyperparameter optimization attempts
    """

    def __init__(self, kernel=None, alpha: float = 1e-6, n_restarts_optimizer: int = 5):
        super().__init__(name='Kriging')

        if kernel is None:
            # Default: constant * RBF kernel with length scale optimization
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KrigingInterpolator':
        import time
        start = time.time()

        self._model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True  # Important for numerical stability
        )

        self._model.fit(X, y)

        self.train_time = time.time() - start
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict with optional uncertainty quantification.

        Args:
            X: Query points
            return_std: If True, return (predictions, std_deviations)

        Returns:
            predictions or (predictions, std) tuple
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict()")

        import time
        start = time.time()

        if return_std:
            predictions, std = self._model.predict(X, return_std=True)
            self.predict_time = time.time() - start
            return predictions, std
        else:
            predictions = self._model.predict(X, return_std=False)
            self.predict_time = time.time() - start
            return predictions

    def get_uncertainty_map(self, X_grid: np.ndarray) -> np.ndarray:
        """Get prediction uncertainty across grid (for visualization)."""
        _, std = self.predict(X_grid, return_std=True)
        return std
```

**Testing:** tests/test_kriging_interpolator.py (10 tests)
1. `test_kriging_fit_sets_flag()`
2. `test_kriging_predict_before_fit_raises()`
3. `test_kriging_predict_without_std()` - Standard prediction works
4. `test_kriging_predict_with_std()` - Returns (mean, std) tuple
5. `test_kriging_uncertainty_at_training_points()` - std ≈ 0 at training points
6. `test_kriging_uncertainty_increases_away_from_data()` - Higher std far from training
7. `test_kriging_shape_consistency()`
8. `test_kriging_different_kernels()` - RBF, Matern work
9. `test_kriging_synthetic_validation()` - RMSE < 0.05 on 50-point synthetic
10. `test_kriging_computational_cost()` - Warn if train_time > 10s for 308 points

**Estimated Time:** 1.5 days

---

### MVP-3 Integration Testing

**tests/test_interpolator_integration.py** (5 tests)
1. `test_all_interpolators_unified_api()` - All implement fit/predict
2. `test_interpolators_on_same_dataset()` - Compare results on synthetic data
3. `test_fit_predict_chain()` - fit().predict() workflow
4. `test_metadata_consistency()` - All return get_metadata() dict
5. `test_edge_cases()` - Single point, collinear points handling

### Acceptance Criteria
- ✅ All three interpolators pass their individual test suites
- ✅ Unified API: fit(), predict(), get_metadata() for all
- ✅ RBF achieves RMSE < 0.05 on 50-point synthetic (smoothing=0)
- ✅ Splines is fastest (train_time < 0.1s for 308 points)
- ✅ Kriging returns uncertainty estimates
- ✅ Test coverage >85% for src/interpolators/

### Dependencies
- MVP-1 (project structure)
- MVP-2 (synthetic data for validation tests)

### Risk Mitigation
- **Risk:** Numerical instability with poorly conditioned data
  **Mitigation:** Normalize inputs to [0,1]² range, add regularization parameters
- **Risk:** Kriging too slow on full UCI dataset
  **Mitigation:** Test on subsamples first, document complexity, consider sparse GP approximations
- **Risk:** Splines fail on very irregular sampling
  **Mitigation:** Validate on Latin Hypercube samples, document grid preference

### Estimated Time: 3.5 days

**Breakdown:**
- BaseInterpolator design: 0.25 day
- RBF + tests: 1 day
- Splines + tests: 1 day
- Kriging + tests: 1.5 days
- Integration tests: 0.25 day
- Bug fixes + refinement: 0.5 day (buffer)

---

## 📊 MVP-4: Benchmarking & Comparative Analysis

### Goal
Systematically compare the three interpolation methods across multiple dimensions: accuracy, speed, robustness to noise, convergence behavior, and extrapolation.

### Components

#### 4.1 Benchmarker Class
**Responsibility:** Run standardized benchmarks across multiple interpolators

```python
class Benchmarker:
    """Orchestrate comparative benchmarks for interpolation methods.

    Supports:
    - Accuracy benchmarking on test sets
    - Convergence analysis (varying training set size)
    - Noise robustness testing
    - Computational performance profiling
    """

    def __init__(self, interpolators: List[BaseInterpolator],
                 metrics_calculator: MetricsCalculator):
        self.interpolators = interpolators
        self.metrics = metrics_calculator
        self.results = {}

    def run_accuracy_benchmark(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Compare prediction accuracy on fixed train/test split.

        Returns DataFrame with columns:
        ['method', 'rmse', 'mae', 'r2', 'max_error', 'train_time', 'predict_time']
        """
        pass

    def run_convergence_analysis(self, X_full: np.ndarray, y_full: np.ndarray,
                                  sample_sizes: List[int],
                                  n_trials: int = 10) -> pd.DataFrame:
        """Test how error decreases with more training data.

        For each sample size, repeat n_trials with different random samples.
        Returns DataFrame with columns:
        ['method', 'n_samples', 'trial', 'rmse', 'mae', 'r2']
        """
        pass

    def run_noise_robustness_test(self, X: np.ndarray, y_clean: np.ndarray,
                                   snr_levels: List[float]) -> pd.DataFrame:
        """Compare performance under different noise levels.

        Returns DataFrame with columns:
        ['method', 'snr_db', 'rmse', 'mae', 'r2']
        """
        pass

    def run_extrapolation_test(self, X_train_interior: np.ndarray, y_train: np.ndarray,
                               X_test_boundary: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Test prediction quality outside training domain.

        Train on interior points, test on boundary.
        """
        pass

    def save_results(self, output_dir: str):
        """Save all benchmark results as CSV files."""
        pass
```

#### 4.2 Benchmark Experiments

**Experiment 1: Synthetic Data Validation**
- Ground truth known
- Sample sizes: [10, 20, 30, 50, 100]
- Noise levels: [∞, 20dB, 10dB] (clean, moderate, high noise)
- Sampling: Latin Hypercube
- **Expected Outcome:** All methods converge, Kriging best at low sample sizes

**Experiment 2: UCI Dataset Comparison**
- Use full 308-point dataset
- 80/20 train/test split
- Stratified sampling if possible (ensure coverage of V-T space)
- **Expected Outcome:** RMSE < 0.005 for best method

**Experiment 3: Sparse Sampling Robustness (UCI)**
- Subsample UCI dataset: [20, 50, 100, 200, 308] points
- 5 random trials per sample size
- **Expected Outcome:** Kriging stable at low counts, Splines require more data

**Experiment 4: Computational Scaling**
- Training set sizes: [10, 30, 100, 308]
- Measure fit() and predict() times
- Predict on 100×100 grid (10,000 points)
- **Expected Outcome:** Kriging O(n³), Splines fastest, RBF O(n²)

**Experiment 5: Extrapolation Quality**
- Train on V∈[12, 23], T∈[6.5, 9.5] (interior)
- Test on boundary points
- **Expected Outcome:** All methods degrade, Kriging shows high uncertainty

#### 4.3 Analysis Notebooks

**notebooks/02_synthetic_benchmark.ipynb**
- Run Experiments 1 (synthetic validation)
- Generate convergence plots
- Compare noise robustness

**notebooks/03_uci_benchmark.ipynb**
- Run Experiments 2, 3 (UCI accuracy and convergence)
- Generate comparison tables
- Statistical significance tests (paired t-test on RMSE)

**notebooks/04_computational_analysis.ipynb**
- Run Experiment 4 (scaling)
- Timing plots: train_time vs n_samples, predict_time vs n_queries
- Memory profiling

### Acceptance Criteria
- ✅ All 5 experiments run without errors
- ✅ Benchmarker produces consistent results across runs (with fixed seeds)
- ✅ At least one method achieves RMSE < 0.005 on UCI test set
- ✅ Kriging provides uncertainty estimates consistent with actual errors
- ✅ Convergence plots show clear error reduction with more data
- ✅ Computational scaling matches theoretical complexity
- ✅ Results saved as CSV files in `results/` directory

### Testing Requirements

**tests/test_benchmarker.py** (6 tests)
1. `test_benchmarker_initialization()` - Accepts list of interpolators
2. `test_accuracy_benchmark_output_format()` - Correct DataFrame columns
3. `test_convergence_analysis_trials()` - n_trials executed per sample size
4. `test_noise_robustness_snr_levels()` - All SNR levels tested
5. `test_save_results_creates_files()` - CSV files written
6. `test_benchmarker_reproducibility()` - Same seed → same results

### Dependencies
- MVP-1 (data loading)
- MVP-2 (synthetic data, metrics)
- MVP-3 (all three interpolators)

### Risk Mitigation
- **Risk:** Benchmarks take too long to run
  **Mitigation:** Start with small sample sizes, use multiprocessing for trials, cache results
- **Risk:** Results not reproducible
  **Mitigation:** Fix all random seeds, document versions in requirements.txt
- **Risk:** Statistical significance unclear
  **Mitigation:** Use 10+ trials for convergence analysis, include confidence intervals

### Estimated Time: 2 days

**Breakdown:**
- Benchmarker class + tests: 0.75 day
- Experiment 1 & 2 (synthetic + UCI): 0.5 day
- Experiment 3 & 4 (convergence + scaling): 0.5 day
- Experiment 5 (extrapolation) + analysis: 0.25 day

---

## 📈 MVP-5: Visualization & Documentation

### Goal
Create publication-quality visualizations and comprehensive technical documentation to communicate results and recommendations.

### Components

#### 5.1 SurfacePlotter Class
**Responsibility:** Generate standardized 3D surface plots and 2D heatmaps

```python
class SurfacePlotter:
    """Visualization utilities for interpolation results.

    Generates:
    - 3D surface plots of R(V, T)
    - 2D error heatmaps
    - Convergence curves
    - Uncertainty maps (for Kriging)
    - Timing comparison bar charts
    """

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        self.style = style
        self.figsize = figsize
        plt.style.use(self.style)

    def plot_3d_surface(self, V: np.ndarray, T: np.ndarray, R: np.ndarray,
                        title: str, scatter_points: Optional[np.ndarray] = None) -> plt.Figure:
        """3D surface with optional training point overlay."""
        pass

    def plot_error_heatmap(self, V: np.ndarray, T: np.ndarray, errors: np.ndarray,
                           title: str) -> plt.Figure:
        """2D heatmap showing spatial distribution of prediction errors."""
        pass

    def plot_convergence_curves(self, convergence_df: pd.DataFrame,
                                metric: str = 'rmse') -> plt.Figure:
        """Line plot: error vs n_samples for each method."""
        pass

    def plot_uncertainty_map(self, V: np.ndarray, T: np.ndarray, std: np.ndarray,
                            title: str = 'Kriging Uncertainty') -> plt.Figure:
        """2D heatmap of Kriging prediction standard deviations."""
        pass

    def plot_timing_comparison(self, benchmark_df: pd.DataFrame) -> plt.Figure:
        """Bar chart comparing train/predict times."""
        pass

    def plot_method_comparison_grid(self, results: Dict[str, np.ndarray]) -> plt.Figure:
        """2x3 grid comparing all three methods side-by-side."""
        pass

    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """Save with publication quality."""
        pass
```

#### 5.2 Visualization Requirements

**Critical Visualizations (Must-Have):**

1. **3D Surface Comparison** (3 subplots)
   - RBF, Splines, Kriging predictions on same test case
   - Training points overlaid as scatter
   - Common colorbar scale
   - File: `figures/surface_comparison.png`

2. **Error Heatmaps** (3 subplots)
   - Spatial error distribution for each method
   - Absolute error: |R_pred - R_true|
   - Highlight regions of high error
   - File: `figures/error_heatmaps.png`

3. **Convergence Curves** (single plot, 3 lines)
   - RMSE vs n_samples for [10, 20, 30, 50, 100, 200, 308]
   - Shaded confidence intervals (std across trials)
   - Log scale on y-axis if needed
   - File: `figures/convergence_analysis.png`

4. **Kriging Uncertainty Map** (2 subplots)
   - Left: Predictions
   - Right: Standard deviation map
   - Show how uncertainty increases far from training data
   - File: `figures/kriging_uncertainty.png`

5. **Computational Performance** (bar chart)
   - Train time and predict time for 308-point dataset
   - Grouped bars by method
   - File: `figures/timing_comparison.png`

**Nice-to-Have Visualizations:**

6. **Noise Robustness** (line plot)
   - RMSE vs SNR for each method
   - Show which method degrades gracefully

7. **Extrapolation Quality** (scatter plot)
   - Predicted vs True resistance for boundary points
   - Color by method

8. **Training Data Distribution** (2D scatter)
   - Show UCI data coverage in V-T space
   - Identify sparse regions

#### 5.3 Technical Documentation

**README.md** (User-Facing)
- Project overview and motivation
- Installation instructions: `pip install -r requirements.txt`
- Quick start example (5-line code snippet)
- Repository structure explanation
- Running tests: `pytest tests/`
- Reproducing results: `jupyter notebook notebooks/`
- Citation and references

**TECHNICAL_REPORT.md** (In-Depth Analysis)

**Structure:**
```markdown
# Technical Report: Interpolation Method Comparison for Ship Performance Prediction

## 1. Introduction
- Problem context: sparse hydrodynamic data
- Business motivation: cost of sea trials
- Three methods compared

## 2. Methodology
### 2.1 Dataset Description
- UCI Yacht Hydrodynamics (308 samples, 7 features)
- Proxy derivation: (Froude, B/T) → (V, T)
- Train/test split strategy

### 2.2 Interpolation Methods
#### Kriging / Gaussian Process Regression
- Mathematical formulation
- Kernel choice: RBF, Matern
- Uncertainty quantification
- Complexity: O(n³)

#### Radial Basis Functions
- Mathematical formulation
- Kernel options: thin_plate_spline, multiquadric
- Smoothing parameter
- Complexity: O(n²)

#### Bivariate Splines
- Mathematical formulation
- Smoothing splines vs interpolating splines
- Degree selection (cubic)
- Complexity: O(n)

### 2.3 Evaluation Protocol
- Metrics: RMSE, MAE, R², max error
- Cross-validation strategy
- Synthetic data validation
- Convergence analysis methodology

## 3. Results
### 3.1 Accuracy Comparison (UCI Test Set)
[Table with RMSE, MAE, R² for each method]

**Winner:** [Method] with RMSE = [value]

### 3.2 Convergence Analysis
[Convergence plot]

**Key Finding:** Kriging superior for n<50, Splines catches up at n>100

### 3.3 Computational Performance
[Timing table]

**Winner:** Splines (0.05s train, 0.02s predict for 308 points)

### 3.4 Uncertainty Quantification
[Kriging uncertainty maps]

**Observation:** Uncertainty increases at domain boundaries as expected

### 3.5 Robustness to Noise
[Noise robustness plot]

**Winner:** Kriging with regularization most robust

## 4. Discussion
### 4.1 Method Strengths and Weaknesses
[Detailed comparison table]

### 4.2 Recommendations by Use Case
- **Very sparse data (<30 points):** Kriging
- **Need uncertainty quantification:** Kriging
- **Large datasets (>200 points):** Splines (speed)
- **Irregular sampling:** RBF
- **Real-time prediction:** Splines (fast predict)

### 4.3 Yacht → Ship Transferability
- Froude scaling justification
- Same methodology applies
- Extensions to multi-dimensional interpolation

## 5. Limitations and Future Work
- 2D limitation (V, T only)
- Resistance only (not power or fuel)
- Extensions: 4D-6D interpolation, physics-informed priors

## 6. Conclusion
[Summary of findings and final recommendations]

## References
[Academic papers, scipy documentation, UCI dataset]
```

**API_REFERENCE.md**
- Class-by-class documentation
- Method signatures with type hints
- Usage examples for each interpolator
- Auto-generated from docstrings (sphinx or mkdocs)

#### 5.4 Notebooks for Visualization

**notebooks/05_visualization_gallery.ipynb**
- Generate all 5 critical visualizations
- Save to `figures/` directory
- Include captions and interpretation

**notebooks/06_final_recommendations.ipynb**
- Summary tables
- Decision tree for method selection
- Interactive widgets (ipywidgets) for exploring surfaces

### Acceptance Criteria
- ✅ All 5 critical visualizations generated and saved
- ✅ Figures are publication-quality (300 dpi, clear labels)
- ✅ README explains project in <5 minutes of reading
- ✅ Technical report is comprehensive (15-20 pages with figures)
- ✅ Code includes docstrings for all public methods (Google style)
- ✅ Project can be reproduced by external user from README alone

### Testing Requirements

**tests/test_plotter.py** (5 tests)
1. `test_plot_3d_surface_creates_figure()` - Returns matplotlib Figure
2. `test_plot_error_heatmap_values()` - Colorbar matches error range
3. `test_convergence_curves_legend()` - All methods in legend
4. `test_save_figure_creates_file()` - PNG file written to disk
5. `test_uncertainty_map_kriging_only()` - Raises error for non-Kriging

**No tests for documentation** (manual review)

### Dependencies
- MVP-1, MVP-2, MVP-3 (data and interpolators)
- MVP-4 (benchmark results to visualize)

### Risk Mitigation
- **Risk:** Visualizations unclear or misleading
  **Mitigation:** Peer review, follow data visualization best practices (colorblind-safe palettes)
- **Risk:** Documentation too technical or too shallow
  **Mitigation:** Write for two audiences (README for users, TECHNICAL_REPORT for engineers)
- **Risk:** Reproducibility fails for external user
  **Mitigation:** Test installation on clean environment (Docker or fresh VM)

### Estimated Time: 2 days

**Breakdown:**
- SurfacePlotter class + tests: 0.75 day
- Generate all visualizations: 0.5 day
- README.md: 0.25 day
- TECHNICAL_REPORT.md: 0.5 day
- API_REFERENCE.md + docstring cleanup: 0.25 day
- Final review and reproducibility test: 0.25 day

---

## 📅 Timeline and Critical Path

### Gantt Chart (Developer Days)

```
Day 1-2:   MVP-1 [Foundation] ████████
Day 2-3:   MVP-2 [Synthetic] (parallel start)  ████
Day 3-4:   MVP-3.1 [RBF] ████
Day 4-5:   MVP-3.2 [Splines] ████
Day 5-7:   MVP-3.3 [Kriging] ██████
Day 7-9:   MVP-4 [Benchmarking] ████████
Day 9-11:  MVP-5 [Visualization] ████████
Day 11-12: Buffer + Final Testing ████

Total: 11-12 developer days
```

### Critical Path
```
MVP-1 → MVP-2 → MVP-3 (RBF, Splines, Kriging sequential) → MVP-4 → MVP-5
```

**No parallelization between MVP-3 components** (RBF, Splines, Kriging must be sequential to maintain focus and testing thoroughness)

### Parallel Opportunities
- **MVP-1 + MVP-2:** Synthetic data generation doesn't require UCI loaded (0.5 day saved)
- **MVP-4 + MVP-5:** Start visualization code while benchmarks run (0.5 day saved)

### Milestones
- **Day 2:** Data pipeline complete, EDA done
- **Day 3.5:** Synthetic data validation complete
- **Day 7:** All three interpolators implemented and tested
- **Day 9:** Complete benchmark suite executed
- **Day 11:** All visualizations and documentation complete
- **Day 12:** Final reproducibility test passed

---

## 🎯 Quality Gates

Each MVP must pass these gates before proceeding:

### Gate 1 (After MVP-1)
- [ ] Test coverage >80% for data module
- [ ] All 11 tests passing
- [ ] EDA notebook runs without errors
- [ ] UCI data loads correctly (308 samples)

### Gate 2 (After MVP-2)
- [ ] Test coverage >80% for synthetic module
- [ ] All 13 tests passing
- [ ] Synthetic surface visually realistic
- [ ] MetricsCalculator matches sklearn outputs

### Gate 3 (After MVP-3)
- [ ] Test coverage >85% for interpolators module
- [ ] All 31 tests passing (8+8+10+5 integration)
- [ ] All three methods achieve RMSE < 0.05 on synthetic
- [ ] Unified API validated

### Gate 4 (After MVP-4)
- [ ] All 6 benchmarker tests passing
- [ ] 5 benchmark experiments completed
- [ ] At least one method: RMSE < 0.005 on UCI test
- [ ] Results reproducible (fixed seeds)

### Gate 5 (After MVP-5)
- [ ] All 5 critical visualizations generated
- [ ] README complete and reviewed
- [ ] Technical report >15 pages
- [ ] External reproducibility test passed

---

## 🚨 Risks and Mitigation Strategies

### High-Priority Risks

**Risk 1: Kriging Too Slow on Full Dataset**
- **Probability:** Medium (30%)
- **Impact:** High (blocks MVP-3 completion)
- **Mitigation:**
  - Test on subsamples (100, 200) first
  - Use sparse GP approximations if needed (sklearn InducingPointsKernel)
  - Document complexity limitation
  - Accept longer training time (<60s acceptable)
- **Contingency:** If >2 minutes, use subset for Kriging (200 points)

**Risk 2: UCI Proxy Derivation Produces Unrealistic Values**
- **Probability:** Low (15%)
- **Impact:** Medium (affects interpretation but not methodology)
- **Mitigation:**
  - Validate derived V, T ranges against yacht design literature
  - Emphasize in documentation that proxies are for demonstration
  - Methodology transferable regardless of absolute values
- **Contingency:** Rescale to physically meaningful ranges, document transformation

**Risk 3: Test Coverage Falls Below 80%**
- **Probability:** Low (20%)
- **Impact:** Medium (violates quality requirements)
- **Mitigation:**
  - Write tests alongside implementation (TDD)
  - Use pytest-cov to monitor coverage continuously
  - Prioritize testing critical path (interpolators, metrics)
- **Contingency:** Add tests during buffer day 12

**Risk 4: Visualizations Not Publication Quality**
- **Probability:** Medium (25%)
- **Impact:** Medium (affects deliverable quality)
- **Mitigation:**
  - Use matplotlib with seaborn styling
  - Follow data visualization best practices (clear labels, legends, colorbars)
  - Peer review figures
- **Contingency:** Iterate on figures during MVP-5, allocate extra 0.5 day if needed

### Medium-Priority Risks

**Risk 5: Benchmark Experiments Take Too Long**
- **Mitigation:** Start with small sample sizes, use multiprocessing, cache results

**Risk 6: Git Merge Conflicts (if multiple branches)**
- **Mitigation:** Work on single branch sequentially, commit frequently with clear messages

**Risk 7: Dependency Version Conflicts**
- **Mitigation:** Pin versions in requirements.txt, test on clean virtualenv

---

## 📦 Deliverables Summary

### Code Deliverables
1. **src/** - Production-quality Python modules
   - data/ (loader, preprocessor, synthetic)
   - interpolators/ (base, rbf, spline, kriging)
   - evaluation/ (metrics, benchmarker)
   - visualization/ (plotter)

2. **tests/** - Comprehensive test suite (>80% coverage)
   - 11 tests (MVP-1)
   - 13 tests (MVP-2)
   - 31 tests (MVP-3)
   - 6 tests (MVP-4)
   - 5 tests (MVP-5)
   - **Total: 66 tests**

3. **notebooks/** - Analysis and exploration
   - 01_exploratory_analysis.ipynb
   - 02_synthetic_benchmark.ipynb
   - 03_uci_benchmark.ipynb
   - 04_computational_analysis.ipynb
   - 05_visualization_gallery.ipynb
   - 06_final_recommendations.ipynb

### Data Deliverables
4. **data/** - Raw and processed data
   - yacht_hydro.xls (original UCI)
   - synthetic_surfaces.pkl (cached)
   - train_test_splits.pkl

5. **results/** - Benchmark outputs
   - accuracy_comparison.csv
   - convergence_analysis.csv
   - noise_robustness.csv
   - timing_benchmarks.csv
   - extrapolation_test.csv

### Visualization Deliverables
6. **figures/** - Publication-quality plots (300 dpi PNG)
   - surface_comparison.png
   - error_heatmaps.png
   - convergence_analysis.png
   - kriging_uncertainty.png
   - timing_comparison.png

### Documentation Deliverables
7. **README.md** - User guide and quick start
8. **TECHNICAL_REPORT.md** - In-depth analysis (15-20 pages)
9. **API_REFERENCE.md** - Class and method documentation
10. **requirements.txt** - Pinned dependencies
11. **setup.py** - Package installation

---

## 🎓 Success Metrics Validation

### Technical Excellence
- [x] Three interpolation methods implemented with unified API
- [ ] Test coverage >80% (target: 66 tests)
- [ ] RMSE < 0.005 on UCI test set (at least one method)
- [ ] Kriging provides uncertainty quantification
- [ ] All code passes pylint score >8.0

### Functional Completeness
- [ ] 5 benchmark experiments completed
- [ ] 5 critical visualizations generated
- [ ] Technical report >15 pages with equations and references
- [ ] Recommendations by use case documented

### Professional Presentation
- [ ] External user can reproduce from README
- [ ] Code readable (type hints, docstrings)
- [ ] Git history clean (meaningful commits)
- [ ] No hardcoded paths or magic numbers

### Demonstration of Skills (for Applied Mathematics Position)
- ✅ **Data Interpolation (PRIMARY):** Kriging, RBF, Splines implemented
- ✅ **Data Analysis:** Feature engineering (V, T derivation), EDA, benchmarking
- ✅ **Mathematical Rigor:** Equations documented, complexity analysis
- ✅ **Software Engineering:** Class-based design, testing, documentation
- ✅ **Communication:** Technical report, visualizations, recommendations

---

## 📞 Next Steps for Development AI

This WBS is now ready for implementation. The Development AI should:

1. **Start with MVP-1:** Create project structure, implement data loaders
2. **Sequential execution:** Complete each MVP before proceeding
3. **Pass quality gates:** Run tests after each MVP
4. **Commit frequently:** Meaningful git messages per component
5. **Document as you go:** Write docstrings during implementation
6. **Flag blockers early:** Report if any assumptions are invalid

**First Task:** Set up project structure and implement YachtDataLoader class from MVP-1.

---

## 📚 Appendix: Technology Stack

### Core Libraries
- **Python:** 3.10+
- **Numerical:** numpy>=1.24, scipy>=1.10
- **ML:** scikit-learn>=1.3 (GaussianProcessRegressor)
- **Data:** pandas>=2.0, openpyxl (for .xls)
- **Visualization:** matplotlib>=3.7, seaborn>=0.12
- **Testing:** pytest>=7.4, pytest-cov>=4.1

### Development Tools
- **Formatting:** black>=23.0
- **Linting:** pylint>=2.17
- **Type checking:** mypy>=1.4 (optional)
- **Notebooks:** jupyter>=1.0, ipywidgets>=8.0

### Repository Structure
```
navig_p1/
├── src/                    # Source code
├── tests/                  # Test suite
├── notebooks/              # Analysis notebooks
├── data/                   # Datasets
├── results/                # Benchmark outputs
├── figures/                # Visualizations
├── docs/                   # Documentation
├── .gitignore
├── requirements.txt
├── setup.py
├── README.md
├── TECHNICAL_REPORT.md
└── WBS_INTERPOLATION_PROJECT.md (this file)
```

---

**WBS Version:** 1.0
**Last Updated:** 2025-11-18
**Status:** Ready for Implementation
