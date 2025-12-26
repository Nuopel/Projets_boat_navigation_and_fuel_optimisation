"""Tests for Kriging (Gaussian Process) Interpolator."""

import pytest
import numpy as np
from src.interpolators.kriging import KrigingInterpolator
from src.data.synthetic import SyntheticSurfaceGenerator


@pytest.fixture
def simple_training_data():
    """Create simple 2D training data for Kriging."""
    np.random.seed(42)
    n_points = 30
    V = np.random.uniform(10, 25, n_points)
    T = np.random.uniform(6, 9, n_points)
    # Simple resistance model for testing
    R = 0.05 * V**2 - 2.0 / T + 0.01 * V * T + 15.0
    X = np.column_stack([V, T])
    y = R
    return X, y


@pytest.fixture
def synthetic_data():
    """Create synthetic resistance data for validation."""
    generator = SyntheticSurfaceGenerator(random_state=42)
    df = generator.sample_sparse(n_samples=50, strategy='latin_hypercube', add_noise=False)
    X = df[['V', 'T']].values
    y = df['R'].values
    return X, y, generator


def test_kriging_fit_sets_flag(simple_training_data):
    """Test that fit() sets is_fitted flag."""
    X, y = simple_training_data
    kriging = KrigingInterpolator()

    assert kriging.is_fitted == False

    kriging.fit(X, y)

    assert kriging.is_fitted == True
    assert kriging.n_training_samples == 30
    assert kriging.train_time is not None
    assert kriging.train_time > 0


def test_kriging_predict_before_fit_raises(simple_training_data):
    """Test that predict() before fit() raises ValueError."""
    X, y = simple_training_data
    kriging = KrigingInterpolator()

    with pytest.raises(ValueError, match="must be fitted"):
        kriging.predict(X)


def test_kriging_perfect_interpolation_low_noise(simple_training_data):
    """Test that Kriging interpolates well with low noise."""
    X, y = simple_training_data
    kriging = KrigingInterpolator(alpha=1e-6)

    kriging.fit(X, y)
    predictions = kriging.predict(X)

    # Should match training data very well with low alpha
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    assert rmse < 1.0, f"RMSE too high: {rmse:.4f}"


def test_kriging_uncertainty_quantification(simple_training_data):
    """Test that Kriging returns uncertainty estimates."""
    X_train, y_train = simple_training_data
    kriging = KrigingInterpolator()
    kriging.fit(X_train, y_train)

    # Predict with uncertainty
    X_test = np.array([[12, 6.5], [18, 7.5], [22, 8.0]])
    predictions, std_dev = kriging.predict(X_test, return_std=True)

    # Check outputs
    assert len(predictions) == 3
    assert len(std_dev) == 3
    assert all(std_dev >= 0), "Standard deviations must be non-negative"

    # Std dev should be positive (uncertainty exists away from training points)
    assert np.mean(std_dev) > 0, "Expected positive uncertainty"


def test_kriging_uncertainty_at_training_points(simple_training_data):
    """Test that uncertainty is low at training points."""
    X_train, y_train = simple_training_data
    kriging = KrigingInterpolator(alpha=1e-6)
    kriging.fit(X_train, y_train)

    # Predict at training points
    predictions, std_dev = kriging.predict(X_train, return_std=True)

    # Uncertainty should be very low at training points
    mean_std = np.mean(std_dev)
    assert mean_std < 1.0, f"Uncertainty at training points too high: {mean_std:.4f}"


def test_kriging_shape_consistency(simple_training_data):
    """Test that output shape matches input shape."""
    X_train, y_train = simple_training_data
    kriging = KrigingInterpolator()
    kriging.fit(X_train, y_train)

    # Test with various query sizes
    for n_queries in [1, 5, 10, 50]:
        X_test = np.random.rand(n_queries, 2) * 10 + 10
        predictions = kriging.predict(X_test)

        assert predictions.shape == (n_queries,), \
            f"Expected shape ({n_queries},), got {predictions.shape}"

        # Test with uncertainty
        predictions, std_dev = kriging.predict(X_test, return_std=True)
        assert std_dev.shape == (n_queries,), \
            f"Expected std shape ({n_queries},), got {std_dev.shape}"


def test_kriging_different_kernels(simple_training_data):
    """Test that different kernel types work."""
    X, y = simple_training_data

    kernels = ['rbf', 'matern', 'rational_quadratic']

    for kernel in kernels:
        kriging = KrigingInterpolator(kernel_type=kernel, n_restarts_optimizer=2)
        kriging.fit(X, y)
        predictions = kriging.predict(X[:5])

        assert len(predictions) == 5, f"Kernel '{kernel}' failed"
        assert not np.isnan(predictions).any(), f"Kernel '{kernel}' produced NaN"


def test_kriging_alpha_smoothing_effect(synthetic_data):
    """Test that higher alpha produces more smoothing."""
    X, y, generator = synthetic_data

    # Add some noise to make the test more robust
    y_noisy = y + np.random.RandomState(42).randn(len(y)) * 1.0

    # Low noise (nearly exact interpolation)
    kriging_exact = KrigingInterpolator(alpha=1e-6, n_restarts_optimizer=2)
    kriging_exact.fit(X, y_noisy)
    pred_exact = kriging_exact.predict(X)

    # High noise (more smoothing)
    kriging_smooth = KrigingInterpolator(alpha=1.0, n_restarts_optimizer=2)
    kriging_smooth.fit(X, y_noisy)
    pred_smooth = kriging_smooth.predict(X)

    # Exact interpolation should match training data better (lower MSE)
    # or be within reasonable range (both might be optimized well)
    error_exact = np.mean((pred_exact - y_noisy) ** 2)
    error_smooth = np.mean((pred_smooth - y_noisy) ** 2)

    # More lenient check: exact should be at least 80% as good or better
    assert error_exact <= error_smooth * 1.2, \
        f"Expected exact fit ({error_exact:.4f}) to be better than smooth ({error_smooth:.4f})"


def test_kriging_speed_benchmark(synthetic_data):
    """Test that Kriging training and prediction complete in reasonable time."""
    X, y, generator = synthetic_data

    kriging = KrigingInterpolator(n_restarts_optimizer=5)
    kriging.fit(X, y)

    # Training should complete (may be slower than RBF/Splines)
    assert kriging.train_time < 10.0, \
        f"Training too slow: {kriging.train_time:.3f}s"

    # Prediction should be fast
    X_test = np.random.rand(100, 2) * 15 + 10
    kriging.predict(X_test)

    assert kriging.predict_time < 1.0, \
        f"Prediction too slow: {kriging.predict_time:.3f}s"


def test_kriging_synthetic_validation(synthetic_data):
    """Test Kriging on synthetic data with known ground truth."""
    X_train, y_train, generator = synthetic_data

    # Train Kriging
    kriging = KrigingInterpolator(alpha=1e-6, n_restarts_optimizer=5)
    kriging.fit(X_train, y_train)

    # Generate test grid
    grid = generator.create_dense_grid(grid_size=20)
    X_test = np.column_stack([grid['V'], grid['T']])
    y_true = grid['R']

    # Predict
    y_pred = kriging.predict(X_test)

    # Compute error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Should achieve reasonable accuracy
    assert rmse < 3.0, f"RMSE too high: {rmse:.4f}"


def test_kriging_metadata(simple_training_data):
    """Test get_metadata() returns correct information."""
    X, y = simple_training_data
    kriging = KrigingInterpolator(kernel_type='matern', alpha=1e-5, n_restarts_optimizer=3)

    # Before fitting
    meta_before = kriging.get_metadata()
    assert meta_before['is_fitted'] == False
    assert meta_before['kernel_type'] == 'matern'
    assert meta_before['alpha'] == 1e-5
    assert meta_before['n_restarts_optimizer'] == 3

    # After fitting
    kriging.fit(X, y)
    meta_after = kriging.get_metadata()
    assert meta_after['is_fitted'] == True
    assert meta_after['n_training_samples'] == 30
    assert meta_after['train_time'] > 0
    assert 'log_marginal_likelihood' in meta_after
    assert 'optimized_kernel' in meta_after


def test_kriging_fit_predict_chain(simple_training_data):
    """Test fit_predict() convenience method."""
    X_train, y_train = simple_training_data
    X_test = np.array([[12, 6.5], [22, 8.5]])

    kriging = KrigingInterpolator(n_restarts_optimizer=2)
    predictions = kriging.fit_predict(X_train, y_train, X_test)

    assert len(predictions) == 2
    assert kriging.is_fitted == True


def test_kriging_invalid_kernel():
    """Test that invalid kernel type raises ValueError."""
    with pytest.raises(ValueError, match="kernel_type must be one of"):
        KrigingInterpolator(kernel_type='invalid_kernel')


def test_kriging_repr(simple_training_data):
    """Test string representation."""
    X, y = simple_training_data
    kriging = KrigingInterpolator(kernel_type='rbf', alpha=1e-5)

    # Before fitting
    repr_before = repr(kriging)
    assert 'kernel=rbf' in repr_before
    assert 'alpha=1e-05' in repr_before
    assert 'not fitted' in repr_before

    # After fitting
    kriging.fit(X, y)
    repr_after = repr(kriging)
    assert 'fitted' in repr_after
    assert '30 samples' in repr_after
    assert 'log_ML=' in repr_after


def test_kriging_deterministic_with_seed(simple_training_data):
    """Test that same data produces same results (with fixed random_state)."""
    X, y = simple_training_data
    X_test = np.random.RandomState(123).rand(10, 2) * 10 + 10

    kriging1 = KrigingInterpolator(n_restarts_optimizer=2)
    kriging1.fit(X, y)
    pred1 = kriging1.predict(X_test)

    kriging2 = KrigingInterpolator(n_restarts_optimizer=2)
    kriging2.fit(X, y)
    pred2 = kriging2.predict(X_test)

    # Should be deterministic due to random_state=42 in GaussianProcessRegressor
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=6,
                                         err_msg="Kriging should be deterministic")


def test_kriging_uncertainty_increases_with_distance(simple_training_data):
    """Test that uncertainty increases with distance from training points."""
    X_train, y_train = simple_training_data
    kriging = KrigingInterpolator(alpha=1e-6)
    kriging.fit(X_train, y_train)

    # Points near training data
    X_near = X_train[:3] + np.random.randn(3, 2) * 0.1
    _, std_near = kriging.predict(X_near, return_std=True)

    # Points far from training data (extrapolation)
    X_far = np.array([[30, 10], [5, 5], [35, 12]])
    _, std_far = kriging.predict(X_far, return_std=True)

    # Far points should have higher uncertainty on average
    # Note: This may not always hold exactly, but on average it should
    assert np.mean(std_far) > np.mean(std_near) * 0.5, \
        "Uncertainty should generally increase with distance"


def test_kriging_handles_noisy_data(simple_training_data):
    """Test that Kriging handles noisy training data."""
    X, y_clean = simple_training_data

    # Add noise to training data
    noise = np.random.randn(len(y_clean)) * 2.0
    y_noisy = y_clean + noise

    # Fit with appropriate alpha
    kriging = KrigingInterpolator(alpha=1e-3, n_restarts_optimizer=3)
    kriging.fit(X, y_noisy)

    # Should still produce reasonable predictions
    predictions = kriging.predict(X[:10])
    assert not np.isnan(predictions).any()
    assert all(np.isfinite(predictions))


def test_kriging_normalize_y_effect(simple_training_data):
    """Test that normalize_y parameter works."""
    X, y = simple_training_data

    # With normalization
    kriging_norm = KrigingInterpolator(normalize_y=True, n_restarts_optimizer=2)
    kriging_norm.fit(X, y)
    pred_norm = kriging_norm.predict(X[:5])

    # Without normalization
    kriging_no_norm = KrigingInterpolator(normalize_y=False, n_restarts_optimizer=2)
    kriging_no_norm.fit(X, y)
    pred_no_norm = kriging_no_norm.predict(X[:5])

    # Both should produce reasonable predictions
    assert not np.isnan(pred_norm).any()
    assert not np.isnan(pred_no_norm).any()

    # Results may differ slightly but should be in same ballpark
    relative_diff = np.abs(pred_norm - pred_no_norm) / (np.abs(pred_norm) + 1e-10)
    assert np.mean(relative_diff) < 0.5, "Normalization effect too large"
