"""Tests for Spline Interpolator."""

import pytest
import numpy as np
from src.interpolators.spline import SplineInterpolator
from src.data.synthetic import SyntheticSurfaceGenerator


@pytest.fixture
def simple_training_data():
    """Create simple 2D training data (40 points to support high degree splines)."""
    # Need at least (kx+1)*(ky+1) points for splines
    # For kx=ky=5 (quintic), need 36 points minimum
    np.random.seed(42)
    n_points = 40
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


def test_spline_fit_sets_flag(simple_training_data):
    """Test that fit() sets is_fitted flag."""
    X, y = simple_training_data
    spline = SplineInterpolator()

    assert spline.is_fitted == False

    spline.fit(X, y)

    assert spline.is_fitted == True
    assert spline.n_training_samples == 40
    assert spline.train_time is not None
    assert spline.train_time > 0


def test_spline_predict_before_fit_raises(simple_training_data):
    """Test that predict() before fit() raises ValueError."""
    X, y = simple_training_data
    spline = SplineInterpolator()

    with pytest.raises(ValueError, match="must be fitted"):
        spline.predict(X)


def test_spline_perfect_interpolation_noiseless(simple_training_data):
    """Test that spline closely interpolates training points with smoothing=0."""
    X, y = simple_training_data
    spline = SplineInterpolator(smoothing=0.0)

    spline.fit(X, y)
    predictions = spline.predict(X)

    # Should match training data reasonably well
    # (not perfect due to spline approximation vs scattered data)
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    assert rmse < 1.0, f"RMSE too high: {rmse:.4f}"


def test_spline_shape_consistency(simple_training_data):
    """Test that output shape matches input shape."""
    X_train, y_train = simple_training_data
    spline = SplineInterpolator()
    spline.fit(X_train, y_train)

    # Test with various query sizes
    for n_queries in [1, 5, 10, 100]:
        X_test = np.random.rand(n_queries, 2) * 10 + 10
        predictions = spline.predict(X_test)

        assert predictions.shape == (n_queries,), \
            f"Expected shape ({n_queries},), got {predictions.shape}"


def test_spline_different_degrees(simple_training_data):
    """Test that different spline degrees work."""
    X, y = simple_training_data

    degrees = [(1, 1), (3, 3), (5, 5), (3, 5)]

    for kx, ky in degrees:
        spline = SplineInterpolator(kx=kx, ky=ky)
        spline.fit(X, y)
        predictions = spline.predict(X)

        assert len(predictions) == len(y), f"Degree ({kx}, {ky}) failed"
        assert not np.isnan(predictions).any(), f"Degree ({kx}, {ky}) produced NaN"


def test_spline_smoothing_effect(synthetic_data):
    """Test that smoothing>0 produces smoother surface."""
    X, y, generator = synthetic_data

    # Exact interpolation
    spline_exact = SplineInterpolator(smoothing=0.0)
    spline_exact.fit(X, y)
    pred_exact = spline_exact.predict(X)

    # Smoothed interpolation
    spline_smooth = SplineInterpolator(smoothing=1.0)
    spline_smooth.fit(X, y)
    pred_smooth = spline_smooth.predict(X)

    # Exact interpolation should match training data better
    error_exact = np.mean((pred_exact - y) ** 2)
    error_smooth = np.mean((pred_smooth - y) ** 2)

    assert error_exact <= error_smooth, \
        "Smoothed spline should have higher or equal training error"


def test_spline_speed_benchmark(synthetic_data):
    """Test that spline is fast compared to expected performance."""
    X, y, generator = synthetic_data

    spline = SplineInterpolator()
    spline.fit(X, y)

    # Training should be fast (<0.1s for 50 points)
    assert spline.train_time < 0.1, \
        f"Training too slow: {spline.train_time:.3f}s"

    # Prediction should be very fast
    X_test = np.random.rand(100, 2) * 15 + 10
    spline.predict(X_test)

    assert spline.predict_time < 0.01, \
        f"Prediction too slow: {spline.predict_time:.3f}s"


def test_spline_extrapolation_warning(simple_training_data):
    """Test extrapolation beyond training domain (document behavior)."""
    X_train, y_train = simple_training_data
    spline = SplineInterpolator()
    spline.fit(X_train, y_train)

    # Query outside training domain
    X_extrap = np.array([[30, 10], [5, 5]])  # Outside [10,25] x [6,9]
    predictions = spline.predict(X_extrap)

    # Should still return predictions (splines extrapolate)
    assert len(predictions) == 2
    assert not np.isnan(predictions).any()
    # Note: extrapolation quality not guaranteed


def test_spline_synthetic_validation(synthetic_data):
    """Test spline on synthetic data with known ground truth."""
    X_train, y_train, generator = synthetic_data

    # Train spline
    spline = SplineInterpolator(smoothing=0.0)
    spline.fit(X_train, y_train)

    # Generate test grid
    grid = generator.create_dense_grid(grid_size=20)
    X_test = np.column_stack([grid['V'], grid['T']])
    y_true = grid['R']

    # Predict
    y_pred = spline.predict(X_test)

    # Compute error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Should achieve reasonable accuracy
    assert rmse < 2.0, f"RMSE too high: {rmse:.4f}"


def test_spline_metadata(simple_training_data):
    """Test get_metadata() returns correct information."""
    X, y = simple_training_data
    spline = SplineInterpolator(smoothing=0.5, kx=3, ky=5)

    # Before fitting
    meta_before = spline.get_metadata()
    assert meta_before['is_fitted'] == False
    assert meta_before['smoothing'] == 0.5
    assert meta_before['kx'] == 3
    assert meta_before['ky'] == 5

    # After fitting
    spline.fit(X, y)
    meta_after = spline.get_metadata()
    assert meta_after['is_fitted'] == True
    assert meta_after['n_training_samples'] == 40
    assert meta_after['train_time'] > 0


def test_spline_fit_predict_chain(simple_training_data):
    """Test fit_predict() convenience method."""
    X_train, y_train = simple_training_data
    X_test = np.array([[12, 6.5], [22, 8.5]])

    spline = SplineInterpolator()
    predictions = spline.fit_predict(X_train, y_train, X_test)

    assert len(predictions) == 2
    assert spline.is_fitted == True


def test_spline_invalid_degrees():
    """Test that invalid spline degrees raise ValueError."""
    # kx too low
    with pytest.raises(ValueError, match="kx must be between"):
        SplineInterpolator(kx=0, ky=3)

    # ky too high
    with pytest.raises(ValueError, match="ky must be between"):
        SplineInterpolator(kx=3, ky=6)


def test_spline_too_few_samples():
    """Test that too few samples for spline degree raises error."""
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Only 3 samples
    y = np.array([5, 6, 7])

    spline = SplineInterpolator(kx=3, ky=3)  # Requires (3+1)*(3+1)=16 samples

    with pytest.raises(ValueError, match="Need at least 16 samples.*got 3"):
        spline.fit(X, y)


def test_spline_evaluate_grid(simple_training_data):
    """Test evaluate_grid() method for efficient grid evaluation."""
    X, y = simple_training_data
    spline = SplineInterpolator()
    spline.fit(X, y)

    # Create grid
    v_grid = np.linspace(10, 25, 20)
    t_grid = np.linspace(6, 9, 15)

    # Evaluate on grid
    R_grid = spline.evaluate_grid(v_grid, t_grid)

    # Check shape
    assert R_grid.shape == (20, 15), f"Expected shape (20, 15), got {R_grid.shape}"
    assert not np.isnan(R_grid).any()


def test_spline_repr(simple_training_data):
    """Test string representation."""
    X, y = simple_training_data
    spline = SplineInterpolator(kx=5, ky=3, smoothing=0.1)

    # Before fitting
    repr_before = repr(spline)
    assert 'kx=5' in repr_before
    assert 'ky=3' in repr_before
    assert 'not fitted' in repr_before

    # After fitting
    spline.fit(X, y)
    repr_after = repr(spline)
    assert 'fitted' in repr_after
    assert '40 samples' in repr_after


def test_spline_deterministic():
    """Test that same data produces same results."""
    X = np.random.RandomState(42).rand(20, 2) * 10
    y = np.random.RandomState(42).rand(20) * 30

    X_test = np.random.RandomState(123).rand(10, 2) * 10

    spline1 = SplineInterpolator()
    spline1.fit(X, y)
    pred1 = spline1.predict(X_test)

    spline2 = SplineInterpolator()
    spline2.fit(X, y)
    pred2 = spline2.predict(X_test)

    np.testing.assert_array_almost_equal(pred1, pred2, decimal=10,
                                         err_msg="Spline should be deterministic")


def test_spline_grid_vs_points(simple_training_data):
    """Test that evaluate_grid and predict give consistent results."""
    X, y = simple_training_data
    spline = SplineInterpolator()
    spline.fit(X, y)

    # Create grid
    v_grid = np.array([12, 15, 18])
    t_grid = np.array([6.5, 7.0, 7.5])

    # Evaluate using grid method
    R_grid = spline.evaluate_grid(v_grid, t_grid)

    # Evaluate using point method at grid corners
    points = [[12, 6.5], [15, 7.0], [18, 7.5]]
    X_points = np.array(points)
    R_points = spline.predict(X_points)

    # Extract diagonal from grid (matching points)
    R_grid_diag = np.array([R_grid[0, 0], R_grid[1, 1], R_grid[2, 2]])

    # Should match closely
    np.testing.assert_array_almost_equal(R_grid_diag, R_points, decimal=6)
