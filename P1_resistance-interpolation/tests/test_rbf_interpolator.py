"""Tests for RBF Interpolator."""

import pytest
import numpy as np
from src.interpolators.rbf import RBFInterpolator
from src.data.synthetic import SyntheticSurfaceGenerator


@pytest.fixture
def simple_training_data():
    """Create simple 2D training data."""
    X = np.array([
        [10, 6],
        [15, 7],
        [20, 8],
        [25, 9]
    ])
    y = np.array([20.0, 25.0, 35.0, 48.0])
    return X, y


@pytest.fixture
def synthetic_data():
    """Create synthetic resistance data for validation."""
    generator = SyntheticSurfaceGenerator(random_state=42)
    df = generator.sample_sparse(n_samples=50, strategy='latin_hypercube', add_noise=False)
    X = df[['V', 'T']].values
    y = df['R'].values
    return X, y, generator


def test_rbf_fit_sets_flag(simple_training_data):
    """Test that fit() sets is_fitted flag."""
    X, y = simple_training_data
    rbf = RBFInterpolator()

    assert rbf.is_fitted == False

    rbf.fit(X, y)

    assert rbf.is_fitted == True
    assert rbf.n_training_samples == 4
    assert rbf.train_time is not None
    assert rbf.train_time > 0


def test_rbf_predict_before_fit_raises(simple_training_data):
    """Test that predict() before fit() raises ValueError."""
    X, y = simple_training_data
    rbf = RBFInterpolator()

    with pytest.raises(ValueError, match="must be fitted"):
        rbf.predict(X)


def test_rbf_perfect_interpolation_noiseless(simple_training_data):
    """Test that RBF exactly interpolates training points with smoothing=0."""
    X, y = simple_training_data
    rbf = RBFInterpolator(kernel='thin_plate_spline', smoothing=0.0)

    rbf.fit(X, y)
    predictions = rbf.predict(X)

    # Should match training data exactly (or very close due to numerics)
    np.testing.assert_array_almost_equal(predictions, y, decimal=6)


def test_rbf_shape_consistency(simple_training_data):
    """Test that output shape matches input shape."""
    X_train, y_train = simple_training_data
    rbf = RBFInterpolator()
    rbf.fit(X_train, y_train)

    # Test with various query sizes
    for n_queries in [1, 5, 10, 100]:
        X_test = np.random.rand(n_queries, 2) * 10 + 10
        predictions = rbf.predict(X_test)

        assert predictions.shape == (n_queries,), \
            f"Expected shape ({n_queries},), got {predictions.shape}"


def test_rbf_different_kernels(simple_training_data):
    """Test that different kernel functions work."""
    X, y = simple_training_data

    kernels = ['thin_plate_spline', 'multiquadric', 'gaussian', 'linear', 'cubic']

    for kernel in kernels:
        rbf = RBFInterpolator(kernel=kernel)
        rbf.fit(X, y)
        predictions = rbf.predict(X)

        assert len(predictions) == len(y), f"Kernel '{kernel}' failed"
        assert not np.isnan(predictions).any(), f"Kernel '{kernel}' produced NaN"


def test_rbf_smoothing_reduces_overfitting(synthetic_data):
    """Test that smoothing>0 has different behavior than exact interpolation."""
    X, y, generator = synthetic_data

    # Exact interpolation
    rbf_exact = RBFInterpolator(smoothing=0.0)
    rbf_exact.fit(X, y)
    pred_exact = rbf_exact.predict(X)

    # Smoothed interpolation
    rbf_smooth = RBFInterpolator(smoothing=0.01)
    rbf_smooth.fit(X, y)
    pred_smooth = rbf_smooth.predict(X)

    # Exact interpolation should match training data better
    error_exact = np.mean((pred_exact - y) ** 2)
    error_smooth = np.mean((pred_smooth - y) ** 2)

    assert error_exact < error_smooth, \
        "Smoothed RBF should have higher training error"


def test_rbf_timing_metadata(simple_training_data):
    """Test that train_time and predict_time are recorded."""
    X, y = simple_training_data
    rbf = RBFInterpolator()

    rbf.fit(X, y)
    assert rbf.train_time > 0

    rbf.predict(X)
    assert rbf.predict_time is not None
    assert rbf.predict_time >= 0


def test_rbf_synthetic_validation(synthetic_data):
    """Test RBF on synthetic data with known ground truth."""
    X_train, y_train, generator = synthetic_data

    # Train RBF
    rbf = RBFInterpolator(kernel='thin_plate_spline')
    rbf.fit(X_train, y_train)

    # Generate test grid
    grid = generator.create_dense_grid(grid_size=20)  # Small grid for speed
    X_test = np.column_stack([grid['V'], grid['T']])
    y_true = grid['R']

    # Predict
    y_pred = rbf.predict(X_test)

    # Compute error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Should achieve reasonable accuracy
    assert rmse < 2.0, f"RMSE too high: {rmse:.4f}"


def test_rbf_metadata(simple_training_data):
    """Test get_metadata() returns correct information."""
    X, y = simple_training_data
    rbf = RBFInterpolator(kernel='multiquadric', smoothing=0.05)

    # Before fitting
    meta_before = rbf.get_metadata()
    assert meta_before['is_fitted'] == False
    assert meta_before['kernel'] == 'multiquadric'
    assert meta_before['smoothing'] == 0.05

    # After fitting
    rbf.fit(X, y)
    meta_after = rbf.get_metadata()
    assert meta_after['is_fitted'] == True
    assert meta_after['n_training_samples'] == 4
    assert meta_after['train_time'] > 0


def test_rbf_fit_predict_chain(simple_training_data):
    """Test fit_predict() convenience method."""
    X_train, y_train = simple_training_data
    X_test = np.array([[12, 6.5], [22, 8.5]])

    rbf = RBFInterpolator()
    predictions = rbf.fit_predict(X_train, y_train, X_test)

    assert len(predictions) == 2
    assert rbf.is_fitted == True


def test_rbf_invalid_input_shapes():
    """Test that invalid input shapes raise ValueError."""
    rbf = RBFInterpolator()

    # 1D array instead of 2D
    with pytest.raises(ValueError, match="2D array"):
        rbf.fit(np.array([1, 2, 3]), np.array([4, 5, 6]))

    # Wrong number of features
    with pytest.raises(ValueError, match="2 features"):
        rbf.fit(np.array([[1], [2], [3]]), np.array([4, 5, 6]))

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        rbf.fit(np.array([[1, 2], [3, 4]]), np.array([5, 6, 7]))


def test_rbf_nan_inf_handling():
    """Test that NaN and inf values raise errors."""
    rbf = RBFInterpolator()

    # NaN in X
    X_nan = np.array([[1, 2], [np.nan, 3], [4, 5]])
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="NaN"):
        rbf.fit(X_nan, y)

    # Inf in y
    X = np.array([[1, 2], [3, 4]])
    y_inf = np.array([1, np.inf])
    with pytest.raises(ValueError, match="infinite"):
        rbf.fit(X, y_inf)


def test_rbf_empty_data():
    """Test that empty data raises error."""
    rbf = RBFInterpolator()

    X_empty = np.array([]).reshape(0, 2)
    y_empty = np.array([])

    with pytest.raises(ValueError, match="cannot be empty"):
        rbf.fit(X_empty, y_empty)


def test_rbf_repr(simple_training_data):
    """Test string representation."""
    X, y = simple_training_data
    rbf = RBFInterpolator(kernel='gaussian', smoothing=0.1)

    # Before fitting
    repr_before = repr(rbf)
    assert 'gaussian' in repr_before
    assert 'not fitted' in repr_before

    # After fitting
    rbf.fit(X, y)
    repr_after = repr(rbf)
    assert 'fitted' in repr_after
    assert '4 samples' in repr_after


def test_rbf_epsilon_parameter():
    """Test that epsilon parameter can be set."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([10, 20, 30])

    rbf = RBFInterpolator(kernel='gaussian', epsilon=0.5)
    rbf.fit(X, y)

    meta = rbf.get_metadata()
    assert meta['epsilon'] == 0.5

    # Should still make predictions
    predictions = rbf.predict(X)
    assert len(predictions) == 3


def test_rbf_deterministic():
    """Test that same seed produces same results."""
    X = np.random.RandomState(42).rand(20, 2) * 10
    y = np.random.RandomState(42).rand(20) * 30

    X_test = np.random.RandomState(123).rand(10, 2) * 10

    rbf1 = RBFInterpolator(kernel='thin_plate_spline')
    rbf1.fit(X, y)
    pred1 = rbf1.predict(X_test)

    rbf2 = RBFInterpolator(kernel='thin_plate_spline')
    rbf2.fit(X, y)
    pred2 = rbf2.predict(X_test)

    np.testing.assert_array_equal(pred1, pred2,
                                   err_msg="RBF should be deterministic")
