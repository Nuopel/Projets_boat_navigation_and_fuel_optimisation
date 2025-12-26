"""Tests for DataPreprocessor class."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance with default parameters."""
    return DataPreprocessor(reference_length=10.0, reference_beam=3.0)


@pytest.fixture
def sample_yacht_data():
    """Create sample yacht data for testing."""
    data = {
        'Froude_number': [0.20, 0.25, 0.30, 0.35, 0.40],
        'Beam_draught_ratio': [3.0, 3.5, 4.0, 3.2, 2.8],
        'Residuary_resistance': [0.5, 1.2, 2.3, 3.1, 4.5]
    }
    return pd.DataFrame(data)


def test_derive_velocity_froude_range(preprocessor):
    """Test that velocity increases monotonically with Froude number."""
    froude_values = np.array([0.2, 0.3, 0.4, 0.5])
    velocities = preprocessor.derive_velocity(froude_values)

    # Check that velocities increase with Froude number
    assert all(velocities[i] < velocities[i+1] for i in range(len(velocities)-1)), \
        "Velocity should increase monotonically with Froude number"

    # Check that velocities are positive
    assert all(velocities > 0), "All velocities should be positive"


def test_derive_velocity_values(preprocessor):
    """Test velocity calculation with known values."""
    # Fr = 0.3, L = 10m, g = 9.81 m/s²
    # V = 0.3 × √(9.81 × 10) = 0.3 × 9.905 = 2.9715 m/s
    # V in knots = 2.9715 × 1.94384 ≈ 5.776 knots
    froude = np.array([0.3])
    velocity = preprocessor.derive_velocity(froude)

    expected_velocity = 0.3 * np.sqrt(9.81 * 10.0) * 1.94384
    assert np.isclose(velocity[0], expected_velocity, rtol=1e-4), \
        f"Expected velocity {expected_velocity:.2f}, got {velocity[0]:.2f}"


def test_derive_draft_ratio_range(preprocessor):
    """Test that draft decreases as beam-draft ratio increases."""
    # Higher B/T ratio means narrower beam relative to draft → smaller draft
    ratios = np.array([2.5, 3.0, 3.5, 4.0])
    drafts = preprocessor.derive_draft(ratios)

    # Check that draft decreases with increasing ratio
    assert all(drafts[i] > drafts[i+1] for i in range(len(drafts)-1)), \
        "Draft should decrease as beam-draft ratio increases"

    # Check that drafts are positive
    assert all(drafts > 0), "All drafts should be positive"


def test_derive_draft_values(preprocessor):
    """Test draft calculation with known values."""
    # Beam = 3m, B/T = 3.0
    # T = 3 / 3.0 = 1.0 m
    ratio = np.array([3.0])
    draft = preprocessor.derive_draft(ratio)

    expected_draft = 3.0 / 3.0
    assert np.isclose(draft[0], expected_draft, rtol=1e-4), \
        f"Expected draft {expected_draft:.2f}, got {draft[0]:.2f}"


def test_derive_draft_zero_ratio_raises(preprocessor):
    """Test that zero or negative ratio raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        preprocessor.derive_draft(np.array([0.0]))

    with pytest.raises(ValueError, match="must be positive"):
        preprocessor.derive_draft(np.array([-1.0]))


def test_create_vt_surface_columns(preprocessor, sample_yacht_data):
    """Test that create_vt_surface returns DataFrame with correct columns."""
    df_vt = preprocessor.create_vt_surface(sample_yacht_data)

    # Check column names
    assert list(df_vt.columns) == ['V', 'T', 'R'], \
        f"Expected columns ['V', 'T', 'R'], got {list(df_vt.columns)}"

    # Check shape
    assert df_vt.shape == (5, 3), f"Expected shape (5, 3), got {df_vt.shape}"


def test_create_vt_surface_missing_columns(preprocessor):
    """Test that missing required columns raises KeyError."""
    incomplete_data = pd.DataFrame({
        'Froude_number': [0.3, 0.4],
        # Missing Beam_draught_ratio and Residuary_resistance
    })

    with pytest.raises(KeyError, match="Missing required columns"):
        preprocessor.create_vt_surface(incomplete_data)


def test_train_test_split_sizes(preprocessor, sample_yacht_data):
    """Test that train-test split produces correct sizes."""
    df_vt = preprocessor.create_vt_surface(sample_yacht_data)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        df_vt, test_size=0.2, random_state=42
    )

    # With 5 samples and test_size=0.2, expect 4 train and 1 test
    assert len(X_train) == 4, f"Expected 4 training samples, got {len(X_train)}"
    assert len(X_test) == 1, f"Expected 1 test sample, got {len(X_test)}"
    assert len(y_train) == 4, f"Expected 4 training labels, got {len(y_train)}"
    assert len(y_test) == 1, f"Expected 1 test label, got {len(y_test)}"


def test_train_test_split_reproducibility(preprocessor, sample_yacht_data):
    """Test that same random seed produces same split."""
    df_vt = preprocessor.create_vt_surface(sample_yacht_data)

    # First split
    X_train1, X_test1, y_train1, y_test1 = preprocessor.train_test_split(
        df_vt, test_size=0.2, random_state=42
    )

    # Second split with same seed
    X_train2, X_test2, y_train2, y_test2 = preprocessor.train_test_split(
        df_vt, test_size=0.2, random_state=42
    )

    # Check that splits are identical
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_train_test_no_overlap(preprocessor, sample_yacht_data):
    """Test that train and test sets have no overlapping indices."""
    df_vt = preprocessor.create_vt_surface(sample_yacht_data)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        df_vt, test_size=0.2, random_state=42
    )

    # Check no index overlap
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    overlap = train_indices.intersection(test_indices)

    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"


def test_get_domain_bounds(preprocessor, sample_yacht_data):
    """Test domain bounds calculation."""
    df_vt = preprocessor.create_vt_surface(sample_yacht_data)
    bounds = preprocessor.get_domain_bounds(df_vt)

    # Check that all required keys are present
    assert 'V_min' in bounds
    assert 'V_max' in bounds
    assert 'T_min' in bounds
    assert 'T_max' in bounds

    # Check that min < max
    assert bounds['V_min'] < bounds['V_max']
    assert bounds['T_min'] < bounds['T_max']


def test_normalize_features(preprocessor, sample_yacht_data):
    """Test feature normalization to [0, 1] range."""
    df_vt = preprocessor.create_vt_surface(sample_yacht_data)
    X = df_vt[['V', 'T']]

    X_norm, params = preprocessor.normalize_features(X)

    # Check that normalized features are in [0, 1]
    assert X_norm['V'].min() >= 0.0
    assert X_norm['V'].max() <= 1.0
    assert X_norm['T'].min() >= 0.0
    assert X_norm['T'].max() <= 1.0

    # For single dataset normalization, min should be 0 and max should be 1
    assert np.isclose(X_norm['V'].min(), 0.0)
    assert np.isclose(X_norm['V'].max(), 1.0)
    assert np.isclose(X_norm['T'].min(), 0.0)
    assert np.isclose(X_norm['T'].max(), 1.0)


def test_denormalize_features(preprocessor, sample_yacht_data):
    """Test that denormalization recovers original features."""
    df_vt = preprocessor.create_vt_surface(sample_yacht_data)
    X_original = df_vt[['V', 'T']]

    # Normalize then denormalize
    X_norm, params = preprocessor.normalize_features(X_original)
    X_recovered = preprocessor.denormalize_features(X_norm, params)

    # Check that recovered values match original
    pd.testing.assert_frame_equal(X_original, X_recovered, rtol=1e-5)


def test_normalize_with_fit_on(preprocessor):
    """Test normalization using separate fit dataset."""
    # Create train and test data
    train_data = pd.DataFrame({
        'V': [10.0, 15.0, 20.0],
        'T': [5.0, 7.0, 9.0]
    })

    test_data = pd.DataFrame({
        'V': [12.0, 18.0],
        'T': [6.0, 8.0]
    })

    # Normalize train data
    train_norm, params = preprocessor.normalize_features(train_data)

    # Normalize test data using train parameters
    test_norm, _ = preprocessor.normalize_features(test_data, fit_on=train_data)

    # Test data should use same min/max from train data
    assert test_norm['V'].min() > 0.0  # Not necessarily 0
    assert test_norm['V'].max() < 1.0  # Not necessarily 1

    # Check that parameters match train data
    assert params['V_min'] == train_data['V'].min()
    assert params['V_max'] == train_data['V'].max()
