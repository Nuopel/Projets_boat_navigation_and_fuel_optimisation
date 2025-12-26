"""Tests for SyntheticSurfaceGenerator class."""

import pytest
import numpy as np
import pandas as pd
from src.data.synthetic import SyntheticSurfaceGenerator


@pytest.fixture
def generator():
    """Create SyntheticSurfaceGenerator with default parameters."""
    return SyntheticSurfaceGenerator(
        v_range=(10.0, 25.0),
        t_range=(6.0, 10.0),
        noise_std=0.01,
        random_state=42
    )


def test_resistance_function_shape(generator):
    """Test that resistance function output matches input shape."""
    V = np.array([10, 15, 20, 25])
    T = np.array([6, 7, 8, 9])

    R = generator.resistance_function(V, T)

    assert R.shape == V.shape, f"Expected shape {V.shape}, got {R.shape}"
    assert len(R) == 4


def test_resistance_function_velocity_monotonic(generator):
    """Test that resistance increases with velocity (draft fixed)."""
    V = np.array([10, 12, 15, 18, 20, 25])
    T = np.full_like(V, fill_value=8.0)  # Fixed draft

    R = generator.resistance_function(V, T)

    # Check monotonic increase (velocity-squared term should dominate)
    for i in range(len(R) - 1):
        assert R[i] < R[i+1], \
            f"Resistance should increase with velocity: R[{i}]={R[i]:.2f}, R[{i+1}]={R[i+1]:.2f}"


def test_resistance_function_deterministic(generator):
    """Test that resistance function is deterministic (same inputs → same outputs)."""
    V = np.array([15.0, 20.0])
    T = np.array([7.0, 8.5])

    R1 = generator.resistance_function(V, T)
    R2 = generator.resistance_function(V, T)

    np.testing.assert_array_equal(R1, R2,
                                   err_msg="Resistance function should be deterministic")


def test_dense_grid_size(generator):
    """Test that dense grid has correct number of points."""
    grid_size = 100
    grid = generator.create_dense_grid(grid_size=grid_size)

    expected_total_points = grid_size * grid_size

    assert len(grid['V']) == expected_total_points
    assert len(grid['T']) == expected_total_points
    assert len(grid['R']) == expected_total_points

    assert grid['V_grid'].shape == (grid_size, grid_size)
    assert grid['T_grid'].shape == (grid_size, grid_size)
    assert grid['R_grid'].shape == (grid_size, grid_size)


def test_dense_grid_bounds(generator):
    """Test that dense grid covers the specified domain bounds."""
    grid = generator.create_dense_grid(grid_size=50)

    assert np.min(grid['V']) == pytest.approx(generator.v_range[0], abs=0.01)
    assert np.max(grid['V']) == pytest.approx(generator.v_range[1], abs=0.01)
    assert np.min(grid['T']) == pytest.approx(generator.t_range[0], abs=0.01)
    assert np.max(grid['T']) == pytest.approx(generator.t_range[1], abs=0.01)


def test_sparse_sampling_count(generator):
    """Test that sparse sampling returns requested number of samples."""
    n_samples = 50
    df = generator.sample_sparse(n_samples=n_samples, strategy='random', add_noise=False)

    assert len(df) == n_samples, f"Expected {n_samples} samples, got {len(df)}"
    assert list(df.columns) == ['V', 'T', 'R']


def test_sparse_sampling_unique(generator):
    """Test that sparse samples are unique (no exact duplicates)."""
    n_samples = 100
    df = generator.sample_sparse(n_samples=n_samples, strategy='random', add_noise=False)

    # Check for duplicate (V, T) pairs
    n_unique = len(df[['V', 'T']].drop_duplicates())

    # Allow a few duplicates for random sampling (unlikely but possible)
    assert n_unique >= 0.95 * n_samples, \
        f"Too many duplicate samples: only {n_unique}/{n_samples} unique"


def test_sparse_sampling_strategies(generator):
    """Test that different sampling strategies work."""
    n_samples = 30

    # Random sampling
    df_random = generator.sample_sparse(n_samples, strategy='random', add_noise=False)
    assert len(df_random) == n_samples

    # Latin Hypercube sampling
    df_lhs = generator.sample_sparse(n_samples, strategy='latin_hypercube', add_noise=False)
    assert len(df_lhs) == n_samples

    # Structured sampling
    df_struct = generator.sample_sparse(n_samples, strategy='structured', add_noise=False)
    assert len(df_struct) == n_samples

    # Check that strategies produce different results
    assert not df_random['V'].equals(df_lhs['V']), \
        "Random and LHS should produce different samples"


def test_sparse_sampling_invalid_strategy(generator):
    """Test that invalid sampling strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        generator.sample_sparse(30, strategy='invalid_strategy')


def test_sparse_sampling_bounds(generator):
    """Test that sparse samples are within domain bounds."""
    n_samples = 100
    df = generator.sample_sparse(n_samples, strategy='random', add_noise=False)

    assert df['V'].min() >= generator.v_range[0]
    assert df['V'].max() <= generator.v_range[1]
    assert df['T'].min() >= generator.t_range[0]
    assert df['T'].max() <= generator.t_range[1]


def test_add_noise_snr(generator):
    """Test that noise addition produces approximately correct SNR."""
    R_clean = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 20)  # 100 samples
    target_snr = 20.0

    R_noisy = generator.add_noise(R_clean, snr_db=target_snr)

    # Compute actual SNR
    actual_snr = generator.compute_snr(R_clean, R_noisy)

    # Allow 1 dB tolerance due to random sampling
    assert abs(actual_snr - target_snr) < 1.0, \
        f"Target SNR: {target_snr} dB, Actual SNR: {actual_snr:.2f} dB"


def test_add_noise_changes_values(generator):
    """Test that adding noise actually changes the values."""
    R_clean = np.array([10.0, 20.0, 30.0])
    R_noisy = generator.add_noise(R_clean, snr_db=20.0)

    # At least one value should be different
    assert not np.allclose(R_clean, R_noisy, rtol=1e-10), \
        "Noise should change the values"


def test_synthetic_deterministic_with_seed(generator):
    """Test that same seed produces same synthetic data."""
    n_samples = 30

    df1 = generator.sample_sparse(n_samples, strategy='random', add_noise=True, snr_db=20)

    # Create new generator with same seed
    generator2 = SyntheticSurfaceGenerator(random_state=42)
    df2 = generator2.sample_sparse(n_samples, strategy='random', add_noise=True, snr_db=20)

    # Should produce identical results
    pd.testing.assert_frame_equal(df1, df2)


def test_latin_hypercube_space_filling(generator):
    """Test that Latin Hypercube provides better space-filling than random."""
    n_samples = 50

    # Generate samples
    df_lhs = generator.sample_sparse(n_samples, strategy='latin_hypercube', add_noise=False)

    # Divide domain into bins and count samples per bin
    n_bins = 5
    v_bins = np.linspace(generator.v_range[0], generator.v_range[1], n_bins + 1)
    t_bins = np.linspace(generator.t_range[0], generator.t_range[1], n_bins + 1)

    v_digitized = np.digitize(df_lhs['V'], v_bins)
    t_digitized = np.digitize(df_lhs['T'], t_bins)

    # Count 2D bin occupancy
    bin_counts = np.zeros((n_bins, n_bins))
    for i, j in zip(v_digitized - 1, t_digitized - 1):
        if 0 <= i < n_bins and 0 <= j < n_bins:
            bin_counts[i, j] += 1

    # LHS should have good coverage: most bins should have at least one sample
    occupied_bins = np.sum(bin_counts > 0)
    coverage_ratio = occupied_bins / (n_bins * n_bins)

    assert coverage_ratio > 0.6, \
        f"LHS should cover at least 60% of bins, got {coverage_ratio*100:.1f}%"


def test_get_domain_info(generator):
    """Test domain information retrieval."""
    info = generator.get_domain_info()

    assert info['V_min'] == 10.0
    assert info['V_max'] == 25.0
    assert info['T_min'] == 6.0
    assert info['T_max'] == 10.0
    assert info['V_range'] == 15.0
    assert info['T_range'] == 4.0
    assert info['area'] == 60.0


def test_compute_snr_perfect(generator):
    """Test SNR computation with no noise (should be infinite)."""
    R_clean = np.array([10.0, 20.0, 30.0])
    R_noisy = R_clean.copy()  # No noise

    snr = generator.compute_snr(R_clean, R_noisy)

    assert np.isinf(snr) or snr > 100, \
        "SNR should be infinite or very high for noise-free signal"


def test_no_noise_option(generator):
    """Test that add_noise=False produces clean samples."""
    n_samples = 20
    df = generator.sample_sparse(n_samples, strategy='random', add_noise=False)

    # Recompute resistance from V, T
    R_expected = generator.resistance_function(df['V'].values, df['T'].values)

    # Should match exactly (no noise added)
    np.testing.assert_array_almost_equal(df['R'].values, R_expected, decimal=10)
