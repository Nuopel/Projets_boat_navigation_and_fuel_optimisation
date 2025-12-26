"""Tests for YachtDataLoader class."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import YachtDataLoader


@pytest.fixture
def yacht_data_path():
    """Provide path to yacht hydrodynamics data file."""
    return "yacht_hydro.xls"


@pytest.fixture
def loader(yacht_data_path):
    """Create YachtDataLoader instance."""
    return YachtDataLoader(yacht_data_path)


def test_load_yacht_data_shape(loader):
    """Test that loaded data has correct shape (308 rows, 7 columns)."""
    df = loader.load()
    assert df.shape == (308, 7), f"Expected shape (308, 7), got {df.shape}"


def test_load_yacht_data_columns(loader):
    """Test that loaded data has expected column names."""
    df = loader.load()
    expected_columns = [
        'Longitudinal_position',
        'Prismatic_coefficient',
        'Length_displacement_ratio',
        'Beam_draught_ratio',
        'Length_beam_ratio',
        'Froude_number',
        'Residuary_resistance'
    ]
    assert list(df.columns) == expected_columns, \
        f"Column names don't match. Got: {list(df.columns)}"


def test_load_yacht_data_no_nulls(loader):
    """Test that loaded data has no missing values."""
    df = loader.load()
    null_count = df.isnull().sum().sum()
    assert null_count == 0, f"Found {null_count} null values in dataset"


def test_load_yacht_data_target_positive(loader):
    """Test that all residuary resistance values are positive."""
    df = loader.load()
    resistance = df['Residuary_resistance']
    assert (resistance > 0).all(), \
        f"Found non-positive resistance values. Min: {resistance.min()}"


def test_load_invalid_filepath():
    """Test that FileNotFoundError is raised for invalid filepath."""
    loader = YachtDataLoader("nonexistent_file.xls")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_get_feature_names(loader):
    """Test that feature names exclude target column."""
    loader.load()
    features = loader.get_feature_names()
    assert len(features) == 6, f"Expected 6 features, got {len(features)}"
    assert 'Residuary_resistance' not in features, \
        "Target column should not be in feature names"


def test_get_target_name(loader):
    """Test that target name is correct."""
    target = loader.get_target_name()
    assert target == 'Residuary_resistance', \
        f"Expected 'Residuary_resistance', got '{target}'"


def test_get_summary_statistics(loader):
    """Test that summary statistics can be computed."""
    loader.load()
    summary = loader.get_summary_statistics()

    # Check that summary includes standard statistics
    assert 'mean' in summary.index
    assert 'std' in summary.index
    assert 'min' in summary.index
    assert 'max' in summary.index

    # Check that all columns are included
    assert summary.shape[1] == 7


def test_validate_method(loader):
    """Test that validate method works correctly."""
    # Before loading, validation should fail
    assert loader.validate() == False

    # After loading, validation should pass
    loader.load()
    assert loader.validate() == True


def test_data_types_numeric(loader):
    """Test that all columns are numeric types."""
    df = loader.load()
    for col in df.columns:
        assert np.issubdtype(df[col].dtype, np.number), \
            f"Column {col} is not numeric: {df[col].dtype}"
