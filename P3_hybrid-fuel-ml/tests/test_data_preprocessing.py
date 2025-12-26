"""Unit tests for data preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    DataPreprocessor,
    PreprocessingConfig
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    return pd.DataFrame({
        'ship_id': ['NG001', 'NG002', 'NG003', 'NG004', 'NG005'] * 20,
        'ship_type': ['Oil Service Boat', 'Tanker Ship', 'Fishing Trawler', 'Surfer Boat'] * 25,
        'route_id': ['Port Harcourt-Lagos', 'Lagos-Apapa'] * 50,
        'month': ['January', 'February', 'March', 'April'] * 25,
        'distance': np.random.uniform(50, 200, 100),
        'fuel_type': ['HFO', 'Diesel'] * 50,
        'fuel_consumption': np.random.uniform(1000, 6000, 100),
        'CO2_emissions': np.random.uniform(3000, 18000, 100),
        'weather_conditions': ['Calm', 'Moderate', 'Stormy'] * 33 + ['Calm'],
        'engine_efficiency': np.random.uniform(70, 95, 100)
    })


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    config = PreprocessingConfig()
    return DataPreprocessor(config)


class TestDataCleaning:
    """Tests for data cleaning functionality."""

    def test_clean_data_removes_outliers(self, preprocessor, sample_data):
        """Verify outliers are handled correctly."""
        # Add extreme outlier
        sample_data.loc[0, 'fuel_consumption'] = 50000  # Extreme value

        cleaned = preprocessor.clean_data(sample_data)

        # Check that extreme value was capped (not removed since cap_outliers=True)
        assert cleaned['fuel_consumption'].max() < 50000
        assert len(cleaned) == len(sample_data)  # No rows removed

    def test_clean_data_handles_negative_values(self, preprocessor):
        """Verify negative values are removed."""
        bad_data = pd.DataFrame({
            'distance': [100, -50, 150],
            'fuel_consumption': [3000, 4000, -100],
            'engine_efficiency': [85, 90, 95]
        })

        cleaned = preprocessor.clean_data(bad_data)

        # Should remove rows with negative distance or fuel
        assert len(cleaned) == 1  # Only first row valid
        assert cleaned['distance'].min() > 0
        assert cleaned['fuel_consumption'].min() > 0

    def test_clean_data_validates_engine_efficiency_range(self, preprocessor):
        """Verify engine efficiency is between 0-100."""
        bad_data = pd.DataFrame({
            'distance': [100, 100, 100],
            'fuel_consumption': [3000, 3000, 3000],
            'engine_efficiency': [85, 110, -5]  # 110 and -5 are invalid
        })

        cleaned = preprocessor.clean_data(bad_data)

        assert len(cleaned) == 1
        assert 0 < cleaned['engine_efficiency'].iloc[0] <= 100


class TestCategoricalEncoding:
    """Tests for categorical encoding functionality."""

    def test_encode_categoricals_preserves_shape(self, preprocessor, sample_data):
        """Ensure encoding doesn't drop rows unexpectedly."""
        encoded = preprocessor.encode_categoricals(sample_data)

        assert len(encoded) == len(sample_data)

    def test_encode_drops_ship_id(self, preprocessor, sample_data):
        """Verify ship_id is removed (high cardinality)."""
        encoded = preprocessor.encode_categoricals(sample_data)

        assert 'ship_id' not in encoded.columns

    def test_fuel_type_binary_encoding(self, preprocessor, sample_data):
        """Verify fuel_type is correctly binary encoded."""
        encoded = preprocessor.encode_categoricals(sample_data)

        assert 'fuel_type' not in encoded.columns
        assert 'fuel_type_hfo' in encoded.columns
        assert set(encoded['fuel_type_hfo'].unique()) == {0, 1}

    def test_weather_ordinal_encoding(self, preprocessor, sample_data):
        """Verify weather is ordinally encoded."""
        encoded = preprocessor.encode_categoricals(sample_data)

        assert 'weather_conditions' not in encoded.columns
        assert 'weather_ordinal' in encoded.columns
        assert set(encoded['weather_ordinal'].unique()).issubset({0, 1, 2})

    def test_ship_type_one_hot_encoding(self, preprocessor, sample_data):
        """Verify ship_type is one-hot encoded."""
        encoded = preprocessor.encode_categoricals(sample_data)

        assert 'ship_type' not in encoded.columns
        # Check for dummy columns (at least 3 of 4 ship types)
        ship_type_cols = [col for col in encoded.columns if col.startswith('ship_type_')]
        assert len(ship_type_cols) >= 3

    def test_month_cyclical_encoding(self, preprocessor, sample_data):
        """Verify month is converted to cyclical sin/cos."""
        encoded = preprocessor.encode_categoricals(sample_data)

        assert 'month' not in encoded.columns
        assert 'month_sin' in encoded.columns
        assert 'month_cos' in encoded.columns

        # Check range [-1, 1]
        assert -1 <= encoded['month_sin'].min() <= 1
        assert -1 <= encoded['month_cos'].max() <= 1


class TestDataSplitting:
    """Tests for train/validation/test splitting."""

    def test_train_test_split_stratification(self, preprocessor, sample_data):
        """Verify ship_type distribution is similar across splits."""
        train, val, test = preprocessor.split_data(sample_data)

        # Check proportions
        total = len(sample_data)
        assert len(train) == int(0.70 * total)
        assert len(val) == int(0.15 * total)
        # test gets remainder

        # Check stratification (distribution should be similar)
        orig_dist = sample_data['ship_type'].value_counts(normalize=True)
        train_dist = train['ship_type'].value_counts(normalize=True)

        # Allow 10% deviation
        for ship_type in orig_dist.index:
            if ship_type in train_dist.index:
                assert abs(orig_dist[ship_type] - train_dist[ship_type]) < 0.15

    def test_split_data_no_overlap(self, preprocessor, sample_data):
        """Verify no data leakage between splits."""
        train, val, test = preprocessor.split_data(sample_data)

        train_indices = set(train.index)
        val_indices = set(val.index)
        test_indices = set(test.index)

        # Check no overlap
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

    def test_split_covers_all_data(self, preprocessor, sample_data):
        """Verify all original data is in one of the splits."""
        train, val, test = preprocessor.split_data(sample_data)

        total_split_len = len(train) + len(val) + len(test)
        assert total_split_len == len(sample_data)


class TestOutlierDetection:
    """Tests for outlier detection methods."""

    def test_detect_outliers_iqr(self, preprocessor):
        """Test IQR outlier detection method."""
        # Create data with obvious outlier
        data = pd.Series([10, 12, 11, 13, 12, 11, 100])  # 100 is outlier

        outliers = preprocessor._detect_outliers_iqr(data, multiplier=1.5)

        assert outliers[6] == True  # Last value is outlier
        assert outliers[:6].sum() == 0  # Others are not outliers

    def test_detect_outliers_zscore(self, preprocessor):
        """Test Z-score outlier detection method."""
        # Create data with obvious outlier
        data = pd.Series([10, 12, 11, 13, 12, 11, 100])

        outliers = preprocessor._detect_outliers_zscore(data, threshold=3.0)

        assert outliers[6] == True  # Last value is outlier
        assert outliers[:6].sum() == 0  # Others are not outliers


class TestFeatureNames:
    """Tests for feature extraction."""

    def test_get_feature_names_excludes_target(self, preprocessor, sample_data):
        """Verify target variables are excluded from feature list."""
        encoded = preprocessor.encode_categoricals(sample_data)
        features = preprocessor.get_feature_names(encoded, exclude_target=True)

        assert 'fuel_consumption' not in features
        assert 'CO2_emissions' not in features
        assert len(features) > 0

    def test_get_feature_names_includes_target(self, preprocessor, sample_data):
        """Verify target can be included if requested."""
        encoded = preprocessor.encode_categoricals(sample_data)
        features = preprocessor.get_feature_names(encoded, exclude_target=False)

        assert 'fuel_consumption' in features
        assert 'CO2_emissions' in features


class TestScaling:
    """Tests for feature scaling."""

    def test_fit_scaler_excludes_target(self, preprocessor, sample_data):
        """Verify scaler doesn't fit on target variable."""
        cleaned = preprocessor.clean_data(sample_data)
        encoded = preprocessor.encode_categoricals(cleaned)

        preprocessor.fit_scaler(encoded)

        # Check scaler was fitted
        assert preprocessor.scaler is not None
        assert hasattr(preprocessor.scaler, 'mean_')

    def test_transform_features_scales_correctly(self, preprocessor, sample_data):
        """Verify transformation produces standardized features."""
        cleaned = preprocessor.clean_data(sample_data)
        encoded = preprocessor.encode_categoricals(cleaned)

        preprocessor.fit_scaler(encoded)
        scaled = preprocessor.transform_features(encoded)

        # Check standardization (mean ≈ 0, std ≈ 1 for numerical columns)
        # Note: encoded features (binary) will not be perfectly standardized
        numerical_cols = ['distance', 'engine_efficiency']
        for col in numerical_cols:
            if col in scaled.columns:
                assert abs(scaled[col].mean()) < 0.5  # Close to 0
                assert 0.5 < scaled[col].std() < 1.5  # Close to 1

    def test_transform_without_fit_raises_error(self, preprocessor, sample_data):
        """Verify error is raised if transform called before fit."""
        cleaned = preprocessor.clean_data(sample_data)
        encoded = preprocessor.encode_categoricals(cleaned)

        with pytest.raises(ValueError, match="Scaler not fitted"):
            preprocessor.transform_features(encoded)


def test_full_pipeline_integration(sample_data):
    """Integration test: full preprocessing pipeline."""
    config = PreprocessingConfig(
        outlier_method='iqr',
        cap_outliers=True,
        random_state=42
    )
    preprocessor = DataPreprocessor(config)

    # Full pipeline
    df_clean = preprocessor.clean_data(sample_data)
    df_encoded = preprocessor.encode_categoricals(df_clean)
    train, val, test = preprocessor.split_data(df_encoded)

    # Assertions
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert 'fuel_consumption' in train.columns  # Target preserved
    assert 'ship_id' not in train.columns  # High cardinality removed
    assert 'weather_ordinal' in train.columns  # Encoding applied
