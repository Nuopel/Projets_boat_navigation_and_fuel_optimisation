"""
Unit tests for weather field generation and interpolation.

Tests WeatherField class, zone creation, and scenario builders.
"""

import pytest
import numpy as np
from src.models.weather_field import (
    WeatherZone, WeatherField,
    create_calm_scenario, create_storm_scenario, create_multi_zone_scenario
)
from src.utils.geometry import Point


class TestWeatherZone:
    """Test WeatherZone dataclass."""

    def test_create_weather_zone(self):
        """Test creating a WeatherZone."""
        zone = WeatherZone(
            center=Point(100.0, 200.0),
            radius=50.0,
            wind_speed=25.0,
            wave_height=5.0,
            intensity=0.8
        )

        assert zone.center == Point(100.0, 200.0)
        assert zone.radius == 50.0
        assert zone.wind_speed == 25.0
        assert zone.wave_height == 5.0
        assert zone.intensity == 0.8

    def test_default_intensity(self):
        """Test that intensity defaults to 1.0."""
        zone = WeatherZone(
            center=Point(0.0, 0.0),
            radius=10.0,
            wind_speed=15.0,
            wave_height=3.0
        )

        assert zone.intensity == 1.0


class TestWeatherFieldInitialization:
    """Test WeatherField initialization."""

    def test_create_weather_field(self):
        """Test creating a WeatherField."""
        grid_shape = (10, 10)
        cell_size = 5.0

        weather = WeatherField(grid_shape, cell_size)

        assert weather.grid_shape == grid_shape
        assert weather.cell_size == cell_size
        assert weather.origin == Point(0, 0)

        # Fields should be initialized to zero (calm)
        assert weather.wind_field.shape == grid_shape
        assert weather.wave_field.shape == grid_shape
        assert np.all(weather.wind_field == 0.0)
        assert np.all(weather.wave_field == 0.0)

    def test_custom_origin(self):
        """Test creating WeatherField with custom origin."""
        weather = WeatherField((5, 5), 10.0, origin=Point(100.0, 200.0))

        assert weather.origin == Point(100.0, 200.0)


class TestWeatherFieldBaseConditions:
    """Test adding base weather conditions."""

    def test_add_base_conditions(self):
        """Test setting baseline weather."""
        weather = WeatherField((10, 10), 5.0)
        weather.add_base_conditions(wind_speed=12.0, wave_height=2.5)

        # All cells should have base values
        assert np.all(weather.wind_field == 12.0)
        assert np.all(weather.wave_field == 2.5)

    def test_add_base_conditions_accumulate(self):
        """Test that base conditions add to existing values."""
        weather = WeatherField((10, 10), 5.0)
        weather.add_base_conditions(wind_speed=10.0, wave_height=1.0)
        weather.add_base_conditions(wind_speed=5.0, wave_height=0.5)

        # Should accumulate
        assert np.all(weather.wind_field == 15.0)
        assert np.all(weather.wave_field == 1.5)


class TestWeatherFieldZones:
    """Test adding weather zones."""

    def test_add_weather_zone_at_center(self):
        """Test that weather zone has maximum effect at center."""
        grid_shape = (20, 20)
        cell_size = 10.0
        weather = WeatherField(grid_shape, cell_size)

        # Zone at grid center
        zone = WeatherZone(
            center=Point(100.0, 100.0),  # Center of 20x20 grid with 10nm cells
            radius=50.0,
            wind_speed=30.0,
            wave_height=6.0,
            intensity=1.0
        )

        weather.add_weather_zone(zone)

        # Get weather at zone center
        wind, wave = weather.get_weather_at_point(Point(100.0, 100.0))

        # Should be close to maximum (with Gaussian decay, exact center may not be exactly 30.0)
        assert wind > 25.0
        assert wave > 5.0

    def test_weather_zone_decays_with_distance(self):
        """Test that weather zone effect decreases with distance."""
        weather = WeatherField((20, 20), 10.0)

        zone = WeatherZone(
            center=Point(100.0, 100.0),
            radius=50.0,
            wind_speed=30.0,
            wave_height=6.0
        )

        weather.add_weather_zone(zone)

        # Sample at different distances
        wind_center, wave_center = weather.get_weather_at_point(Point(100.0, 100.0))
        wind_near, wave_near = weather.get_weather_at_point(Point(120.0, 100.0))  # 20 nm away
        wind_far, wave_far = weather.get_weather_at_point(Point(180.0, 100.0))  # 80 nm away

        # Weather should decay with distance
        assert wind_center > wind_near > wind_far
        assert wave_center > wave_near > wave_far

    def test_blend_mode_max(self):
        """Test that blend mode takes maximum value."""
        weather = WeatherField((20, 20), 10.0)

        # Add two overlapping zones
        zone1 = WeatherZone(
            center=Point(100.0, 100.0),
            radius=60.0,
            wind_speed=20.0,
            wave_height=3.0
        )
        zone2 = WeatherZone(
            center=Point(120.0, 100.0),
            radius=60.0,
            wind_speed=30.0,
            wave_height=5.0
        )

        weather.add_weather_zone(zone1, blend=True)
        weather.add_weather_zone(zone2, blend=True)

        # At overlap point, should have higher values from zone2
        wind, wave = weather.get_weather_at_point(Point(110.0, 100.0))

        # Should be closer to zone2's higher values
        assert wind > 20.0
        assert wave > 3.0


class TestWeatherFieldInterpolation:
    """Test weather interpolation at arbitrary points."""

    def test_get_weather_at_grid_point(self):
        """Test getting weather at exact grid point."""
        weather = WeatherField((10, 10), 10.0)
        weather.add_base_conditions(wind_speed=15.0, wave_height=3.0)

        # Grid point (5, 5) -> (50, 50) nm
        point = Point(50.0, 50.0)
        wind, wave = weather.get_weather_at_point(point)

        assert pytest.approx(wind, rel=0.01) == 15.0
        assert pytest.approx(wave, rel=0.01) == 3.0

    def test_bilinear_interpolation(self):
        """Test that interpolation works between grid points."""
        weather = WeatherField((10, 10), 10.0)

        # Set specific values at corners of a cell
        weather.wind_field[0, 0] = 10.0
        weather.wind_field[0, 1] = 20.0
        weather.wind_field[1, 0] = 15.0
        weather.wind_field[1, 1] = 25.0

        # Query point at center of cell (0.5, 0.5) in grid coords = (5, 5) nm
        wind, _ = weather.get_weather_at_point(Point(5.0, 5.0))

        # Bilinear interpolation: (10+20+15+25)/4 = 17.5
        assert pytest.approx(wind, rel=0.1) == 17.5

    def test_weather_clamped_to_bounds(self):
        """Test that out-of-bounds points are clamped."""
        weather = WeatherField((10, 10), 10.0)
        weather.add_base_conditions(wind_speed=12.0, wave_height=2.0)

        # Point outside grid bounds
        point = Point(200.0, 200.0)
        wind, wave = weather.get_weather_at_point(point)

        # Should return edge values (not crash)
        assert wind >= 0
        assert wave >= 0


class TestWeatherFieldSmoothing:
    """Test Gaussian smoothing."""

    def test_smooth_field(self):
        """Test that smoothing reduces sharp gradients."""
        weather = WeatherField((20, 20), 10.0)

        # Create sharp spike
        weather.wind_field[10, 10] = 100.0

        # Get values before smoothing
        val_before = weather.wind_field[10, 10]
        neighbor_before = weather.wind_field[10, 11]

        # Smooth
        weather.smooth_field(sigma=2.0)

        # Get values after smoothing
        val_after = weather.wind_field[10, 10]
        neighbor_after = weather.wind_field[10, 11]

        # Peak should be reduced
        assert val_after < val_before

        # Neighbor should increase (smoothing spreads values)
        assert neighbor_after > neighbor_before


class TestStormZoneDetection:
    """Test storm zone identification."""

    def test_is_storm_zone(self):
        """Test identifying storm zones."""
        weather = WeatherField((20, 20), 10.0)

        # Add calm base
        weather.add_base_conditions(wind_speed=10.0, wave_height=2.0)

        # Add storm
        storm = WeatherZone(
            center=Point(100.0, 100.0),
            radius=40.0,
            wind_speed=35.0,
            wave_height=7.0
        )
        weather.add_weather_zone(storm)

        # Test points
        assert weather.is_storm_zone(Point(100.0, 100.0), wave_threshold=5.0) is True
        assert weather.is_storm_zone(Point(200.0, 200.0), wave_threshold=5.0) is False


class TestWeatherFieldStatistics:
    """Test weather field statistics."""

    def test_get_field_statistics(self):
        """Test calculating field statistics."""
        weather = WeatherField((20, 20), 10.0)
        weather.add_base_conditions(wind_speed=15.0, wave_height=3.0)

        stats = weather.get_field_statistics()

        assert 'wind' in stats
        assert 'waves' in stats

        # Check wind stats
        assert pytest.approx(stats['wind']['mean'], rel=0.01) == 15.0
        assert pytest.approx(stats['wind']['min'], rel=0.01) == 15.0
        assert pytest.approx(stats['wind']['max'], rel=0.01) == 15.0
        assert pytest.approx(stats['wind']['std'], abs=0.1) == 0.0  # Uniform field

        # Check wave stats
        assert pytest.approx(stats['waves']['mean'], rel=0.01) == 3.0


class TestScenarioBuilders:
    """Test scenario creation functions."""

    def test_create_calm_scenario(self):
        """Test calm scenario creation."""
        grid_shape = (30, 30)
        cell_size = 10.0

        weather = create_calm_scenario(grid_shape, cell_size)

        assert weather.grid_shape == grid_shape
        assert weather.cell_size == cell_size

        # Should have calm conditions
        stats = weather.get_field_statistics()
        assert stats['wind']['mean'] > 0
        assert stats['wind']['mean'] < 20.0  # Calm winds
        assert stats['waves']['mean'] < 3.0  # Low waves

    def test_create_storm_scenario(self):
        """Test storm scenario creation."""
        grid_shape = (30, 30)
        cell_size = 10.0
        storm_center = Point(150.0, 150.0)

        weather = create_storm_scenario(grid_shape, cell_size, storm_center)

        assert weather.grid_shape == grid_shape

        # Check storm at center
        wind, wave = weather.get_weather_at_point(storm_center)
        assert wind > 20.0  # Stormy winds
        assert wave > 4.0  # High waves

        # Check calmer at edge
        wind_edge, wave_edge = weather.get_weather_at_point(Point(10.0, 10.0))
        assert wind_edge < wind
        assert wave_edge < wave

    def test_create_multi_zone_scenario(self):
        """Test multi-zone scenario creation."""
        grid_shape = (40, 40)
        cell_size = 10.0

        zones = [
            WeatherZone(Point(100.0, 100.0), 50.0, 20.0, 4.0),
            WeatherZone(Point(300.0, 300.0), 60.0, 25.0, 5.0)
        ]

        weather = create_multi_zone_scenario(grid_shape, cell_size, zones)

        assert weather.grid_shape == grid_shape

        # Check that zones have effect
        wind1, wave1 = weather.get_weather_at_point(Point(100.0, 100.0))
        wind2, wave2 = weather.get_weather_at_point(Point(300.0, 300.0))

        # Both zones should increase weather
        assert wind1 > 12.0  # Above base
        assert wind2 > 12.0
        assert wave1 > 2.0
        assert wave2 > 2.0


class TestWeatherFieldEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_sigma_smoothing(self):
        """Test that zero sigma doesn't crash."""
        weather = WeatherField((10, 10), 10.0)
        weather.add_base_conditions(15.0, 3.0)

        # Should not crash
        weather.smooth_field(sigma=0.0)

    def test_large_smoothing(self):
        """Test that large sigma smooths heavily."""
        weather = WeatherField((20, 20), 10.0)
        weather.wind_field[10, 10] = 100.0

        weather.smooth_field(sigma=10.0)

        # Heavy smoothing should spread values widely
        # Peak should be significantly reduced
        assert weather.wind_field[10, 10] < 50.0
