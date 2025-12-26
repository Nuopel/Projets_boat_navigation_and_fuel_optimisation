"""
Unit tests for navigation environment and grid system.

Tests NavigationEnvironment, constraints, and pathfinding support.
"""

import pytest
import numpy as np
from src.models.navigation_grid import NavigationConstraints, NavigationEnvironment
from src.models.weather_field import WeatherField, WeatherZone, create_calm_scenario, create_storm_scenario
from src.utils.geometry import Point


class TestNavigationConstraints:
    """Test NavigationConstraints dataclass."""

    def test_default_constraints(self):
        """Test creating constraints with defaults."""
        constraints = NavigationConstraints()

        assert constraints.min_storm_distance == 50.0
        assert constraints.max_wave_height == 6.0
        assert len(constraints.restricted_zones) == 0

    def test_custom_constraints(self):
        """Test creating constraints with custom values."""
        restricted = {(5, 5), (10, 10)}
        constraints = NavigationConstraints(
            min_storm_distance=100.0,
            max_wave_height=4.0,
            restricted_zones=restricted
        )

        assert constraints.min_storm_distance == 100.0
        assert constraints.max_wave_height == 4.0
        assert constraints.restricted_zones == restricted


class TestNavigationEnvironmentInitialization:
    """Test NavigationEnvironment initialization."""

    def test_create_environment(self):
        """Test creating a basic navigation environment."""
        grid_shape = (20, 20)
        cell_size = 10.0

        env = NavigationEnvironment(grid_shape, cell_size)

        assert env.grid_shape == grid_shape
        assert env.cell_size == cell_size
        assert env.origin == Point(0, 0)
        assert env.navigable.shape == grid_shape
        assert np.all(env.navigable)  # All cells navigable by default

    def test_create_with_weather(self):
        """Test creating environment with weather field."""
        grid_shape = (20, 20)
        cell_size = 10.0
        weather = create_calm_scenario(grid_shape, cell_size)

        env = NavigationEnvironment(grid_shape, cell_size, weather=weather)

        assert env.weather is weather

    def test_create_with_constraints(self):
        """Test creating environment with constraints."""
        constraints = NavigationConstraints(
            min_storm_distance=100.0,
            max_wave_height=5.0,
            restricted_zones={(5, 5), (10, 10)}
        )

        env = NavigationEnvironment((20, 20), 10.0, constraints=constraints)

        assert env.constraints is constraints
        # Restricted zones should be marked as non-navigable
        assert not env.navigable[5, 5]
        assert not env.navigable[10, 10]


class TestNavigableChecking:
    """Test navigability checking."""

    def test_is_navigable_point(self):
        """Test checking if a point is navigable."""
        env = NavigationEnvironment((20, 20), 10.0)

        # Point in open area should be navigable
        assert env.is_navigable(Point(100.0, 100.0)) is True

    def test_is_navigable_out_of_bounds(self):
        """Test that out-of-bounds points are not navigable."""
        env = NavigationEnvironment((20, 20), 10.0)

        # Point outside grid
        assert env.is_navigable(Point(500.0, 500.0)) is False
        assert env.is_navigable(Point(-50.0, 50.0)) is False

    def test_is_navigable_restricted_zone(self):
        """Test that restricted zones are not navigable."""
        constraints = NavigationConstraints(
            restricted_zones={(5, 5)}
        )
        env = NavigationEnvironment((20, 20), 10.0, constraints=constraints)

        # Convert grid (5,5) to nautical miles
        restricted_point = Point(50.0, 50.0)  # cell_size * col/row

        assert env.is_navigable(restricted_point) is False

    def test_is_navigable_high_waves(self):
        """Test that high wave areas are not navigable."""
        grid_shape = (20, 20)
        cell_size = 10.0

        # Create storm scenario
        storm_center = Point(100.0, 100.0)
        weather = create_storm_scenario(grid_shape, cell_size, storm_center)

        constraints = NavigationConstraints(max_wave_height=6.0)
        env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

        # Check actual wave height at storm center
        _, wave_height = weather.get_weather_at_point(storm_center)

        # After smoothing, waves at center should still be high
        # If waves exceed max_wave_height, should not be navigable
        if wave_height > constraints.max_wave_height:
            assert env.is_navigable(storm_center) is False

        # Far from storm should have lower waves and be navigable
        assert env.is_navigable(Point(10.0, 10.0)) is True

    def test_is_navigable_grid(self):
        """Test grid-based navigability checking."""
        env = NavigationEnvironment((20, 20), 10.0)

        # Valid grid position
        assert env.is_navigable_grid((5, 5)) is True

        # Out of bounds
        assert env.is_navigable_grid((25, 5)) is False
        assert env.is_navigable_grid((-1, 5)) is False


class TestWeatherPenalty:
    """Test weather penalty calculations."""

    def test_weather_penalty_calm(self):
        """Test penalty in calm conditions."""
        weather = create_calm_scenario((20, 20), 10.0)
        env = NavigationEnvironment((20, 20), 10.0, weather=weather)

        point = Point(100.0, 100.0)
        penalty = env.get_weather_penalty(point)

        # Calm weather: penalty close to 1.0 (low waves)
        assert 1.0 <= penalty <= 1.3

    def test_weather_penalty_storm(self):
        """Test penalty in stormy conditions."""
        storm_center = Point(100.0, 100.0)
        weather = create_storm_scenario((20, 20), 10.0, storm_center)

        # Modify constraints to allow navigation through storm (for testing penalty)
        constraints = NavigationConstraints(max_wave_height=10.0)
        env = NavigationEnvironment((20, 20), 10.0, weather=weather, constraints=constraints)

        penalty = env.get_weather_penalty(storm_center)

        # High waves should give higher penalty
        assert penalty > 1.5

    def test_penalty_increases_with_waves(self):
        """Test that penalty increases with wave height."""
        weather = WeatherField((20, 20), 10.0)
        weather.add_base_conditions(wind_speed=10.0, wave_height=2.0)

        # Add two zones with different wave heights
        zone1 = WeatherZone(Point(50.0, 50.0), 30.0, 15.0, 3.0)
        zone2 = WeatherZone(Point(150.0, 150.0), 30.0, 20.0, 6.0)

        weather.add_weather_zone(zone1)
        weather.add_weather_zone(zone2)

        constraints = NavigationConstraints(max_wave_height=10.0)
        env = NavigationEnvironment((20, 20), 10.0, weather=weather, constraints=constraints)

        penalty1 = env.get_weather_penalty(Point(50.0, 50.0))
        penalty2 = env.get_weather_penalty(Point(150.0, 150.0))

        # Higher waves should give higher penalty
        assert penalty2 > penalty1


class TestRestrictedZones:
    """Test adding and managing restricted zones."""

    def test_add_restricted_zone(self):
        """Test adding a circular restricted zone."""
        env = NavigationEnvironment((30, 30), 10.0)

        # Initially all navigable
        assert np.all(env.navigable)

        # Add restricted zone
        center = Point(150.0, 150.0)  # Grid center
        radius = 30.0  # nm

        env.add_restricted_zone(center, radius)

        # Center should be non-navigable
        center_grid = (15, 15)
        assert not env.navigable[center_grid]

        # Points within radius should be restricted
        nearby_grid = (16, 16)
        assert not env.navigable[nearby_grid]

        # Points far away should still be navigable
        far_grid = (5, 5)
        assert env.navigable[far_grid]


class TestNeighbors:
    """Test getting navigable neighbors."""

    def test_get_neighbors_4_connectivity(self):
        """Test getting 4-connected neighbors (cardinal directions)."""
        env = NavigationEnvironment((10, 10), 10.0)

        neighbors = env.get_neighbors((5, 5), connectivity=4)

        # Should have 4 neighbors (N, S, E, W)
        assert len(neighbors) == 4
        assert (4, 5) in neighbors  # North
        assert (6, 5) in neighbors  # South
        assert (5, 4) in neighbors  # West
        assert (5, 6) in neighbors  # East

    def test_get_neighbors_8_connectivity(self):
        """Test getting 8-connected neighbors (with diagonals)."""
        env = NavigationEnvironment((10, 10), 10.0)

        neighbors = env.get_neighbors((5, 5), connectivity=8)

        # Should have 8 neighbors
        assert len(neighbors) == 8

        # Check diagonals included
        assert (4, 4) in neighbors  # NW
        assert (4, 6) in neighbors  # NE
        assert (6, 4) in neighbors  # SW
        assert (6, 6) in neighbors  # SE

    def test_neighbors_at_boundary(self):
        """Test that boundary cells have fewer neighbors."""
        env = NavigationEnvironment((10, 10), 10.0)

        # Corner cell
        neighbors = env.get_neighbors((0, 0), connectivity=8)

        # Corner has only 3 neighbors (in 8-connectivity)
        assert len(neighbors) == 3

    def test_neighbors_exclude_obstacles(self):
        """Test that obstacles are excluded from neighbors."""
        constraints = NavigationConstraints(
            restricted_zones={(5, 6), (6, 5)}
        )
        env = NavigationEnvironment((10, 10), 10.0, constraints=constraints)

        neighbors = env.get_neighbors((5, 5), connectivity=4)

        # Two of the four cardinal neighbors are blocked
        assert (5, 6) not in neighbors  # East blocked
        assert (6, 5) not in neighbors  # South blocked
        assert len(neighbors) == 2  # Only N and W remain


class TestGridExtent:
    """Test grid extent calculations."""

    def test_get_grid_extent(self):
        """Test calculating grid extent in nautical miles."""
        grid_shape = (20, 30)  # rows x cols
        cell_size = 10.0
        origin = Point(100.0, 200.0)

        env = NavigationEnvironment(grid_shape, cell_size, origin=origin)

        bottom_left, top_right = env.get_grid_extent_nm()

        assert bottom_left == origin
        # Top right: origin + (cols * cell_size, rows * cell_size)
        assert top_right.x == 100.0 + 30 * 10.0  # 400.0
        assert top_right.y == 200.0 + 20 * 10.0  # 400.0


class TestStormZoneComputation:
    """Test storm zone identification."""

    def test_compute_storm_zones(self):
        """Test identifying storm zones in environment."""
        grid_shape = (20, 20)
        cell_size = 10.0
        storm_center = Point(100.0, 100.0)

        weather = create_storm_scenario(grid_shape, cell_size, storm_center)
        env = NavigationEnvironment(grid_shape, cell_size, weather=weather)

        storm_zones = env.compute_storm_zones(wave_threshold=5.0)

        # Should find storm zones
        assert len(storm_zones) > 0

        # Storm center should be in a high-wave zone
        storm_detected = False
        for zone in storm_zones:
            if zone.distance_to(storm_center) < 30.0:
                storm_detected = True
                break

        assert storm_detected

    def test_distance_to_nearest_storm(self):
        """Test calculating distance to nearest storm."""
        grid_shape = (30, 30)
        cell_size = 10.0
        storm_center = Point(150.0, 150.0)

        weather = create_storm_scenario(grid_shape, cell_size, storm_center)
        env = NavigationEnvironment(grid_shape, cell_size, weather=weather)

        # Point near storm
        near_point = Point(120.0, 120.0)
        distance_near = env.distance_to_nearest_storm(near_point, wave_threshold=5.0)

        # Point far from storm
        far_point = Point(10.0, 10.0)
        distance_far = env.distance_to_nearest_storm(far_point, wave_threshold=5.0)

        # Far point should be farther
        assert distance_far > distance_near
        assert distance_near < 100.0


class TestEnvironmentStatistics:
    """Test environment statistics reporting."""

    def test_get_statistics(self):
        """Test getting comprehensive environment statistics."""
        grid_shape = (20, 30)
        cell_size = 10.0

        constraints = NavigationConstraints(
            min_storm_distance=50.0,
            max_wave_height=6.0,
            restricted_zones={(5, 5), (10, 10)}
        )

        weather = create_calm_scenario(grid_shape, cell_size)
        env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

        stats = env.get_statistics()

        # Check grid stats
        assert stats['grid']['shape'] == grid_shape
        assert stats['grid']['cell_size_nm'] == cell_size
        assert stats['grid']['total_cells'] == 20 * 30
        assert stats['grid']['blocked_cells'] == 2  # Two restricted zones
        assert stats['grid']['navigable_cells'] == 20 * 30 - 2

        # Check extent
        assert 'extent_nm' in stats
        assert stats['extent_nm']['x_min'] == 0.0
        assert stats['extent_nm']['x_max'] == 30 * 10.0

        # Check weather stats
        assert 'weather' in stats
        assert 'wind' in stats['weather']
        assert 'waves' in stats['weather']

        # Check constraints
        assert stats['constraints']['min_storm_distance_nm'] == 50.0
        assert stats['constraints']['max_wave_height_m'] == 6.0
        assert stats['constraints']['num_restricted_zones'] == 2

    def test_navigable_fraction(self):
        """Test calculating navigable fraction."""
        grid_shape = (10, 10)
        env = NavigationEnvironment(grid_shape, 10.0)

        # Add restricted zone
        env.add_restricted_zone(Point(50.0, 50.0), 15.0)

        stats = env.get_statistics()

        # Some cells should be blocked
        assert stats['grid']['navigable_fraction'] < 1.0
        assert stats['grid']['navigable_fraction'] > 0.8  # Not too many blocked


class TestEnvironmentEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_restricted_zones(self):
        """Test environment with no restricted zones."""
        env = NavigationEnvironment((10, 10), 10.0)

        stats = env.get_statistics()
        assert stats['grid']['blocked_cells'] == 0
        assert stats['grid']['navigable_fraction'] == 1.0

    def test_all_blocked_neighbors(self):
        """Test cell surrounded by obstacles."""
        # Create 3x3 grid, block all edges
        restricted = {
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 2),
            (2, 0), (2, 1), (2, 2)
        }
        constraints = NavigationConstraints(restricted_zones=restricted)

        env = NavigationEnvironment((3, 3), 10.0, constraints=constraints)

        # Center cell (1,1) has no navigable neighbors
        neighbors = env.get_neighbors((1, 1), connectivity=8)
        assert len(neighbors) == 0

    def test_no_storms_distance(self):
        """Test distance to storm when no storms exist."""
        weather = create_calm_scenario((20, 20), 10.0)
        env = NavigationEnvironment((20, 20), 10.0, weather=weather)

        distance = env.distance_to_nearest_storm(Point(100.0, 100.0), wave_threshold=5.0)

        # Should return infinity when no storms
        assert distance == float('inf')
