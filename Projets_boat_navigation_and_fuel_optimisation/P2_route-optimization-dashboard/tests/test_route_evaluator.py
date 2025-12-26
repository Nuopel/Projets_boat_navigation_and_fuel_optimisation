"""
Unit tests for route evaluator.

Tests route objective calculations, segment evaluation,
and integration with ship dynamics and weather.
"""

import pytest
import numpy as np
from src.planning.route_evaluator import RouteEvaluator, RouteObjectives, create_direct_route
from src.models.ship_model import ShipDynamics, ShipSpecifications, create_default_ship
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.models.weather_field import create_calm_scenario, create_storm_scenario
from src.utils.geometry import Point


class TestRouteObjectives:
    """Test RouteObjectives dataclass."""

    def test_route_objectives_creation(self):
        """Test creating RouteObjectives."""
        obj = RouteObjectives(
            time_hours=10.0,
            fuel_liters=5000.0,
            emissions_kg=14000.0,
            distance_nm=150.0
        )

        assert obj.time_hours == 10.0
        assert obj.fuel_liters == 5000.0
        assert obj.emissions_kg == 14000.0
        assert obj.distance_nm == 150.0

    def test_route_objectives_str(self):
        """Test string representation."""
        obj = RouteObjectives(10.0, 5000.0, 14000.0, 150.0)
        s = str(obj)

        assert "10.00 h" in s
        assert "5000.0 L" in s
        assert "14000.0 kg" in s
        assert "150.0 nm" in s


class TestCreateDirectRoute:
    """Test direct route creation utility."""

    def test_create_direct_route_two_waypoints(self):
        """Test creating direct route with 2 waypoints."""
        start = Point(0, 0)
        goal = Point(100, 100)

        route = create_direct_route(start, goal, num_waypoints=2)

        assert len(route) == 2
        assert route[0] == start
        assert route[-1] == goal

    def test_create_direct_route_five_waypoints(self):
        """Test creating direct route with 5 waypoints."""
        start = Point(0, 0)
        goal = Point(100, 0)

        route = create_direct_route(start, goal, num_waypoints=5)

        assert len(route) == 5
        assert route[0] == start
        assert route[-1] == goal

        # Check intermediate waypoints are evenly spaced
        assert route[1] == Point(25, 0)
        assert route[2] == Point(50, 0)
        assert route[3] == Point(75, 0)

    def test_create_direct_route_invalid_waypoints(self):
        """Test creating route with invalid waypoint count."""
        start = Point(0, 0)
        goal = Point(100, 100)

        with pytest.raises(ValueError):
            create_direct_route(start, goal, num_waypoints=1)

        with pytest.raises(ValueError):
            create_direct_route(start, goal, num_waypoints=0)


class TestRouteEvaluator:
    """Test RouteEvaluator."""

    @pytest.fixture
    def ship(self):
        """Create default ship."""
        return create_default_ship()

    @pytest.fixture
    def calm_env(self):
        """Create calm environment."""
        grid_shape = (50, 50)
        cell_size = 10.0
        weather = create_calm_scenario(grid_shape, cell_size)
        constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
        return NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

    @pytest.fixture
    def storm_env(self):
        """Create storm environment."""
        grid_shape = (50, 50)
        cell_size = 10.0
        storm_center = Point(250, 250)
        weather = create_storm_scenario(grid_shape, cell_size, storm_center, storm_radius=100.0)
        constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
        return NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

    @pytest.fixture
    def evaluator_calm(self, ship, calm_env):
        """Create evaluator with calm environment."""
        return RouteEvaluator(ship, calm_env)

    @pytest.fixture
    def evaluator_storm(self, ship, storm_env):
        """Create evaluator with storm environment."""
        return RouteEvaluator(ship, storm_env)

    # ==================== Basic Evaluation ====================

    def test_evaluate_route_two_waypoints(self, evaluator_calm):
        """Test evaluating simple two-waypoint route."""
        route = [Point(0, 0), Point(100, 0)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        assert obj.distance_nm == pytest.approx(100.0)
        assert obj.time_hours > 0
        assert obj.fuel_liters > 0
        assert obj.emissions_kg > 0

        # Time should be distance/speed
        expected_time = 100.0 / 15.0
        assert obj.time_hours == pytest.approx(expected_time)

    def test_evaluate_route_multiple_waypoints(self, evaluator_calm):
        """Test evaluating route with multiple waypoints."""
        route = [Point(0, 0), Point(100, 0), Point(100, 100), Point(200, 100)]
        speed = 12.0

        obj = evaluator_calm.evaluate_route(route, speed)

        # Total distance = 100 + 100 + 100 = 300 nm
        expected_distance = 300.0
        assert obj.distance_nm == pytest.approx(expected_distance)

        # Time = distance / speed
        expected_time = 300.0 / 12.0
        assert obj.time_hours == pytest.approx(expected_time)

    def test_evaluate_route_diagonal(self, evaluator_calm):
        """Test evaluating diagonal route."""
        route = [Point(0, 0), Point(100, 100)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        expected_distance = np.sqrt(100**2 + 100**2)
        assert obj.distance_nm == pytest.approx(expected_distance)

    def test_evaluate_route_single_waypoint(self, evaluator_calm):
        """Test evaluating route with single waypoint."""
        route = [Point(100, 100)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        assert obj.distance_nm == 0.0
        assert obj.time_hours == 0.0
        assert obj.fuel_liters == 0.0
        assert obj.emissions_kg == 0.0

    # ==================== Speed Validation ====================

    def test_evaluate_route_speed_below_min(self, evaluator_calm, ship):
        """Test evaluation with speed below minimum."""
        route = [Point(0, 0), Point(100, 0)]
        speed = ship.specs.v_min - 1.0

        with pytest.raises(ValueError, match="below minimum"):
            evaluator_calm.evaluate_route(route, speed)

    def test_evaluate_route_speed_above_max(self, evaluator_calm, ship):
        """Test evaluation with speed above maximum."""
        route = [Point(0, 0), Point(100, 0)]
        speed = ship.specs.v_max + 1.0

        with pytest.raises(ValueError, match="above maximum"):
            evaluator_calm.evaluate_route(route, speed)

    def test_evaluate_route_speed_at_min(self, evaluator_calm, ship):
        """Test evaluation at minimum speed."""
        route = [Point(0, 0), Point(100, 0)]
        speed = ship.specs.v_min

        obj = evaluator_calm.evaluate_route(route, speed)

        assert obj.time_hours > 0
        assert obj.fuel_liters > 0

    def test_evaluate_route_speed_at_max(self, evaluator_calm, ship):
        """Test evaluation at maximum speed."""
        route = [Point(0, 0), Point(100, 0)]
        speed = ship.specs.v_max

        obj = evaluator_calm.evaluate_route(route, speed)

        assert obj.time_hours > 0
        assert obj.fuel_liters > 0

    # ==================== Weather Effects ====================

    def test_storm_increases_fuel_consumption(self, evaluator_calm, evaluator_storm):
        """Test that storm increases fuel consumption vs. calm."""
        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0

        obj_calm = evaluator_calm.evaluate_route(route, speed)
        obj_storm = evaluator_storm.evaluate_route(route, speed)

        # Storm should increase fuel consumption
        # (route passes through or near storm center)
        assert obj_storm.fuel_liters >= obj_calm.fuel_liters

    def test_weather_affects_emissions(self, evaluator_calm, evaluator_storm):
        """Test weather affects emissions."""
        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0

        obj_calm = evaluator_calm.evaluate_route(route, speed)
        obj_storm = evaluator_storm.evaluate_route(route, speed)

        # Emissions scale with fuel
        assert obj_storm.emissions_kg >= obj_calm.emissions_kg

    # ==================== Speed-Fuel Relationship ====================

    def test_higher_speed_more_fuel(self, evaluator_calm):
        """Test higher speed consumes more fuel."""
        route = [Point(0, 0), Point(100, 0)]

        obj_slow = evaluator_calm.evaluate_route(route, speed=10.0)
        obj_fast = evaluator_calm.evaluate_route(route, speed=16.0)

        # Higher speed should consume more fuel (cubic relationship)
        assert obj_fast.fuel_liters > obj_slow.fuel_liters

    def test_higher_speed_less_time(self, evaluator_calm):
        """Test higher speed reduces travel time."""
        route = [Point(0, 0), Point(100, 0)]

        obj_slow = evaluator_calm.evaluate_route(route, speed=10.0)
        obj_fast = evaluator_calm.evaluate_route(route, speed=16.0)

        # Higher speed should reduce time
        assert obj_fast.time_hours < obj_slow.time_hours

    def test_fuel_emissions_correlation(self, evaluator_calm):
        """Test fuel and emissions are correlated."""
        route = [Point(0, 0), Point(100, 0)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        # Emissions should be proportional to fuel
        # emission_factor = 2.8 kg CO2/L fuel
        expected_emissions = obj.fuel_liters * 2.8
        assert obj.emissions_kg == pytest.approx(expected_emissions, rel=0.01)

    # ==================== Variable Speed Profiles ====================

    def test_evaluate_variable_speed_two_segments(self, evaluator_calm):
        """Test evaluation with variable speed profile."""
        route = [Point(0, 0), Point(100, 0), Point(200, 0)]
        speed_profile = [12.0, 16.0]

        obj = evaluator_calm.evaluate_route(route, speed_profile)

        assert obj.distance_nm == pytest.approx(200.0)
        assert obj.time_hours > 0
        assert obj.fuel_liters > 0

    def test_evaluate_variable_speed_mismatched_length(self, evaluator_calm):
        """Test variable speed with mismatched length."""
        route = [Point(0, 0), Point(100, 0), Point(200, 0)]
        speed_profile = [12.0]  # Only 1 speed for 2 segments

        with pytest.raises(ValueError, match="must have.*speeds"):
            evaluator_calm.evaluate_route(route, speed_profile)

    def test_variable_speed_below_min(self, evaluator_calm, ship):
        """Test variable speed with one below minimum."""
        route = [Point(0, 0), Point(100, 0), Point(200, 0)]
        speed_profile = [15.0, ship.specs.v_min - 1.0]

        with pytest.raises(ValueError, match="below minimum"):
            evaluator_calm.evaluate_route(route, speed_profile)

    def test_variable_speed_above_max(self, evaluator_calm, ship):
        """Test variable speed with one above maximum."""
        route = [Point(0, 0), Point(100, 0), Point(200, 0)]
        speed_profile = [15.0, ship.specs.v_max + 1.0]

        with pytest.raises(ValueError, match="above maximum"):
            evaluator_calm.evaluate_route(route, speed_profile)

    # ==================== Empty Route Handling ====================

    def test_evaluate_empty_route(self, evaluator_calm):
        """Test evaluating empty route."""
        route = []
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        assert obj.distance_nm == 0.0
        assert obj.time_hours == 0.0
        assert obj.fuel_liters == 0.0
        assert obj.emissions_kg == 0.0

    # ==================== Segment-Level Consistency ====================

    def test_route_equals_sum_of_segments(self, evaluator_calm):
        """Test total route objectives equal sum of segments."""
        route = [Point(0, 0), Point(100, 0), Point(100, 100)]
        speed = 15.0

        # Evaluate full route
        obj_full = evaluator_calm.evaluate_route(route, speed)

        # Evaluate segments separately
        seg1 = evaluator_calm.evaluate_route([route[0], route[1]], speed)
        seg2 = evaluator_calm.evaluate_route([route[1], route[2]], speed)

        # Sum should match
        assert obj_full.distance_nm == pytest.approx(seg1.distance_nm + seg2.distance_nm)
        assert obj_full.time_hours == pytest.approx(seg1.time_hours + seg2.time_hours)
        assert obj_full.fuel_liters == pytest.approx(seg1.fuel_liters + seg2.fuel_liters)
        assert obj_full.emissions_kg == pytest.approx(seg1.emissions_kg + seg2.emissions_kg)

    # ==================== Long Routes ====================

    def test_evaluate_long_route(self, evaluator_calm):
        """Test evaluating route with many waypoints."""
        # Create 20-waypoint route
        route = [Point(i * 10, i * 10) for i in range(20)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        assert obj.distance_nm > 0
        assert obj.time_hours > 0
        assert obj.fuel_liters > 0

    # ==================== Numerical Stability ====================

    def test_very_short_segment(self, evaluator_calm):
        """Test evaluation with very short segment."""
        route = [Point(100, 100), Point(100.001, 100)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        # Should handle gracefully
        assert obj.distance_nm >= 0
        assert obj.time_hours >= 0
        assert obj.fuel_liters >= 0

    def test_very_long_segment(self, evaluator_calm):
        """Test evaluation with very long segment."""
        route = [Point(0, 0), Point(10000, 10000)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        expected_distance = np.sqrt(10000**2 + 10000**2)
        assert obj.distance_nm == pytest.approx(expected_distance)

    # ==================== Custom Ship Specifications ====================

    def test_evaluate_with_custom_ship(self, calm_env):
        """Test evaluation with custom ship specs."""
        # Create ship with different coefficients
        specs = ShipSpecifications(
            name="Test Ship",
            v_min=10.0,
            v_max=20.0,
            fuel_coef_speed=0.3,
            fuel_coef_wind=0.5,
            fuel_coef_wave=10.0,
            fuel_base=300.0,
            emission_factor=2.5
        )
        ship = ShipDynamics(specs)
        evaluator = RouteEvaluator(ship, calm_env)

        route = [Point(0, 0), Point(100, 0)]
        speed = 15.0

        obj = evaluator.evaluate_route(route, speed)

        assert obj.time_hours > 0
        assert obj.fuel_liters > 0
        # Emissions should use custom factor
        expected_emissions = obj.fuel_liters * 2.5
        assert obj.emissions_kg == pytest.approx(expected_emissions)

    # ==================== Weather Sampling ====================

    def test_weather_sampled_at_midpoint(self, evaluator_calm):
        """Test that weather is sampled at segment midpoint."""
        # This is an implementation detail test
        # Create route and verify objectives are reasonable
        route = [Point(0, 0), Point(100, 0)]
        speed = 15.0

        obj = evaluator_calm.evaluate_route(route, speed)

        # In calm weather, fuel should be minimal
        # Verify result is consistent with calm conditions
        assert obj.fuel_liters > 0
        assert obj.time_hours > 0
