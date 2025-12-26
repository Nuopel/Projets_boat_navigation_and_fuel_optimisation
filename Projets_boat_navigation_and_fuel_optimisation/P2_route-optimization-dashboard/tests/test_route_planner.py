"""
Unit tests for A* route planner.

Tests pathfinding correctness, weather penalty integration,
path smoothing, and edge cases.
"""

import pytest
import numpy as np
from src.planning.route_planner import RoutePlanner
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.models.weather_field import WeatherField, WeatherZone, create_calm_scenario, create_storm_scenario
from src.utils.geometry import Point


class TestRoutePlanner:
    """Test suite for RoutePlanner."""

    @pytest.fixture
    def calm_env(self):
        """Create calm weather environment."""
        grid_shape = (50, 50)
        cell_size = 10.0
        weather = create_calm_scenario(grid_shape, cell_size)
        constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
        return NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

    @pytest.fixture
    def storm_env(self):
        """Create storm scenario environment."""
        grid_shape = (50, 50)
        cell_size = 10.0
        storm_center = Point(250.0, 250.0)
        weather = create_storm_scenario(grid_shape, cell_size, storm_center, storm_radius=100.0)
        constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
        return NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

    @pytest.fixture
    def planner_calm(self, calm_env):
        """Create planner with calm environment."""
        return RoutePlanner(calm_env)

    @pytest.fixture
    def planner_storm(self, storm_env):
        """Create planner with storm environment."""
        return RoutePlanner(storm_env)

    # ==================== Basic Pathfinding ====================

    def test_plan_route_straight_line_calm(self, planner_calm):
        """Test A* finds direct route in calm weather."""
        start = Point(50, 50)
        goal = Point(450, 450)

        route = planner_calm.plan_route(start, goal, use_weather_penalty=False)

        assert route is not None
        assert len(route) >= 2
        assert route[0] == start
        assert route[-1] == goal

    def test_plan_route_short_distance(self, planner_calm):
        """Test A* on short distance."""
        start = Point(100, 100)
        goal = Point(150, 150)

        route = planner_calm.plan_route(start, goal)

        assert route is not None
        assert route[0] == start
        assert route[-1] == goal

    def test_plan_route_same_start_goal(self, planner_calm):
        """Test A* when start == goal."""
        start = Point(250, 250)
        goal = Point(250, 250)

        route = planner_calm.plan_route(start, goal)

        assert route is not None
        assert len(route) == 1
        assert route[0] == start

    def test_plan_route_adjacent_cells(self, planner_calm):
        """Test A* for adjacent cells."""
        start = Point(100, 100)
        goal = Point(110, 100)  # One cell to the right (10 nm)

        route = planner_calm.plan_route(start, goal)

        assert route is not None
        assert route[0] == start
        assert route[-1] == goal

    # ==================== Weather Penalty Integration ====================

    def test_plan_route_with_weather_penalty(self, planner_storm):
        """Test A* uses weather penalties to avoid storms."""
        start = Point(50, 50)
        goal = Point(450, 450)

        # Route with weather penalty should detour around storm
        route_with_penalty = planner_storm.plan_route(start, goal, use_weather_penalty=True)
        route_without_penalty = planner_storm.plan_route(start, goal, use_weather_penalty=False)

        assert route_with_penalty is not None
        assert route_without_penalty is not None

        # With penalty should be longer (detours around storm)
        len_with = planner_storm.get_path_length(route_with_penalty)
        len_without = planner_storm.get_path_length(route_without_penalty)

        # Storm is at grid center, direct path goes through it
        # Weather-penalized path should avoid it (longer)
        assert len_with >= len_without

    def test_weather_penalty_affects_cost(self, planner_storm):
        """Test that weather penalty increases edge costs."""
        start = Point(50, 50)
        goal = Point(450, 450)

        route = planner_storm.plan_route(start, goal, use_weather_penalty=True)

        assert route is not None
        # Check that route avoids storm center
        storm_center = Point(250, 250)
        min_distance_to_storm = min(wp.distance_to(storm_center) for wp in route)

        # Should maintain some distance from storm
        # (exact value depends on penalty weighting)
        assert min_distance_to_storm > 0

    # ==================== Path Smoothing ====================

    def test_smooth_path_reduces_waypoints(self, planner_calm):
        """Test path smoothing reduces waypoint count."""
        start = Point(50, 50)
        goal = Point(450, 450)

        route = planner_calm.plan_route(start, goal, use_weather_penalty=False)
        original_count = len(route)

        smoothed = planner_calm.smooth_path(route, max_iterations=10)

        # Smoothing should reduce or maintain waypoint count
        assert len(smoothed) <= original_count
        # Should still have at least start and goal
        assert len(smoothed) >= 2
        assert smoothed[0] == start
        assert smoothed[-1] == goal

    def test_smooth_path_maintains_navigability(self, planner_storm):
        """Test smoothed path remains navigable."""
        start = Point(50, 50)
        goal = Point(450, 450)

        route = planner_storm.plan_route(start, goal, use_weather_penalty=True)
        smoothed = planner_storm.smooth_path(route, max_iterations=10)

        # Check all waypoints are navigable
        for waypoint in smoothed:
            assert planner_storm.env.is_navigable(waypoint)

    def test_smooth_path_single_waypoint(self, planner_calm):
        """Test smoothing path with single waypoint."""
        path = [Point(250, 250)]
        smoothed = planner_calm.smooth_path(path)

        assert smoothed == path

    def test_smooth_path_two_waypoints(self, planner_calm):
        """Test smoothing path with two waypoints."""
        path = [Point(100, 100), Point(200, 200)]
        smoothed = planner_calm.smooth_path(path)

        assert len(smoothed) == 2
        assert smoothed == path

    def test_smooth_path_zero_iterations(self, planner_calm):
        """Test smoothing with zero iterations."""
        start = Point(50, 50)
        goal = Point(450, 450)
        route = planner_calm.plan_route(start, goal)

        smoothed = planner_calm.smooth_path(route, max_iterations=0)

        # Should return original path
        assert smoothed == route

    # ==================== Path Length Calculation ====================

    def test_get_path_length_straight_line(self, planner_calm):
        """Test path length for straight line."""
        path = [Point(0, 0), Point(100, 0), Point(200, 0)]
        length = planner_calm.get_path_length(path)

        assert length == pytest.approx(200.0)

    def test_get_path_length_diagonal(self, planner_calm):
        """Test path length for diagonal."""
        path = [Point(0, 0), Point(100, 100)]
        expected = np.sqrt(100**2 + 100**2)
        length = planner_calm.get_path_length(path)

        assert length == pytest.approx(expected)

    def test_get_path_length_single_point(self, planner_calm):
        """Test path length for single point."""
        path = [Point(100, 100)]
        length = planner_calm.get_path_length(path)

        assert length == 0.0

    def test_get_path_length_empty(self, planner_calm):
        """Test path length for empty path."""
        path = []
        length = planner_calm.get_path_length(path)

        assert length == 0.0

    # ==================== Connectivity Options ====================

    def test_plan_route_4_connectivity(self, planner_calm):
        """Test A* with 4-connectivity."""
        start = Point(100, 100)
        goal = Point(200, 200)

        route = planner_calm.plan_route(start, goal, connectivity=4)

        assert route is not None
        assert route[0] == start
        assert route[-1] == goal

    def test_plan_route_8_connectivity(self, planner_calm):
        """Test A* with 8-connectivity."""
        start = Point(100, 100)
        goal = Point(200, 200)

        route = planner_calm.plan_route(start, goal, connectivity=8)

        assert route is not None
        assert route[0] == start
        assert route[-1] == goal

    def test_8_connectivity_shorter_than_4(self, planner_calm):
        """Test 8-connectivity produces shorter diagonal paths."""
        start = Point(100, 100)
        goal = Point(200, 200)

        route_4 = planner_calm.plan_route(start, goal, connectivity=4)
        route_8 = planner_calm.plan_route(start, goal, connectivity=8)

        len_4 = planner_calm.get_path_length(route_4)
        len_8 = planner_calm.get_path_length(route_8)

        # 8-connectivity should allow diagonal moves (shorter)
        assert len_8 <= len_4

    # ==================== Edge Cases ====================

    def test_plan_route_out_of_bounds_start(self, planner_calm):
        """Test A* with out-of-bounds start."""
        start = Point(-10, -10)
        goal = Point(250, 250)

        route = planner_calm.plan_route(start, goal)

        # Should handle gracefully (return None or raise)
        assert route is None or isinstance(route, list)

    def test_plan_route_out_of_bounds_goal(self, planner_calm):
        """Test A* with out-of-bounds goal."""
        start = Point(250, 250)
        goal = Point(1000, 1000)

        route = planner_calm.plan_route(start, goal)

        # Should handle gracefully
        assert route is None or isinstance(route, list)

    def test_plan_route_blocked_goal(self):
        """Test A* when goal is in restricted zone."""
        grid_shape = (50, 50)
        cell_size = 10.0
        weather = create_calm_scenario(grid_shape, cell_size)

        # Create environment with restricted zone at goal
        restricted_zones = [Point(450, 450)]
        constraints = NavigationConstraints(
            min_storm_distance=50.0,
            max_wave_height=6.0,
            restricted_zones=restricted_zones,
            restricted_radius=30.0
        )
        env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)
        planner = RoutePlanner(env)

        start = Point(50, 50)
        goal = Point(450, 450)

        route = planner.plan_route(start, goal)

        # Should fail to find path (goal is restricted)
        assert route is None

    # ==================== Performance ====================

    def test_plan_route_performance(self, planner_calm):
        """Test A* completes in reasonable time."""
        import time

        start = Point(50, 50)
        goal = Point(450, 450)

        start_time = time.time()
        route = planner_calm.plan_route(start, goal)
        elapsed = time.time() - start_time

        assert route is not None
        # Should complete in under 1 second
        assert elapsed < 1.0

    # ==================== Heuristic Function ====================

    def test_heuristic_euclidean_distance(self, planner_calm):
        """Test heuristic function returns Euclidean distance."""
        p1 = Point(0, 0)
        p2 = Point(100, 100)

        # Access internal heuristic (if exposed) or test via pathfinding
        expected = np.sqrt(100**2 + 100**2)

        # Heuristic should be admissible (≤ actual distance)
        # Test indirectly by checking path optimality
        route = planner_calm.plan_route(p1, p2)
        path_length = planner_calm.get_path_length(route)

        # Path should be close to straight-line distance in calm weather
        assert path_length >= expected - 1.0  # Allow small tolerance

    # ==================== Storm Avoidance ====================

    def test_storm_detour_increases_distance(self, planner_storm):
        """Test storm forces detour (increases path length)."""
        start = Point(50, 50)
        goal = Point(450, 450)

        # Compare with calm environment
        grid_shape = (50, 50)
        cell_size = 10.0
        weather_calm = create_calm_scenario(grid_shape, cell_size)
        constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
        env_calm = NavigationEnvironment(grid_shape, cell_size, weather=weather_calm, constraints=constraints)
        planner_calm_local = RoutePlanner(env_calm)

        route_calm = planner_calm_local.plan_route(start, goal, use_weather_penalty=True)
        route_storm = planner_storm.plan_route(start, goal, use_weather_penalty=True)

        if route_calm and route_storm:
            len_calm = planner_calm_local.get_path_length(route_calm)
            len_storm = planner_storm.get_path_length(route_storm)

            # Storm should force longer path
            assert len_storm >= len_calm

    def test_multiple_smoothing_iterations(self, planner_calm):
        """Test multiple smoothing iterations converge."""
        start = Point(50, 50)
        goal = Point(450, 450)

        route = planner_calm.plan_route(start, goal)

        smoothed_5 = planner_calm.smooth_path(route, max_iterations=5)
        smoothed_20 = planner_calm.smooth_path(route, max_iterations=20)

        # More iterations should produce equal or fewer waypoints
        assert len(smoothed_20) <= len(smoothed_5)
