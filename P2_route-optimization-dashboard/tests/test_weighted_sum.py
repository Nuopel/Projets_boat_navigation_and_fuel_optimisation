"""
Unit tests for weighted sum optimizer.

Tests optimization convergence, weight variations, constraint
satisfaction, and integration with A* pathfinding.
"""

import pytest
import numpy as np
from src.optimizers.weighted_sum import WeightedSumOptimizer
from src.optimizers.base_optimizer import OptimizationResult
from src.models.ship_model import create_default_ship
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.models.weather_field import create_calm_scenario, create_storm_scenario
from src.planning.constraints import TimeWindow
from src.utils.geometry import Point


class TestWeightedSumOptimizer:
    """Test WeightedSumOptimizer."""

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
    def optimizer_calm(self, ship, calm_env):
        """Create optimizer with calm environment."""
        return WeightedSumOptimizer(ship, calm_env, weights=(0.3, 0.6, 0.1))

    @pytest.fixture
    def optimizer_storm(self, ship, storm_env):
        """Create optimizer with storm environment."""
        return WeightedSumOptimizer(ship, storm_env, weights=(0.3, 0.6, 0.1))

    # ==================== Initialization ====================

    def test_initialization_default_weights(self, ship, calm_env):
        """Test initialization with default weights."""
        optimizer = WeightedSumOptimizer(ship, calm_env)

        assert optimizer.weights == (0.2, 0.7, 0.1)

    def test_initialization_custom_weights(self, ship, calm_env):
        """Test initialization with custom weights."""
        optimizer = WeightedSumOptimizer(ship, calm_env, weights=(0.5, 0.3, 0.2))

        assert optimizer.weights == (0.5, 0.3, 0.2)

    def test_initialization_invalid_weights_length(self, ship, calm_env):
        """Test initialization with wrong number of weights."""
        with pytest.raises(ValueError, match="must be tuple of 3"):
            WeightedSumOptimizer(ship, calm_env, weights=(0.5, 0.5))

    def test_initialization_negative_weights(self, ship, calm_env):
        """Test initialization with negative weights."""
        with pytest.raises(ValueError, match="non-negative"):
            WeightedSumOptimizer(ship, calm_env, weights=(0.5, -0.3, 0.2))

    def test_initialization_all_zero_weights(self, ship, calm_env):
        """Test initialization with all zero weights."""
        with pytest.raises(ValueError, match="at least one weight"):
            WeightedSumOptimizer(ship, calm_env, weights=(0.0, 0.0, 0.0))

    def test_initialization_some_zero_weights(self, ship, calm_env):
        """Test initialization with some zero weights."""
        # Should be valid
        optimizer = WeightedSumOptimizer(ship, calm_env, weights=(1.0, 0.0, 0.0))
        assert optimizer.weights == (1.0, 0.0, 0.0)

    # ==================== Basic Optimization ====================

    def test_optimize_calm_weather_astar(self, optimizer_calm):
        """Test optimization in calm weather with A*."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal, use_astar=True)

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert len(result.waypoints) >= 2
        assert result.waypoints[0] == start
        assert result.waypoints[-1] == goal
        assert result.speed > 0
        assert result.objectives.time_hours > 0
        assert result.objectives.fuel_liters > 0
        assert result.objectives.emissions_kg > 0

    def test_optimize_calm_weather_direct(self, optimizer_calm):
        """Test optimization with direct route (no A*)."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal, use_astar=False)

        assert result.success is True
        assert len(result.waypoints) == 2
        assert result.waypoints[0] == start
        assert result.waypoints[-1] == goal

    def test_optimize_returns_valid_speed(self, optimizer_calm, ship):
        """Test optimized speed is within bounds."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal)

        assert ship.specs.v_min <= result.speed <= ship.specs.v_max

    def test_optimize_iterations(self, optimizer_calm):
        """Test optimizer reports iteration count."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal)

        assert result.iterations >= 0
        assert result.iterations < 100  # Should converge quickly

    # ==================== Weight Effects ====================

    def test_time_priority_chooses_higher_speed(self, ship, calm_env):
        """Test time-weighted optimization prefers higher speed."""
        optimizer_time = WeightedSumOptimizer(ship, calm_env, weights=(1.0, 0.0, 0.0))
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_time.optimize(start, goal, use_astar=False)

        # Time priority should choose higher speed
        assert result.speed > (ship.specs.v_min + ship.specs.v_max) / 2.0

    def test_fuel_priority_chooses_lower_speed(self, ship, calm_env):
        """Test fuel-weighted optimization prefers lower speed."""
        optimizer_fuel = WeightedSumOptimizer(ship, calm_env, weights=(0.0, 1.0, 0.0))
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_fuel.optimize(start, goal, use_astar=False)

        # Fuel priority should choose lower speed
        assert result.speed < (ship.specs.v_min + ship.specs.v_max) / 2.0

    def test_balanced_weights_mid_speed(self, ship, calm_env):
        """Test balanced weights choose intermediate speed."""
        optimizer_balanced = WeightedSumOptimizer(ship, calm_env, weights=(0.5, 0.5, 0.0))
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_balanced.optimize(start, goal, use_astar=False)

        # Balanced should be somewhere in middle
        assert ship.specs.v_min < result.speed < ship.specs.v_max

    def test_weight_variation_affects_objectives(self, ship, calm_env):
        """Test different weights produce different objective trade-offs."""
        start = Point(50, 50)
        goal = Point(450, 450)

        # Time-focused
        opt_time = WeightedSumOptimizer(ship, calm_env, weights=(1.0, 0.0, 0.0))
        result_time = opt_time.optimize(start, goal, use_astar=False)

        # Fuel-focused
        opt_fuel = WeightedSumOptimizer(ship, calm_env, weights=(0.0, 1.0, 0.0))
        result_fuel = opt_fuel.optimize(start, goal, use_astar=False)

        # Time-focused should have lower time, higher fuel
        assert result_time.objectives.time_hours < result_fuel.objectives.time_hours
        assert result_time.objectives.fuel_liters > result_fuel.objectives.fuel_liters

    # ==================== Time Window Constraints ====================

    def test_optimize_with_time_window(self, optimizer_calm):
        """Test optimization with time window constraint."""
        start = Point(50, 50)
        goal = Point(450, 450)
        time_window = TimeWindow(min_hours=20.0, max_hours=40.0)

        result = optimizer_calm.optimize(start, goal, time_window=time_window)

        # Should satisfy time window if feasible
        if result.success:
            assert time_window.min_hours <= result.objectives.time_hours <= time_window.max_hours

    def test_optimize_tight_time_window(self, optimizer_calm):
        """Test optimization with very tight time window."""
        start = Point(50, 50)
        goal = Point(450, 450)
        time_window = TimeWindow(min_hours=25.0, max_hours=26.0)

        result = optimizer_calm.optimize(start, goal, time_window=time_window)

        # May or may not be feasible
        if result.success:
            assert time_window.min_hours <= result.objectives.time_hours <= time_window.max_hours

    def test_optimize_impossible_time_window(self, optimizer_calm):
        """Test optimization with impossible time window."""
        start = Point(50, 50)
        goal = Point(450, 450)
        # Impossible: requires too fast speed
        time_window = TimeWindow(max_hours=1.0)

        result = optimizer_calm.optimize(start, goal, time_window=time_window)

        # Should fail or report violation
        assert result.success is False or len(result.metadata.get('violations', [])) > 0

    # ==================== Initial Speed Guess ====================

    def test_optimize_with_initial_speed(self, optimizer_calm):
        """Test optimization with custom initial speed."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal, initial_speed=12.0)

        assert result.success is True
        assert result.speed > 0

    def test_optimize_no_initial_speed(self, optimizer_calm):
        """Test optimization without initial speed (uses midpoint)."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal)

        assert result.success is True
        assert result.speed > 0

    # ==================== Storm Scenarios ====================

    def test_optimize_storm_detour(self, optimizer_storm):
        """Test optimization with storm (should use A* detour)."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_storm.optimize(start, goal, use_astar=True)

        # A* should find a route (may or may not satisfy constraints)
        assert isinstance(result, OptimizationResult)
        if result.success:
            assert len(result.waypoints) >= 2

    def test_storm_increases_distance(self, optimizer_calm, optimizer_storm):
        """Test storm scenario increases route distance."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result_calm = optimizer_calm.optimize(start, goal, use_astar=True)
        result_storm = optimizer_storm.optimize(start, goal, use_astar=True)

        # Storm should increase distance (detour)
        # Only compare if both successful
        if result_calm.success and result_storm.success:
            assert result_storm.objectives.distance_nm >= result_calm.objectives.distance_nm

    # ==================== A* Integration ====================

    def test_astar_failure_handling(self, ship, calm_env):
        """Test handling when A* fails to find path."""
        # Create environment with blocked goal
        restricted_zones = [Point(450, 450)]
        constraints = NavigationConstraints(
            min_storm_distance=50.0,
            max_wave_height=6.0,
            restricted_zones=restricted_zones,
            restricted_radius=100.0
        )
        grid_shape = (50, 50)
        cell_size = 10.0
        weather = create_calm_scenario(grid_shape, cell_size)
        env_blocked = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)
        optimizer = WeightedSumOptimizer(ship, env_blocked)

        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer.optimize(start, goal, use_astar=True)

        # Should handle gracefully
        assert result.success is False
        assert "pathfinding failed" in result.message.lower() or "failed" in result.message.lower()

    def test_astar_smoothing_reduces_waypoints(self, optimizer_calm):
        """Test A* path is smoothed."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal, use_astar=True)

        # Smoothing should reduce waypoints significantly
        if result.success:
            # In calm weather, should smooth to nearly direct route
            assert len(result.waypoints) <= 10  # Reasonable after smoothing

    # ==================== Metadata ====================

    def test_result_includes_metadata(self, optimizer_calm):
        """Test result includes metadata."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal)

        assert 'weights' in result.metadata
        assert 'num_waypoints' in result.metadata
        assert 'path_length_nm' in result.metadata
        assert result.metadata['weights'] == optimizer_calm.weights

    def test_metadata_violations_when_infeasible(self, optimizer_calm, ship):
        """Test metadata includes violations when infeasible."""
        time_window = TimeWindow(max_hours=1.0)  # Impossible

        start = Point(50, 50)
        goal = Point(450, 450)

        # Temporarily modify to check violations
        from src.planning.constraints import ConstraintChecker
        checker = ConstraintChecker(optimizer_calm.env, ship, time_window)

        # Optimize and manually validate
        result = optimizer_calm.optimize(start, goal, time_window=time_window)

        if not result.success:
            assert 'violations' in result.metadata
            assert len(result.metadata['violations']) > 0

    # ==================== Optimizer Interface ====================

    def test_get_name(self, optimizer_calm):
        """Test get_name method."""
        name = optimizer_calm.get_name()

        assert "WeightedSum" in name
        assert "0.30" in name  # Formatted weights
        assert "0.60" in name
        assert "0.10" in name

    def test_get_parameters(self, optimizer_calm):
        """Test get_parameters method."""
        params = optimizer_calm.get_parameters()

        assert 'weights' in params
        assert 'w_time' in params
        assert 'w_fuel' in params
        assert 'w_emissions' in params
        assert params['weights'] == (0.3, 0.6, 0.1)
        assert params['w_time'] == 0.3
        assert params['w_fuel'] == 0.6
        assert params['w_emissions'] == 0.1

    # ==================== Max Iterations ====================

    def test_optimize_max_iterations(self, optimizer_calm):
        """Test optimization with custom max iterations."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal, max_iterations=5)

        # Should stop early
        assert result.iterations <= 5

    def test_optimize_default_max_iterations(self, optimizer_calm):
        """Test optimization uses default max iterations."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer_calm.optimize(start, goal)

        # Default is 100
        assert result.iterations < 100  # Should converge before max

    # ==================== Edge Cases ====================

    def test_optimize_same_start_goal(self, optimizer_calm):
        """Test optimization when start == goal."""
        start = Point(250, 250)
        goal = Point(250, 250)

        result = optimizer_calm.optimize(start, goal)

        assert result.success is True
        assert result.objectives.distance_nm == 0.0
        assert result.objectives.time_hours == 0.0

    def test_optimize_short_distance(self, optimizer_calm):
        """Test optimization for very short distance."""
        start = Point(100, 100)
        goal = Point(110, 100)  # 10 nm

        result = optimizer_calm.optimize(start, goal)

        assert result.success is True
        assert result.objectives.distance_nm < 20.0

    # ==================== Weight Space Scanning ====================

    def test_scan_weight_space(self, optimizer_calm):
        """Test scanning different weight combinations."""
        start = Point(50, 50)
        goal = Point(450, 450)

        results = optimizer_calm.scan_weight_space(start, goal, num_samples=5)

        assert len(results) == 5
        # All should be OptimizationResults
        for result in results:
            assert isinstance(result, OptimizationResult)

        # Should explore different weight combinations
        # Check that results have varying objectives
        times = [r.objectives.time_hours for r in results if r.success]
        if len(times) > 1:
            # Should have some variation
            assert max(times) > min(times)

    def test_scan_weight_space_preserves_original_weights(self, optimizer_calm):
        """Test weight scanning restores original weights."""
        original_weights = optimizer_calm.weights
        start = Point(50, 50)
        goal = Point(450, 450)

        optimizer_calm.scan_weight_space(start, goal, num_samples=3)

        # Original weights should be preserved
        assert optimizer_calm.weights == original_weights

    def test_scan_weight_space_single_sample(self, optimizer_calm):
        """Test weight scanning with single sample."""
        start = Point(50, 50)
        goal = Point(450, 450)

        results = optimizer_calm.scan_weight_space(start, goal, num_samples=1)

        assert len(results) == 1

    # ==================== Numerical Stability ====================

    def test_optimize_extreme_weights_time(self, ship, calm_env):
        """Test optimization with extreme time weight."""
        optimizer = WeightedSumOptimizer(ship, calm_env, weights=(100.0, 1.0, 0.0))
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer.optimize(start, goal, use_astar=False)

        # Should still converge
        assert result.success is True

    def test_optimize_extreme_weights_fuel(self, ship, calm_env):
        """Test optimization with extreme fuel weight."""
        optimizer = WeightedSumOptimizer(ship, calm_env, weights=(1.0, 100.0, 0.0))
        start = Point(50, 50)
        goal = Point(450, 450)

        result = optimizer.optimize(start, goal, use_astar=False)

        assert result.success is True

    # ==================== Reproducibility ====================

    def test_optimize_reproducible(self, optimizer_calm):
        """Test optimization produces consistent results."""
        start = Point(50, 50)
        goal = Point(450, 450)

        result1 = optimizer_calm.optimize(start, goal, use_astar=False, initial_speed=15.0)
        result2 = optimizer_calm.optimize(start, goal, use_astar=False, initial_speed=15.0)

        # Should get same result
        assert result1.speed == pytest.approx(result2.speed, rel=0.01)
        assert result1.objectives.fuel_liters == pytest.approx(result2.objectives.fuel_liters, rel=0.01)
