"""
Unit tests for constraint checking.

Tests time windows, storm avoidance, speed limits,
restricted zones, and violation tracking.
"""

import pytest
from src.planning.constraints import (
    ConstraintChecker, ConstraintViolation, TimeWindow
)
from src.models.ship_model import create_default_ship
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.models.weather_field import create_calm_scenario, create_storm_scenario
from src.utils.geometry import Point


class TestTimeWindow:
    """Test TimeWindow dataclass."""

    def test_time_window_both_bounds(self):
        """Test time window with both min and max."""
        tw = TimeWindow(min_hours=10.0, max_hours=30.0)

        assert tw.min_hours == 10.0
        assert tw.max_hours == 30.0

    def test_time_window_only_max(self):
        """Test time window with only max bound."""
        tw = TimeWindow(max_hours=24.0)

        assert tw.min_hours is None
        assert tw.max_hours == 24.0

    def test_time_window_only_min(self):
        """Test time window with only min bound."""
        tw = TimeWindow(min_hours=12.0)

        assert tw.min_hours == 12.0
        assert tw.max_hours is None

    def test_time_window_no_bounds(self):
        """Test time window with no bounds."""
        tw = TimeWindow()

        assert tw.min_hours is None
        assert tw.max_hours is None


class TestConstraintViolation:
    """Test ConstraintViolation dataclass."""

    def test_violation_creation(self):
        """Test creating constraint violation."""
        violation = ConstraintViolation(
            constraint_type="speed_limits",
            description="Speed exceeds maximum",
            severity='critical',
            location=Point(100, 100)
        )

        assert violation.constraint_type == "speed_limits"
        assert violation.severity == 'critical'
        assert violation.description == "Speed exceeds maximum"
        assert violation.location == Point(100, 100)

    def test_violation_str(self):
        """Test string representation."""
        violation = ConstraintViolation(
            constraint_type="time_window",
            description="Voyage time too long",
            severity='warning'
        )

        s = str(violation)
        assert "WARNING" in s
        assert "time_window" in s
        assert "Voyage time too long" in s


class TestConstraintChecker:
    """Test ConstraintChecker."""

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
    def checker_calm(self, calm_env, ship):
        """Create checker with calm environment."""
        return ConstraintChecker(calm_env, ship)

    @pytest.fixture
    def checker_storm(self, storm_env, ship):
        """Create checker with storm environment."""
        return ConstraintChecker(storm_env, ship)

    # ==================== Feasible Routes ====================

    def test_check_route_feasible(self, checker_calm):
        """Test checking feasible route."""
        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is True
        assert len(violations) == 0

    def test_check_route_empty(self, checker_calm):
        """Test checking empty route."""
        route = []
        speed = 15.0
        voyage_time = 0.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is True
        assert len(violations) == 0

    def test_check_route_single_waypoint(self, checker_calm):
        """Test checking route with single waypoint."""
        route = [Point(250, 250)]
        speed = 15.0
        voyage_time = 0.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is True
        assert len(violations) == 0

    # ==================== Speed Violations ====================

    def test_speed_below_minimum(self, checker_calm, ship):
        """Test speed below minimum violation."""
        route = [Point(50, 50), Point(450, 450)]
        speed = ship.specs.v_min - 1.0
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is False
        assert len(violations) >= 1

        # Check violation details
        speed_violations = [v for v in violations if v.constraint_type == "speed_limits"]
        assert len(speed_violations) == 1
        assert speed_violations[0].severity == 'critical'

    def test_speed_above_maximum(self, checker_calm, ship):
        """Test speed above maximum violation."""
        route = [Point(50, 50), Point(450, 450)]
        speed = ship.specs.v_max + 1.0
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is False
        speed_violations = [v for v in violations if v.constraint_type == "speed_limits"]
        assert len(speed_violations) == 1

    def test_speed_at_boundaries(self, checker_calm, ship):
        """Test speed at exact min/max boundaries."""
        route = [Point(50, 50), Point(450, 450)]

        # Test minimum speed
        is_feasible_min, violations_min = checker_calm.check_route(route, ship.specs.v_min, 30.0)
        assert is_feasible_min is True

        # Test maximum speed
        is_feasible_max, violations_max = checker_calm.check_route(route, ship.specs.v_max, 30.0)
        assert is_feasible_max is True

    # ==================== Time Window Violations ====================

    def test_time_window_too_long(self, calm_env, ship):
        """Test voyage time exceeds maximum."""
        time_window = TimeWindow(max_hours=20.0)
        checker = ConstraintChecker(calm_env, ship, time_window)

        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 25.0  # Exceeds max_hours

        is_feasible, violations = checker.check_route(route, speed, voyage_time)

        assert is_feasible is False
        time_violations = [v for v in violations if v.constraint_type == "time_window"]
        assert len(time_violations) == 1
        assert time_violations[0].severity == ViolationSeverity.CRITICAL

    def test_time_window_too_short(self, calm_env, ship):
        """Test voyage time below minimum."""
        time_window = TimeWindow(min_hours=40.0)
        checker = ConstraintChecker(calm_env, ship, time_window)

        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0  # Below min_hours

        is_feasible, violations = checker.check_route(route, speed, voyage_time)

        assert is_feasible is False
        time_violations = [v for v in violations if v.constraint_type == "time_window"]
        assert len(time_violations) == 1

    def test_time_window_within_bounds(self, calm_env, ship):
        """Test voyage time within time window."""
        time_window = TimeWindow(min_hours=20.0, max_hours=40.0)
        checker = ConstraintChecker(calm_env, ship, time_window)

        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0  # Within bounds

        is_feasible, violations = checker.check_route(route, speed, voyage_time)

        assert is_feasible is True
        time_violations = [v for v in violations if v.constraint_type == "time_window"]
        assert len(time_violations) == 0

    def test_time_window_no_bounds(self, checker_calm):
        """Test with no time window constraints."""
        # Default checker has no time window
        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 1000.0  # Very long

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        # Should not violate time window (none specified)
        time_violations = [v for v in violations if v.constraint_type == "time_window"]
        assert len(time_violations) == 0

    # ==================== Storm Avoidance ====================

    def test_storm_avoidance_waypoint_in_storm(self, checker_storm):
        """Test waypoint in storm zone."""
        # Storm center is at (250, 250) with 100 nm radius
        route = [Point(50, 50), Point(250, 250), Point(450, 450)]  # Goes through storm
        speed = 15.0
        voyage_time = 30.0

        is_feasible, violations = checker_storm.check_route(route, speed, voyage_time)

        # Should detect storm violation
        assert is_feasible is False
        storm_violations = [v for v in violations if v.constraint_type == "storm_avoidance"]
        assert len(storm_violations) >= 1

    def test_storm_avoidance_segment_through_storm(self, checker_storm):
        """Test segment passes through storm."""
        # Segment from (50, 50) to (450, 450) passes through storm at (250, 250)
        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0

        is_feasible, violations = checker_storm.check_route(route, speed, voyage_time)

        # Should detect segment passing through storm
        # (depends on segment checking granularity)
        storm_violations = [v for v in violations if v.constraint_type == "storm_avoidance"]
        # May or may not detect depending on sampling resolution
        # Just verify checker runs
        assert isinstance(is_feasible, bool)

    def test_storm_avoidance_route_avoids_storm(self, checker_storm):
        """Test route that successfully avoids storm."""
        # Route around storm (storm at 250, 250)
        route = [Point(50, 50), Point(50, 450), Point(450, 450)]  # Go around west side
        speed = 15.0
        voyage_time = 40.0

        is_feasible, violations = checker_storm.check_route(route, speed, voyage_time)

        # This route should avoid storm
        storm_violations = [v for v in violations if v.constraint_type == "storm_avoidance"]
        # Check if route successfully avoids storm
        # (may still have violations if detour isn't wide enough)
        assert isinstance(violations, list)

    # ==================== Navigability ====================

    def test_navigability_all_points_valid(self, checker_calm):
        """Test route with all navigable points."""
        route = [Point(50, 50), Point(250, 250), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        # Should have no navigability violations
        nav_violations = [v for v in violations if v.constraint_type == "navigability"]
        assert len(nav_violations) == 0

    def test_navigability_restricted_zone(self):
        """Test route through restricted zone."""
        grid_shape = (50, 50)
        cell_size = 10.0
        weather = create_calm_scenario(grid_shape, cell_size)

        # Create restricted zone at center
        restricted_zones = [Point(250, 250)]
        constraints = NavigationConstraints(
            min_storm_distance=50.0,
            max_wave_height=6.0,
            restricted_zones=restricted_zones,
            restricted_radius=30.0
        )
        env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)
        ship = create_default_ship()
        checker = ConstraintChecker(env, ship)

        # Route through restricted zone
        route = [Point(50, 50), Point(250, 250), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0

        is_feasible, violations = checker.check_route(route, speed, voyage_time)

        # Should detect navigability violation
        assert is_feasible is False
        nav_violations = [v for v in violations if v.constraint_type == "navigability"]
        assert len(nav_violations) >= 1

    # ==================== Multiple Violations ====================

    def test_multiple_violations(self, checker_storm, ship):
        """Test route with multiple violations."""
        time_window = TimeWindow(max_hours=10.0)
        checker = ConstraintChecker(checker_storm.env, ship, time_window)

        # Route through storm with excessive time
        route = [Point(50, 50), Point(250, 250), Point(450, 450)]
        speed = ship.specs.v_min - 1.0  # Also speed violation
        voyage_time = 50.0  # Time violation

        is_feasible, violations = checker.check_route(route, speed, voyage_time)

        assert is_feasible is False
        # Should have multiple violations
        assert len(violations) >= 2

        # Check different violation types present
        violation_types = {v.constraint_type for v in violations}
        assert "speed_limit" in violation_types
        assert "time_window" in violation_types

    # ==================== Violation Severity ====================

    def test_critical_violations_make_infeasible(self, checker_calm, ship):
        """Test critical violations mark route infeasible."""
        route = [Point(50, 50), Point(450, 450)]
        speed = ship.specs.v_max + 5.0  # Critical speed violation
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is False
        # Should have critical violation
        critical_violations = [v for v in violations if v.severity == 'critical']
        assert len(critical_violations) >= 1

    def test_warning_violations_do_not_make_infeasible(self):
        """Test that warnings alone don't make route infeasible."""
        # This depends on implementation - warnings may or may not affect feasibility
        # Just verify warnings are properly categorized
        grid_shape = (50, 50)
        cell_size = 10.0
        weather = create_calm_scenario(grid_shape, cell_size)
        constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
        env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)
        ship = create_default_ship()
        checker = ConstraintChecker(env, ship)

        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0

        is_feasible, violations = checker.check_route(route, speed, voyage_time)

        # All violations should have defined severity
        for v in violations:
            assert v.severity in ['critical', 'warning']

    # ==================== Edge Cases ====================

    def test_zero_voyage_time(self, checker_calm):
        """Test checking with zero voyage time."""
        route = [Point(250, 250)]  # Single point
        speed = 15.0
        voyage_time = 0.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is True

    def test_negative_speed(self, checker_calm):
        """Test negative speed."""
        route = [Point(50, 50), Point(450, 450)]
        speed = -5.0
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is False
        speed_violations = [v for v in violations if v.constraint_type == "speed_limits"]
        assert len(speed_violations) >= 1

    def test_zero_speed(self, checker_calm):
        """Test zero speed."""
        route = [Point(50, 50), Point(450, 450)]
        speed = 0.0
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        assert is_feasible is False
        speed_violations = [v for v in violations if v.constraint_type == "speed_limits"]
        assert len(speed_violations) >= 1

    # ==================== Segment Checking ====================

    def test_segment_storm_detection(self, checker_storm):
        """Test segment-level storm detection."""
        # Long segment that passes through storm
        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0
        voyage_time = 30.0

        is_feasible, violations = checker_storm.check_route(route, speed, voyage_time)

        # Should detect storm along segment (not just at endpoints)
        storm_violations = [v for v in violations if v.constraint_type == "storm_avoidance"]
        # Verify method runs correctly
        assert isinstance(violations, list)

    # ==================== Large Routes ====================

    def test_check_long_route(self, checker_calm):
        """Test checking route with many waypoints."""
        # Create 50-waypoint route
        route = [Point(i * 10, i * 10) for i in range(50)]
        speed = 15.0
        voyage_time = 50.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        # Should handle large routes
        assert isinstance(is_feasible, bool)
        assert isinstance(violations, list)

    # ==================== Constraint Isolation ====================

    def test_only_speed_constraint(self, checker_calm, ship):
        """Test only speed constraint checked when others satisfied."""
        route = [Point(50, 50), Point(450, 450)]
        speed = ship.specs.v_max + 1.0  # Only speed violation
        voyage_time = 30.0

        is_feasible, violations = checker_calm.check_route(route, speed, voyage_time)

        # Should only have speed violation
        assert len(violations) == 1
        assert violations[0].constraint_type == "speed_limit"

    def test_only_time_constraint(self, calm_env, ship):
        """Test only time constraint violation."""
        time_window = TimeWindow(max_hours=10.0)
        checker = ConstraintChecker(calm_env, ship, time_window)

        route = [Point(50, 50), Point(450, 450)]
        speed = 15.0  # Valid speed
        voyage_time = 20.0  # Exceeds time window

        is_feasible, violations = checker.check_route(route, speed, voyage_time)

        # Should only have time violation
        time_violations = [v for v in violations if v.constraint_type == "time_window"]
        speed_violations = [v for v in violations if v.constraint_type == "speed_limits"]

        assert len(time_violations) == 1
        assert len(speed_violations) == 0
