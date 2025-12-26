"""
Constraint checking for route optimization.

Validates routes against operational constraints: storm avoidance,
speed limits, arrival time windows, and restricted zones.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.utils.geometry import Point, point_to_segment_distance
from src.models.navigation_grid import NavigationEnvironment
from src.models.ship_model import ShipDynamics


@dataclass
class TimeWindow:
    """Arrival time window constraint.

    Attributes:
        min_hours: Minimum acceptable arrival time (hours from start)
        max_hours: Maximum acceptable arrival time (hours from start)
    """
    min_hours: Optional[float] = None
    max_hours: Optional[float] = None

    def is_satisfied(self, arrival_time: float) -> bool:
        """Check if arrival time satisfies window.

        Args:
            arrival_time: Actual arrival time in hours

        Returns:
            True if time is within window (or window not specified)
        """
        if self.min_hours is not None and arrival_time < self.min_hours:
            return False
        if self.max_hours is not None and arrival_time > self.max_hours:
            return False
        return True

    def __str__(self) -> str:
        """Human-readable representation."""
        min_str = f"{self.min_hours:.1f}" if self.min_hours is not None else "—"
        max_str = f"{self.max_hours:.1f}" if self.max_hours is not None else "—"
        return f"TimeWindow([{min_str}, {max_str}] hours)"


@dataclass
class ConstraintViolation:
    """Records a constraint violation.

    Attributes:
        constraint_type: Type of constraint violated
        description: Human-readable description
        severity: 'critical' or 'warning'
        location: Optional location where violation occurs
    """
    constraint_type: str
    description: str
    severity: str = 'critical'
    location: Optional[Point] = None

    def __str__(self) -> str:
        """Human-readable representation."""
        loc_str = f" at {self.location}" if self.location else ""
        return f"[{self.severity.upper()}] {self.constraint_type}: {self.description}{loc_str}"


class ConstraintChecker:
    """Validates routes against operational constraints."""

    def __init__(self,
                 environment: NavigationEnvironment,
                 ship: ShipDynamics,
                 time_window: Optional[TimeWindow] = None):
        """Initialize constraint checker.

        Args:
            environment: NavigationEnvironment with constraints
            ship: ShipDynamics for speed validation
            time_window: Optional arrival time window
        """
        self.env = environment
        self.ship = ship
        self.time_window = time_window or TimeWindow()

    def check_route(self,
                    waypoints: List[Point],
                    speed: float,
                    voyage_time: float) -> Tuple[bool, List[ConstraintViolation]]:
        """Check if route satisfies all constraints.

        Args:
            waypoints: Route waypoints
            speed: Ship speed in knots
            voyage_time: Total voyage time in hours

        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []

        # Check speed bounds
        if not self._check_speed(speed):
            violations.append(ConstraintViolation(
                constraint_type='speed_limits',
                description=f"Speed {speed:.2f} kn outside range [{self.ship.specs.v_min}, {self.ship.specs.v_max}]",
                severity='critical'
            ))

        # Check time window
        if not self.time_window.is_satisfied(voyage_time):
            violations.append(ConstraintViolation(
                constraint_type='time_window',
                description=f"Arrival time {voyage_time:.2f}h violates window {self.time_window}",
                severity='critical'
            ))

        # Check storm avoidance
        storm_violations = self._check_storm_avoidance(waypoints)
        violations.extend(storm_violations)

        # Check navigability (restricted zones)
        nav_violations = self._check_navigability(waypoints)
        violations.extend(nav_violations)

        is_feasible = all(v.severity != 'critical' for v in violations)

        return (is_feasible, violations)

    def _check_speed(self, speed: float) -> bool:
        """Check if speed is within operational limits.

        Args:
            speed: Ship speed in knots

        Returns:
            True if speed is valid
        """
        return self.ship.specs.v_min <= speed <= self.ship.specs.v_max

    def _check_storm_avoidance(self, waypoints: List[Point]) -> List[ConstraintViolation]:
        """Check if route maintains safe distance from storms.

        Args:
            waypoints: Route waypoints

        Returns:
            List of storm avoidance violations
        """
        violations = []
        min_safe_distance = self.env.constraints.min_storm_distance
        wave_threshold = 5.0  # Storm threshold in meters

        # Check each waypoint
        for i, waypoint in enumerate(waypoints):
            distance = self.env.distance_to_nearest_storm(waypoint, wave_threshold)

            if distance < min_safe_distance:
                violations.append(ConstraintViolation(
                    constraint_type='storm_avoidance',
                    description=(
                        f"Waypoint {i} too close to storm "
                        f"(distance: {distance:.1f} nm, minimum: {min_safe_distance:.1f} nm)"
                    ),
                    severity='critical',
                    location=waypoint
                ))

        # Also check segments (storms might be between waypoints)
        for i in range(len(waypoints) - 1):
            segment_violations = self._check_segment_storm_avoidance(
                waypoints[i],
                waypoints[i + 1],
                min_safe_distance,
                wave_threshold
            )
            violations.extend(segment_violations)

        return violations

    def _check_segment_storm_avoidance(self,
                                        start: Point,
                                        end: Point,
                                        min_distance: float,
                                        wave_threshold: float,
                                        num_samples: int = 5) -> List[ConstraintViolation]:
        """Check storm avoidance along a segment.

        Args:
            start: Segment start
            end: Segment end
            min_distance: Minimum safe distance from storms
            wave_threshold: Wave height threshold for storms
            num_samples: Number of points to sample along segment

        Returns:
            List of violations found
        """
        violations = []

        for i in range(1, num_samples):  # Skip endpoints (already checked)
            fraction = i / num_samples
            sample_point = Point(
                start.x + fraction * (end.x - start.x),
                start.y + fraction * (end.y - start.y)
            )

            distance = self.env.distance_to_nearest_storm(sample_point, wave_threshold)

            if distance < min_distance:
                violations.append(ConstraintViolation(
                    constraint_type='storm_avoidance',
                    description=(
                        f"Route segment passes too close to storm "
                        f"(distance: {distance:.1f} nm, minimum: {min_distance:.1f} nm)"
                    ),
                    severity='critical',
                    location=sample_point
                ))
                break  # One violation per segment sufficient

        return violations

    def _check_navigability(self, waypoints: List[Point]) -> List[ConstraintViolation]:
        """Check if all waypoints are in navigable areas.

        Args:
            waypoints: Route waypoints

        Returns:
            List of navigability violations
        """
        violations = []

        for i, waypoint in enumerate(waypoints):
            if not self.env.is_navigable(waypoint):
                # Determine reason for non-navigability
                _, wave_height = self.env.weather.get_weather_at_point(waypoint)

                if wave_height > self.env.constraints.max_wave_height:
                    reason = (
                        f"excessive wave height ({wave_height:.1f}m, "
                        f"max: {self.env.constraints.max_wave_height:.1f}m)"
                    )
                else:
                    reason = "restricted zone or out of bounds"

                violations.append(ConstraintViolation(
                    constraint_type='navigability',
                    description=f"Waypoint {i} not navigable: {reason}",
                    severity='critical',
                    location=waypoint
                ))

        return violations

    def check_speed_profile(self,
                            speeds: List[float]) -> Tuple[bool, List[ConstraintViolation]]:
        """Check if speed profile satisfies constraints.

        Args:
            speeds: Speed for each segment

        Returns:
            Tuple of (is_feasible, violations)
        """
        violations = []

        for i, speed in enumerate(speeds):
            if not self._check_speed(speed):
                violations.append(ConstraintViolation(
                    constraint_type='speed_limits',
                    description=(
                        f"Segment {i} speed {speed:.2f} kn outside range "
                        f"[{self.ship.specs.v_min}, {self.ship.specs.v_max}]"
                    ),
                    severity='critical'
                ))

        is_feasible = len(violations) == 0

        return (is_feasible, violations)

    def get_constraint_summary(self) -> dict:
        """Get summary of active constraints.

        Returns:
            Dictionary with constraint information
        """
        return {
            'speed': {
                'min_kn': self.ship.specs.v_min,
                'max_kn': self.ship.specs.v_max
            },
            'storm_avoidance': {
                'min_distance_nm': self.env.constraints.min_storm_distance,
                'wave_threshold_m': self.env.constraints.max_wave_height
            },
            'time_window': {
                'min_hours': self.time_window.min_hours,
                'max_hours': self.time_window.max_hours,
                'active': (
                    self.time_window.min_hours is not None or
                    self.time_window.max_hours is not None
                )
            },
            'restricted_zones': {
                'count': len(self.env.constraints.restricted_zones)
            }
        }


def create_baseline_route(start: Point,
                          end: Point,
                          num_waypoints: int = 2) -> List[Point]:
    """Create baseline direct route for comparison.

    Args:
        start: Start position
        end: End position
        num_waypoints: Number of waypoints (default: just start and end)

    Returns:
        List of waypoints forming straight line
    """
    waypoints = []

    for i in range(num_waypoints):
        fraction = i / (num_waypoints - 1) if num_waypoints > 1 else 0

        waypoint = Point(
            start.x + fraction * (end.x - start.x),
            start.y + fraction * (end.y - start.y)
        )
        waypoints.append(waypoint)

    return waypoints
