"""
Route evaluation for computing objectives (time, fuel, emissions).

Evaluates route quality by integrating fuel consumption, time, and emissions
along the path considering weather conditions.
"""

from typing import List, Tuple, Union
from dataclasses import dataclass
import numpy as np

from src.utils.geometry import Point
from src.models.ship_model import ShipDynamics
from src.models.navigation_grid import NavigationEnvironment


@dataclass
class RouteObjectives:
    """Objectives for a ship route.

    Attributes:
        time_hours: Total voyage time in hours
        fuel_liters: Total fuel consumption in liters
        emissions_kg: Total CO2 emissions in kilograms
        distance_nm: Total distance traveled in nautical miles
    """
    time_hours: float
    fuel_liters: float
    emissions_kg: float
    distance_nm: float

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return (time, fuel, emissions) tuple for optimization."""
        return (self.time_hours, self.fuel_liters, self.emissions_kg)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"RouteObjectives(time={self.time_hours:.2f}h, "
                f"fuel={self.fuel_liters:.1f}L, "
                f"CO2={self.emissions_kg:.1f}kg, "
                f"distance={self.distance_nm:.1f}nm)")


class RouteEvaluator:
    """Evaluates route quality (time, fuel, emissions).

    Integrates ship dynamics and weather conditions along the route
    to compute total objectives.
    """

    def __init__(self,
                 ship: ShipDynamics,
                 environment: NavigationEnvironment):
        """Initialize route evaluator.

        Args:
            ship: ShipDynamics model for fuel consumption
            environment: NavigationEnvironment with weather data
        """
        self.ship = ship
        self.env = environment

    def evaluate_route(self,
                       waypoints: List[Point],
                       speed: float) -> RouteObjectives:
        """Evaluate route with constant speed.

        Args:
            waypoints: Route as list of waypoints in nautical miles
            speed: Constant ship speed in knots

        Returns:
            RouteObjectives with time, fuel, emissions, distance

        Raises:
            ValueError: If speed outside operational range or waypoints invalid
        """
        if len(waypoints) < 2:
            raise ValueError("Route must have at least 2 waypoints")

        if speed < self.ship.specs.v_min or speed > self.ship.specs.v_max:
            raise ValueError(
                f"Speed {speed:.2f} kn outside range "
                f"[{self.ship.specs.v_min}, {self.ship.specs.v_max}]"
            )

        total_time = 0.0
        total_fuel = 0.0
        total_emissions = 0.0
        total_distance = 0.0

        # Integrate over segments
        for i in range(1, len(waypoints)):
            segment_start = waypoints[i - 1]
            segment_end = waypoints[i]

            # Calculate segment objectives
            seg_time, seg_fuel, seg_co2, seg_dist = self._evaluate_segment(
                segment_start,
                segment_end,
                speed
            )

            total_time += seg_time
            total_fuel += seg_fuel
            total_emissions += seg_co2
            total_distance += seg_dist

        return RouteObjectives(
            time_hours=total_time,
            fuel_liters=total_fuel,
            emissions_kg=total_emissions,
            distance_nm=total_distance
        )

    def evaluate_route_variable_speed(self,
                                       waypoints: List[Point],
                                       speeds: Union[List[float], np.ndarray]) -> RouteObjectives:
        """Evaluate route with variable speed between waypoints.

        Args:
            waypoints: Route as list of waypoints
            speeds: Speed for each segment (length = len(waypoints) - 1)

        Returns:
            RouteObjectives with time, fuel, emissions, distance

        Raises:
            ValueError: If speeds length mismatch or invalid speeds
        """
        if len(waypoints) < 2:
            raise ValueError("Route must have at least 2 waypoints")

        num_segments = len(waypoints) - 1
        if len(speeds) != num_segments:
            raise ValueError(
                f"Speeds length ({len(speeds)}) must match number of segments ({num_segments})"
            )

        # Check all speeds valid
        for i, speed in enumerate(speeds):
            if speed < self.ship.specs.v_min or speed > self.ship.specs.v_max:
                raise ValueError(
                    f"Speed {speed:.2f} kn at segment {i} outside range "
                    f"[{self.ship.specs.v_min}, {self.ship.specs.v_max}]"
                )

        total_time = 0.0
        total_fuel = 0.0
        total_emissions = 0.0
        total_distance = 0.0

        # Integrate over segments
        for i in range(num_segments):
            segment_start = waypoints[i]
            segment_end = waypoints[i + 1]
            speed = speeds[i]

            seg_time, seg_fuel, seg_co2, seg_dist = self._evaluate_segment(
                segment_start,
                segment_end,
                speed
            )

            total_time += seg_time
            total_fuel += seg_fuel
            total_emissions += seg_co2
            total_distance += seg_dist

        return RouteObjectives(
            time_hours=total_time,
            fuel_liters=total_fuel,
            emissions_kg=total_emissions,
            distance_nm=total_distance
        )

    def _evaluate_segment(self,
                          start: Point,
                          end: Point,
                          speed: float) -> Tuple[float, float, float, float]:
        """Evaluate a single route segment.

        Uses midpoint weather for the entire segment (simple approximation).

        Args:
            start: Segment start point
            end: Segment end point
            speed: Ship speed in knots

        Returns:
            Tuple of (time_h, fuel_L, co2_kg, distance_nm)
        """
        # Calculate distance
        distance = start.distance_to(end)

        # Sample weather at segment midpoint
        midpoint = Point(
            (start.x + end.x) / 2.0,
            (start.y + end.y) / 2.0
        )
        wind_speed, wave_height = self.env.weather.get_weather_at_point(midpoint)

        # Calculate objectives using ship model
        time_h, fuel_L, co2_kg = self.ship.fuel_for_segment(
            distance,
            speed,
            wind_speed,
            wave_height
        )

        return (time_h, fuel_L, co2_kg, distance)

    def evaluate_with_refined_sampling(self,
                                        waypoints: List[Point],
                                        speed: float,
                                        samples_per_segment: int = 5) -> RouteObjectives:
        """Evaluate route with multiple samples per segment for accuracy.

        Args:
            waypoints: Route waypoints
            speed: Constant ship speed
            samples_per_segment: Number of weather samples per segment

        Returns:
            RouteObjectives with refined evaluation
        """
        if len(waypoints) < 2:
            raise ValueError("Route must have at least 2 waypoints")

        total_time = 0.0
        total_fuel = 0.0
        total_emissions = 0.0
        total_distance = 0.0

        for i in range(1, len(waypoints)):
            segment_start = waypoints[i - 1]
            segment_end = waypoints[i]
            segment_dist = segment_start.distance_to(segment_end)

            # Sample weather at multiple points along segment
            segment_fuel = 0.0
            segment_emissions = 0.0

            for j in range(samples_per_segment):
                fraction = (j + 0.5) / samples_per_segment
                sample_point = Point(
                    segment_start.x + fraction * (segment_end.x - segment_start.x),
                    segment_start.y + fraction * (segment_end.y - segment_start.y)
                )

                wind_speed, wave_height = self.env.weather.get_weather_at_point(sample_point)

                # Subsegment distance
                subseg_dist = segment_dist / samples_per_segment

                # Calculate fuel for subsegment
                _, subseg_fuel, subseg_co2 = self.ship.fuel_for_segment(
                    subseg_dist,
                    speed,
                    wind_speed,
                    wave_height
                )

                segment_fuel += subseg_fuel
                segment_emissions += subseg_co2

            # Time for segment
            segment_time = segment_dist / speed

            total_time += segment_time
            total_fuel += segment_fuel
            total_emissions += segment_emissions
            total_distance += segment_dist

        return RouteObjectives(
            time_hours=total_time,
            fuel_liters=total_fuel,
            emissions_kg=total_emissions,
            distance_nm=total_distance
        )

    def compare_routes(self,
                       route1: List[Point],
                       route2: List[Point],
                       speed: float) -> dict:
        """Compare two routes at the same speed.

        Args:
            route1: First route waypoints
            route2: Second route waypoints
            speed: Speed for both routes

        Returns:
            Dictionary with comparison results
        """
        obj1 = self.evaluate_route(route1, speed)
        obj2 = self.evaluate_route(route2, speed)

        time_diff_pct = ((obj2.time_hours - obj1.time_hours) / obj1.time_hours) * 100
        fuel_diff_pct = ((obj2.fuel_liters - obj1.fuel_liters) / obj1.fuel_liters) * 100
        co2_diff_pct = ((obj2.emissions_kg - obj1.emissions_kg) / obj1.emissions_kg) * 100

        return {
            'route1': obj1,
            'route2': obj2,
            'time_diff_hours': obj2.time_hours - obj1.time_hours,
            'fuel_diff_liters': obj2.fuel_liters - obj1.fuel_liters,
            'co2_diff_kg': obj2.emissions_kg - obj1.emissions_kg,
            'time_diff_pct': time_diff_pct,
            'fuel_diff_pct': fuel_diff_pct,
            'co2_diff_pct': co2_diff_pct,
            'route1_better': (
                obj1.fuel_liters < obj2.fuel_liters or
                obj1.time_hours < obj2.time_hours
            )
        }


def create_direct_route(start: Point, end: Point, num_waypoints: int = 2) -> List[Point]:
    """Create a direct route (straight line) between two points.

    Args:
        start: Start position
        end: End position
        num_waypoints: Number of waypoints including start and end

    Returns:
        List of waypoints from start to end
    """
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least 2")

    waypoints = []
    for i in range(num_waypoints):
        fraction = i / (num_waypoints - 1)
        waypoint = Point(
            start.x + fraction * (end.x - start.x),
            start.y + fraction * (end.y - start.y)
        )
        waypoints.append(waypoint)

    return waypoints
