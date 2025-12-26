"""
Ship dynamics and fuel consumption model.

Implements fuel consumption physics based on speed and weather conditions
using a simplified model: f(V, W, H) = a*V³ + b*W² + c*H + d
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ShipSpecifications:
    """Physical specifications and operational parameters of a ship.

    Attributes:
        name: Ship identifier
        v_min: Minimum operational speed (knots)
        v_max: Maximum operational speed (knots)
        fuel_coef_speed: Coefficient 'a' for speed term (L/h per knot³)
        fuel_coef_wind: Coefficient 'b' for wind term (L/h per knot²)
        fuel_coef_wave: Coefficient 'c' for wave term (L/h per meter)
        fuel_base: Base fuel consumption 'd' (L/h)
        emission_factor: CO2 emissions per liter fuel (kg/L)
    """
    name: str
    v_min: float  # knots
    v_max: float  # knots
    fuel_coef_speed: float  # a: L/h per knot³
    fuel_coef_wind: float  # b: L/h per knot²
    fuel_coef_wave: float  # c: L/h per meter
    fuel_base: float  # d: L/h
    emission_factor: float = 2.8  # kg CO2/L (typical for HFO)

    def __post_init__(self):
        """Validate ship specifications."""
        if self.v_min <= 0:
            raise ValueError(f"v_min must be positive, got {self.v_min}")
        if self.v_max <= self.v_min:
            raise ValueError(f"v_max ({self.v_max}) must be greater than v_min ({self.v_min})")
        if self.fuel_coef_speed < 0:
            raise ValueError(f"fuel_coef_speed must be non-negative, got {self.fuel_coef_speed}")


class ShipDynamics:
    """Computes ship fuel consumption and emissions.

    The fuel consumption model is:
        fuel_rate(V, W, H) = a*V³ + b*W² + c*H + d

    where:
        V: ship speed (knots)
        W: wind speed (knots)
        H: significant wave height (meters)
        a, b, c, d: calibrated coefficients
    """

    def __init__(self, specs: ShipSpecifications):
        """Initialize ship dynamics model.

        Args:
            specs: Ship specifications including fuel coefficients
        """
        self.specs = specs

    def fuel_rate(self,
                  velocity: float,
                  wind_speed: float = 0.0,
                  wave_height: float = 0.0) -> float:
        """Calculate instantaneous fuel consumption rate.

        Model: fuel_rate = a*V³ + b*W² + c*H + d

        Args:
            velocity: Ship speed in knots
            wind_speed: Wind speed in knots (default: 0)
            wave_height: Significant wave height in meters (default: 0)

        Returns:
            Fuel consumption rate in liters/hour

        Raises:
            ValueError: If velocity outside [v_min, v_max]
        """
        if velocity < self.specs.v_min:
            raise ValueError(
                f"Velocity {velocity:.2f} kn below minimum {self.specs.v_min:.2f} kn"
            )
        if velocity > self.specs.v_max:
            raise ValueError(
                f"Velocity {velocity:.2f} kn exceeds maximum {self.specs.v_max:.2f} kn"
            )

        # Cubic speed term (dominant)
        speed_term = self.specs.fuel_coef_speed * (velocity ** 3)

        # Quadratic wind term (headwind resistance)
        wind_term = self.specs.fuel_coef_wind * (wind_speed ** 2)

        # Linear wave term (added resistance)
        wave_term = self.specs.fuel_coef_wave * wave_height

        # Base consumption
        base_term = self.specs.fuel_base

        fuel = speed_term + wind_term + wave_term + base_term

        return max(0.0, fuel)  # Ensure non-negative

    def emissions_rate(self,
                       velocity: float,
                       wind_speed: float = 0.0,
                       wave_height: float = 0.0) -> float:
        """Calculate instantaneous CO2 emissions rate.

        Args:
            velocity: Ship speed in knots
            wind_speed: Wind speed in knots (default: 0)
            wave_height: Significant wave height in meters (default: 0)

        Returns:
            CO2 emissions rate in kg/hour
        """
        fuel = self.fuel_rate(velocity, wind_speed, wave_height)
        return fuel * self.specs.emission_factor

    def optimal_speed_calm_weather(self) -> float:
        """Calculate fuel-optimal speed in calm weather (W=0, H=0).

        In calm conditions, only the speed term matters:
            f(V) = a*V³ + d

        For fixed distance D, total fuel is:
            F_total = f(V) * (D/V) = a*V² + d*D/V

        Minimizing dF/dV = 0 gives:
            V_opt = (d / (2*a))^(1/3)

        Returns:
            Optimal speed in knots (clamped to [v_min, v_max])
        """
        if self.specs.fuel_coef_speed == 0:
            # No speed penalty, go max speed
            return self.specs.v_max

        # Analytical optimum
        v_opt = (self.specs.fuel_base / (2 * self.specs.fuel_coef_speed)) ** (1/3)

        # Clamp to operational range
        return np.clip(v_opt, self.specs.v_min, self.specs.v_max)

    def fuel_for_segment(self,
                         distance_nm: float,
                         velocity: float,
                         wind_speed: float = 0.0,
                         wave_height: float = 0.0) -> tuple[float, float, float]:
        """Calculate fuel, time, and emissions for a route segment.

        Args:
            distance_nm: Segment length in nautical miles
            velocity: Ship speed in knots
            wind_speed: Wind speed in knots
            wave_height: Wave height in meters

        Returns:
            Tuple of (time_hours, fuel_liters, co2_kg)
        """
        time_hours = distance_nm / velocity
        fuel_rate_value = self.fuel_rate(velocity, wind_speed, wave_height)
        fuel_liters = fuel_rate_value * time_hours
        co2_kg = fuel_liters * self.specs.emission_factor

        return (time_hours, fuel_liters, co2_kg)


def create_default_ship() -> ShipDynamics:
    """Create a default generic cargo ship.

    Based on typical values for medium-sized cargo vessels:
    - Speed range: 8-18 knots
    - Fuel consumption: ~2000-4000 L/h at cruise speed
    - Weather penalties: moderate sensitivity

    Returns:
        ShipDynamics instance with default specifications
    """
    specs = ShipSpecifications(
        name="Generic Cargo Vessel",
        v_min=8.0,  # knots
        v_max=18.0,  # knots
        fuel_coef_speed=0.5,  # L/h per knot³ (will be calibrated)
        fuel_coef_wind=0.8,  # L/h per knot²
        fuel_coef_wave=15.0,  # L/h per meter
        fuel_base=500.0,  # L/h
        emission_factor=2.8  # kg CO2/L (HFO)
    )
    return ShipDynamics(specs)
