"""Wind resistance (windage) calculation.

This module implements aerodynamic drag resistance from wind acting on the
ship's above-water structure (windage area).

References:
    Blendermann, W. (1994). "Parameter identification of wind loads on ships."
    Journal of Wind Engineering and Industrial Aerodynamics, 51(3), 339-351.
"""

import math
from typing import Optional

from ..core import OperatingConditions, ShipParameters
from ..utils.constants import AIR_DENSITY


class WindResistance:
    """Calculate wind resistance (aerodynamic drag) on ship superstructure.

    Wind resistance is the aerodynamic drag force acting on the above-water
    portion of the ship. It depends on:
    - Relative wind speed (ship speed + wind speed vector)
    - Wind direction relative to ship heading
    - Frontal windage area
    - Aerodynamic drag coefficient

    The force is calculated using:
        R_wind = 0.5 × ρ_air × Cd × A_frontal × V_rel²

    where:
        - ρ_air: Air density (kg/m³)
        - Cd: Drag coefficient (dimensionless, ~0.4-0.8 for ships)
        - A_frontal: Frontal windage area (m²)
        - V_rel: Relative wind velocity (m/s)

    Example:
        >>> from ship_performance.core import ShipParameters, OperatingConditions
        >>> from ship_performance.resistance import WindResistance
        >>>
        >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
        >>> # 10 m/s wind at 45 degrees
        >>> conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=45)
        >>> calc = WindResistance()
        >>> R_wind = calc.calculate(ship, conditions)
        >>> print(f"Wind resistance: {R_wind/1000:.1f} kN")
    """

    def __init__(
        self,
        drag_coefficient: Optional[float] = None,
        air_density: float = AIR_DENSITY,
    ):
        """Initialize wind resistance calculator.

        Args:
            drag_coefficient: Aerodynamic drag coefficient Cd. If None, estimated.
                Typical values: 0.4 (streamlined) to 0.8 (bluff containers)
            air_density: Air density in kg/m³ (default: 1.225 at sea level, 15°C)
        """
        self._drag_coefficient = drag_coefficient
        self._air_density = air_density

    @property
    def name(self) -> str:
        """Component name for reporting."""
        return "Wind Resistance (Windage)"

    def calculate(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate wind resistance in Newtons.

        Args:
            ship: Ship parameters including frontal windage area
            conditions: Operating conditions including wind speed and direction

        Returns:
            Wind resistance in Newtons (N)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=45)
            >>> calc = WindResistance()
            >>> R_wind = calc.calculate(ship, conditions)
        """
        # If no wind, return zero
        if conditions.wind_speed == 0:
            return 0.0

        # Calculate relative wind velocity
        v_rel = self._calculate_relative_wind_speed(conditions)

        # Get drag coefficient
        cd = self._drag_coefficient or self._estimate_drag_coefficient(
            ship, conditions.wind_angle
        )

        # Wind resistance: R = 0.5 × ρ × Cd × A × V²
        resistance = (
            0.5 * self._air_density * cd * ship.frontal_area * v_rel**2
        )

        return max(0.0, resistance)

    def _calculate_relative_wind_speed(
        self, conditions: OperatingConditions
    ) -> float:
        """Calculate relative wind velocity magnitude.

        Combines ship velocity and wind velocity vectors to get relative wind.

        Maritime convention: wind_angle is direction wind is FROM
        - 0° = head wind (wind from dead ahead)
        - 90° = beam wind from starboard
        - 180° = following wind (wind from astern)

        The relative (apparent) wind magnitude is:
            V_rel² = V_ship² + V_wind² + 2×V_ship×V_wind×cos(θ)

        This accounts for:
        - Head wind (θ=0°): V_rel = V_ship + V_wind (they add)
        - Following wind (θ=180°): V_rel = |V_ship - V_wind| (they subtract)
        - Beam wind (θ=90°): V_rel = sqrt(V_ship² + V_wind²) (perpendicular)

        Args:
            conditions: Operating conditions with ship speed and wind

        Returns:
            Relative wind speed in m/s
        """
        v_ship = conditions.speed_ms
        v_wind = conditions.wind_speed
        theta_rad = math.radians(conditions.wind_angle)

        # Relative wind magnitude using vector addition
        # Wind FROM angle θ means wind velocity is -V_wind * [cos(θ), sin(θ)]
        # Relative wind = True wind - Ship velocity
        # |V_rel|² = V_wind² + V_ship² + 2×V_wind×V_ship×cos(θ)
        v_rel_squared = v_wind**2 + v_ship**2 + 2 * v_wind * v_ship * math.cos(theta_rad)

        # Ensure non-negative (can be slightly negative due to numerical errors)
        v_rel_squared = max(0.0, v_rel_squared)

        return math.sqrt(v_rel_squared)

    def _estimate_drag_coefficient(
        self, ship: ShipParameters, wind_angle: float
    ) -> float:
        """Estimate aerodynamic drag coefficient based on ship type and wind angle.

        Cd varies with wind direction:
        - Head/stern winds: Lower Cd (more streamlined profile)
        - Beam winds: Higher Cd (larger profile area)

        Typical values:
        - Tankers, bulk carriers: Cd ≈ 0.5-0.6
        - Container ships (loaded): Cd ≈ 0.6-0.8
        - Cruise ships: Cd ≈ 0.4-0.6
        - General cargo: Cd ≈ 0.5-0.7

        Args:
            ship: Ship parameters
            wind_angle: Wind direction relative to heading (degrees, 0-180)

        Returns:
            Estimated drag coefficient (dimensionless)
        """
        # Base drag coefficient (beam wind)
        # Higher for fuller ships with more superstructure
        if ship.block_coefficient > 0.75:
            # Full tanker/bulk carrier
            cd_base = 0.60
        elif ship.block_coefficient > 0.65:
            # Typical cargo
            cd_base = 0.65
        else:
            # Fine container ship
            cd_base = 0.70  # Higher due to containers above deck

        # Adjust for wind angle
        # Cd is maximum at beam winds (90°), minimum at head/stern
        theta_rad = math.radians(wind_angle)
        angle_factor = 0.7 + 0.3 * abs(math.sin(theta_rad))  # 0.7 to 1.0

        cd = cd_base * angle_factor

        # Ensure reasonable bounds
        cd = max(0.4, min(cd, 0.9))

        return cd

    def _calculate_apparent_wind_angle(
        self, conditions: OperatingConditions
    ) -> float:
        """Calculate apparent (relative) wind angle.

        The apparent wind angle differs from true wind angle due to ship motion.

        Args:
            conditions: Operating conditions

        Returns:
            Apparent wind angle in degrees (0-180)
        """
        v_ship = conditions.speed_ms
        v_wind = conditions.wind_speed
        theta_true = math.radians(conditions.wind_angle)

        # If no wind or no ship speed, use true wind angle
        if v_wind == 0 or v_ship == 0:
            return conditions.wind_angle

        # Components of wind velocity in ship frame
        # x: along ship axis (positive forward)
        # y: perpendicular to ship (positive to starboard)
        wind_x = v_wind * math.cos(theta_true)
        wind_y = v_wind * math.sin(theta_true)

        # Relative wind components (subtract ship velocity)
        rel_x = wind_x - v_ship
        rel_y = wind_y

        # Apparent wind angle
        theta_apparent = math.atan2(abs(rel_y), abs(rel_x))

        return math.degrees(theta_apparent)

    def breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, float]:
        """Get detailed breakdown of wind resistance components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary with components:
                - true_wind_speed: True wind speed (m/s)
                - relative_wind_speed: Relative wind speed (m/s)
                - true_wind_angle: True wind direction (degrees)
                - apparent_wind_angle: Apparent wind direction (degrees)
                - drag_coefficient: Cd
                - frontal_area: Windage area (m²)
                - resistance: Wind resistance (N)
        """
        v_rel = self._calculate_relative_wind_speed(conditions)
        cd = self._drag_coefficient or self._estimate_drag_coefficient(
            ship, conditions.wind_angle
        )
        theta_apparent = self._calculate_apparent_wind_angle(conditions)
        resistance = self.calculate(ship, conditions)

        return {
            "true_wind_speed": conditions.wind_speed,
            "relative_wind_speed": v_rel,
            "true_wind_angle": conditions.wind_angle,
            "apparent_wind_angle": theta_apparent,
            "drag_coefficient": cd,
            "frontal_area": ship.frontal_area,
            "resistance": resistance,
        }
