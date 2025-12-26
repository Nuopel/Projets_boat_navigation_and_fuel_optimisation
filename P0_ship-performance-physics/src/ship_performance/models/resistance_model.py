"""Complete ship resistance model integrating all components.

This module provides a unified interface for calculating total ship resistance
by integrating all resistance components:
- Calm water resistance (friction + wave-making)
- Wind resistance (windage)
- Added wave resistance (from sea state)

The model provides comprehensive breakdown and reporting capabilities for
analyzing resistance contributions under various operating conditions.
"""

from typing import Optional

from ..core import OperatingConditions, ShipParameters
from ..resistance import (
    AddedWaveResistance,
    CalmWaterResistance,
    WindResistance,
)


class ShipResistanceModel:
    """Complete ship resistance model with all environmental components.

    This class integrates all resistance calculation components into a unified
    model for complete ship performance prediction. It calculates:
    - Calm water resistance (friction + wave-making)
    - Wind resistance (aerodynamic drag)
    - Added wave resistance (from sea state)
    - Total resistance (sum of all components)

    The model provides detailed breakdown reporting and supports various
    environmental conditions (calm, moderate weather, storms, etc.).

    Example:
        >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
        >>> conditions = OperatingConditions(speed=15, wind_speed=10, wind_angle=45,
        ...                                   wave_height=3, wave_period=10, wave_angle=30)
        >>> model = ShipResistanceModel()
        >>> R_total = model.calculate_total_resistance(ship, conditions)
        >>> breakdown = model.get_breakdown(ship, conditions)
        >>> print(f"Total resistance: {R_total/1000:.1f} kN")
        >>> print(f"  Calm water: {breakdown['calm_water']/1000:.1f} kN ({breakdown['calm_water_percent']:.1f}%)")
        >>> print(f"  Wind:       {breakdown['wind']/1000:.1f} kN ({breakdown['wind_percent']:.1f}%)")
        >>> print(f"  Waves:      {breakdown['added_waves']/1000:.1f} kN ({breakdown['added_waves_percent']:.1f}%)")
    """

    def __init__(
        self,
        calm_water_calculator: Optional[CalmWaterResistance] = None,
        wind_calculator: Optional[WindResistance] = None,
        wave_calculator: Optional[AddedWaveResistance] = None,
    ):
        """Initialize complete resistance model.

        Args:
            calm_water_calculator: Custom calm water resistance calculator (optional)
            wind_calculator: Custom wind resistance calculator (optional)
            wave_calculator: Custom added wave resistance calculator (optional)

        If not provided, default calculators will be instantiated.
        """
        self.calm_water = calm_water_calculator or CalmWaterResistance()
        self.windage = wind_calculator or WindResistance()
        self.added_waves = wave_calculator or AddedWaveResistance()

    def calculate_total_resistance(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate total resistance from all components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions (speed, wind, waves)

        Returns:
            Total resistance in Newtons (N)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15, wind_speed=10, wave_height=3)
            >>> model = ShipResistanceModel()
            >>> R_total = model.calculate_total_resistance(ship, conditions)
        """
        R_calm = self.calm_water.calculate(ship, conditions)
        R_wind = self.windage.calculate(ship, conditions)
        R_waves = self.added_waves.calculate(ship, conditions)

        return R_calm + R_wind + R_waves

    def get_breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, float]:
        """Get detailed resistance breakdown by component.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary with resistance breakdown:
                - calm_water: Calm water resistance (N)
                - wind: Wind resistance (N)
                - added_waves: Added wave resistance (N)
                - total: Total resistance (N)
                - calm_water_percent: % contribution from calm water
                - wind_percent: % contribution from wind
                - added_waves_percent: % contribution from waves

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15, wind_speed=10, wave_height=3)
            >>> model = ShipResistanceModel()
            >>> breakdown = model.get_breakdown(ship, conditions)
        """
        R_calm = self.calm_water.calculate(ship, conditions)
        R_wind = self.windage.calculate(ship, conditions)
        R_waves = self.added_waves.calculate(ship, conditions)
        R_total = R_calm + R_wind + R_waves

        # Calculate percentages
        if R_total > 0:
            pct_calm = 100 * R_calm / R_total
            pct_wind = 100 * R_wind / R_total
            pct_waves = 100 * R_waves / R_total
        else:
            pct_calm = pct_wind = pct_waves = 0.0

        return {
            "calm_water": R_calm,
            "wind": R_wind,
            "added_waves": R_waves,
            "total": R_total,
            "calm_water_percent": pct_calm,
            "wind_percent": pct_wind,
            "added_waves_percent": pct_waves,
        }

    def get_detailed_breakdown(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> dict[str, any]:
        """Get comprehensive breakdown including sub-components.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Dictionary with detailed breakdown:
                - calm_water: Dict with friction, wave-making, total
                - wind: Dict with relative wind, drag coefficient, resistance
                - added_waves: Dict with wave parameters, resistance
                - total: Total resistance (N)
                - percentages: Dict with % contributions

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15, wind_speed=10, wave_height=3, wave_period=10)
            >>> model = ShipResistanceModel()
            >>> detailed = model.get_detailed_breakdown(ship, conditions)
        """
        # Get calm water breakdown
        calm_breakdown = self.calm_water.breakdown(ship, conditions)

        # Get wind breakdown
        wind_breakdown = self.windage.breakdown(ship, conditions)

        # Get wave breakdown
        wave_breakdown = self.added_waves.breakdown(ship, conditions)

        # Total resistance
        R_total = (
            calm_breakdown["total"]
            + wind_breakdown["resistance"]
            + wave_breakdown["resistance"]
        )

        # Calculate percentages
        if R_total > 0:
            pct_calm = 100 * calm_breakdown["total"] / R_total
            pct_wind = 100 * wind_breakdown["resistance"] / R_total
            pct_waves = 100 * wave_breakdown["resistance"] / R_total
        else:
            pct_calm = pct_wind = pct_waves = 0.0

        return {
            "calm_water": calm_breakdown,
            "wind": wind_breakdown,
            "added_waves": wave_breakdown,
            "total": R_total,
            "percentages": {
                "calm_water": pct_calm,
                "wind": pct_wind,
                "added_waves": pct_waves,
            },
        }

    def calculate_effective_power(
        self, ship: ShipParameters, conditions: OperatingConditions
    ) -> float:
        """Calculate effective power (PE = R × V).

        Effective power is the power required to overcome total resistance
        at the given speed. This is the power delivered to the water.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Effective power in Watts (W)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15)
            >>> model = ShipResistanceModel()
            >>> PE = model.calculate_effective_power(ship, conditions)
            >>> print(f"Effective power: {PE/1000:.1f} kW")
        """
        R_total = self.calculate_total_resistance(ship, conditions)
        speed_ms = conditions.speed_ms

        # Power = Force × Velocity
        return R_total * speed_ms

    def resistance_comparison(
        self, ship: ShipParameters, calm: OperatingConditions, environmental: OperatingConditions
    ) -> dict[str, float]:
        """Compare resistance in calm vs environmental conditions.

        Useful for analyzing the impact of weather on ship performance.

        Args:
            ship: Ship parameters
            calm: Calm water conditions (no wind, no waves)
            environmental: Environmental conditions (with wind and/or waves)

        Returns:
            Dictionary with comparison:
                - calm_total: Total resistance in calm (N)
                - environmental_total: Total resistance in environment (N)
                - increase: Absolute increase (N)
                - increase_percent: Percentage increase (%)

        Example:
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> calm = OperatingConditions(speed=15)
            >>> storm = OperatingConditions(speed=15, wind_speed=20, wind_angle=0, wave_height=6, wave_period=12)
            >>> model = ShipResistanceModel()
            >>> comparison = model.resistance_comparison(ship, calm, storm)
            >>> print(f"Storm increases resistance by {comparison['increase_percent']:.1f}%")
        """
        R_calm_total = self.calculate_total_resistance(ship, calm)
        R_env_total = self.calculate_total_resistance(ship, environmental)

        increase = R_env_total - R_calm_total
        increase_pct = 100 * increase / R_calm_total if R_calm_total > 0 else 0.0

        return {
            "calm_total": R_calm_total,
            "environmental_total": R_env_total,
            "increase": increase,
            "increase_percent": increase_pct,
        }
