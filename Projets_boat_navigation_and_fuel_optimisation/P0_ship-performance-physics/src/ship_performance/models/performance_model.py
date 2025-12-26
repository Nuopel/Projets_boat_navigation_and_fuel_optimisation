"""Complete ship performance model integrating all components.

This module provides the top-level ShipPerformanceModel that integrates:
- Resistance model (calm water + wind + waves)
- Propulsion model (power chain)
- Fuel consumption model (SFOC-based)

Complete workflow: Ship + Conditions → Resistance → Power → Fuel Consumption
"""

from dataclasses import dataclass
from typing import Optional

from ..core import OperatingConditions, ShipParameters
from ..propulsion import FuelConsumptionModel, FuelType, PropulsionModel
from .resistance_model import ShipResistanceModel


@dataclass(frozen=True)
class PerformanceResult:
    """Complete performance prediction result.

    This dataclass contains all intermediate and final results from the
    complete performance prediction workflow.

    Attributes:
        # Resistance components
        resistance_calm_water: Calm water resistance (N)
        resistance_wind: Wind resistance (N)
        resistance_waves: Added wave resistance (N)
        resistance_total: Total resistance (N)

        # Power chain
        effective_power: Effective power P_E (kW)
        delivered_power: Delivered power P_D (kW)
        brake_power: Brake power P_B (kW)

        # Efficiencies
        propeller_efficiency: Propeller efficiency η_P
        shaft_efficiency: Shaft efficiency η_S
        overall_efficiency: Overall propulsive efficiency η_D

        # Fuel consumption
        fuel_rate: Fuel consumption rate (kg/h)
        fuel_rate_per_day: Daily fuel consumption (tonnes/day)
        specific_consumption: SFOC (g/kWh)
        fuel_type: Type of fuel used

        # Operating conditions
        speed_knots: Ship speed (knots)
        speed_ms: Ship speed (m/s)

        # Optional
        co2_rate: CO2 emission rate (kg/h), if calculated
        engine_load_factor: Engine load factor (0-1), if available
    """

    # Resistance
    resistance_calm_water: float
    resistance_wind: float
    resistance_waves: float
    resistance_total: float

    # Power
    effective_power: float
    delivered_power: float
    brake_power: float

    # Efficiencies
    propeller_efficiency: float
    shaft_efficiency: float
    overall_efficiency: float

    # Fuel
    fuel_rate: float
    fuel_rate_per_day: float
    specific_consumption: float
    fuel_type: FuelType

    # Operating conditions
    speed_knots: float
    speed_ms: float

    # Optional
    co2_rate: Optional[float] = None
    engine_load_factor: Optional[float] = None

    def summary_string(self) -> str:
        """Generate formatted summary string.

        Returns:
            Multi-line formatted string with key results

        Example:
            >>> result = model.predict(ship, conditions)
            >>> print(result.summary_string())
        """
        lines = [
            "=" * 60,
            "Ship Performance Prediction Results",
            "=" * 60,
            "",
            f"Operating Speed: {self.speed_knots:.1f} knots ({self.speed_ms:.2f} m/s)",
            "",
            "RESISTANCE BREAKDOWN:",
            f"  Calm Water:    {self.resistance_calm_water/1000:>8.1f} kN  ({100*self.resistance_calm_water/self.resistance_total:>5.1f}%)",
            f"  Wind:          {self.resistance_wind/1000:>8.1f} kN  ({100*self.resistance_wind/self.resistance_total:>5.1f}%)",
            f"  Waves:         {self.resistance_waves/1000:>8.1f} kN  ({100*self.resistance_waves/self.resistance_total:>5.1f}%)",
            f"  TOTAL:         {self.resistance_total/1000:>8.1f} kN",
            "",
            "POWER REQUIREMENTS:",
            f"  Effective Power (P_E):   {self.effective_power:>8.1f} kW",
            f"  Delivered Power (P_D):   {self.delivered_power:>8.1f} kW  (η_P = {self.propeller_efficiency:.3f})",
            f"  Brake Power (P_B):       {self.brake_power:>8.1f} kW  (η_S = {self.shaft_efficiency:.3f})",
            f"  Overall Efficiency (η_D): {self.overall_efficiency:.3f}",
            "",
            "FUEL CONSUMPTION:",
            f"  Fuel Type:        {self.fuel_type.value}",
            f"  SFOC:             {self.specific_consumption:.1f} g/kWh",
            f"  Fuel Rate:        {self.fuel_rate:.1f} kg/h",
            f"  Daily Consumption: {self.fuel_rate_per_day:.1f} tonnes/day",
        ]

        if self.co2_rate is not None:
            lines.extend([
                "",
                "EMISSIONS:",
                f"  CO2 Rate:         {self.co2_rate:.1f} kg/h",
                f"  Daily CO2:        {self.co2_rate * 24 / 1000:.1f} tonnes/day",
            ])

        if self.engine_load_factor is not None:
            lines.extend([
                "",
                f"Engine Load Factor: {self.engine_load_factor:.1%}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


class ShipPerformanceModel:
    """Complete ship performance prediction model.

    This class provides end-to-end ship performance prediction from ship
    parameters and operating conditions to fuel consumption and emissions.

    The model integrates:
    1. Resistance Model: Calculates total resistance (calm + wind + waves)
    2. Propulsion Model: Calculates power requirements (P_E → P_B)
    3. Fuel Model: Calculates fuel consumption from brake power

    Example:
        >>> model = ShipPerformanceModel()
        >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
        >>> conditions = OperatingConditions(speed=15, wind_speed=10, wave_height=3)
        >>> result = model.predict(ship, conditions)
        >>> print(f"Fuel consumption: {result.fuel_rate:.1f} kg/h")
        >>> print(f"Daily fuel: {result.fuel_rate_per_day:.1f} tonnes/day")
    """

    def __init__(
        self,
        resistance_model: Optional[ShipResistanceModel] = None,
        propulsion_model: Optional[PropulsionModel] = None,
        fuel_model: Optional[FuelConsumptionModel] = None,
    ):
        """Initialize complete performance model.

        Args:
            resistance_model: Custom resistance model (optional)
            propulsion_model: Custom propulsion model (optional)
            fuel_model: Custom fuel consumption model (optional)

        If not provided, default models will be instantiated with typical parameters.

        Example:
            >>> # Use defaults
            >>> model = ShipPerformanceModel()
            >>>
            >>> # Custom propulsion efficiency
            >>> custom_propulsion = PropulsionModel(propeller_efficiency=0.70)
            >>> model = ShipPerformanceModel(propulsion_model=custom_propulsion)
        """
        self.resistance = resistance_model or ShipResistanceModel()
        self.propulsion = propulsion_model or PropulsionModel(
            propeller_efficiency=0.65,
            shaft_efficiency=0.98,
        )
        self.fuel = fuel_model or FuelConsumptionModel(fuel_type=FuelType.HFO)

    def predict(
        self,
        ship: ShipParameters,
        conditions: OperatingConditions,
        rated_power: Optional[float] = None,
    ) -> PerformanceResult:
        """Predict complete ship performance.

        This is the main method that runs the complete prediction workflow:
        1. Calculate total resistance
        2. Calculate power requirements
        3. Calculate fuel consumption

        Args:
            ship: Ship parameters
            conditions: Operating conditions
            rated_power: Engine rated power in kW (for load factor), optional

        Returns:
            PerformanceResult with all prediction results

        Example:
            >>> model = ShipPerformanceModel()
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15, wind_speed=10, wave_height=2)
            >>> result = model.predict(ship, conditions)
            >>> print(result.summary_string())
        """
        # Step 1: Calculate resistance
        resistance_breakdown = self.resistance.get_breakdown(ship, conditions)

        # Step 2: Calculate power
        power_breakdown = self.propulsion.calculate_power(
            total_resistance=resistance_breakdown["total"],
            speed_ms=conditions.speed_ms,
        )

        # Step 3: Calculate fuel consumption
        fuel_result = self.fuel.calculate_consumption(
            brake_power=power_breakdown.brake_power,
            rated_power=rated_power,
        )

        # Create comprehensive result
        return PerformanceResult(
            # Resistance
            resistance_calm_water=resistance_breakdown["calm_water"],
            resistance_wind=resistance_breakdown["wind"],
            resistance_waves=resistance_breakdown["added_waves"],
            resistance_total=resistance_breakdown["total"],
            # Power
            effective_power=power_breakdown.effective_power,
            delivered_power=power_breakdown.delivered_power,
            brake_power=power_breakdown.brake_power,
            # Efficiencies
            propeller_efficiency=power_breakdown.propeller_efficiency,
            shaft_efficiency=power_breakdown.shaft_efficiency,
            overall_efficiency=power_breakdown.overall_efficiency,
            # Fuel
            fuel_rate=fuel_result.fuel_rate,
            fuel_rate_per_day=fuel_result.fuel_rate_per_day,
            specific_consumption=fuel_result.specific_consumption,
            fuel_type=fuel_result.fuel_type,
            # Operating conditions
            speed_knots=conditions.speed,
            speed_ms=conditions.speed_ms,
            # Optional
            co2_rate=fuel_result.co2_rate,
            engine_load_factor=fuel_result.engine_load_factor,
        )

    def speed_consumption_curve(
        self,
        ship: ShipParameters,
        conditions_template: OperatingConditions,
        speeds: list[float],
        rated_power: Optional[float] = None,
    ) -> list[PerformanceResult]:
        """Generate speed-consumption curve.

        Args:
            ship: Ship parameters
            conditions_template: Template operating conditions (wind, waves, etc.)
            speeds: List of speeds in knots to evaluate
            rated_power: Engine rated power in kW

        Returns:
            List of PerformanceResult for each speed

        Example:
            >>> model = ShipPerformanceModel()
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> template = OperatingConditions(speed=15, wind_speed=10, wave_height=2)
            >>> speeds = [10, 12, 14, 16, 18, 20]
            >>> curve = model.speed_consumption_curve(ship, template, speeds)
            >>> for result in curve:
            ...     print(f"{result.speed_knots} knots: {result.fuel_rate_per_day:.1f} tonnes/day")
        """
        results = []
        for speed in speeds:
            # Create conditions with this speed
            conditions = OperatingConditions(
                speed=speed,
                wind_speed=conditions_template.wind_speed,
                wind_angle=conditions_template.wind_angle,
                wave_height=conditions_template.wave_height,
                wave_period=conditions_template.wave_period,
                wave_angle=conditions_template.wave_angle,
            )

            result = self.predict(ship, conditions, rated_power)
            results.append(result)

        return results

    def voyage_simulation(
        self,
        ship: ShipParameters,
        conditions: OperatingConditions,
        duration_hours: float,
        rated_power: Optional[float] = None,
    ) -> dict:
        """Simulate complete voyage fuel consumption.

        Args:
            ship: Ship parameters
            conditions: Operating conditions
            duration_hours: Voyage duration in hours
            rated_power: Engine rated power in kW

        Returns:
            Dictionary with voyage summary:
                - fuel_consumed: Total fuel consumed (tonnes)
                - average_fuel_rate: Average consumption rate (tonnes/day)
                - co2_emitted: Total CO2 emissions (tonnes), if available
                - average_power: Average brake power (kW)
                - distance: Distance covered (nautical miles)

        Example:
            >>> model = ShipPerformanceModel()
            >>> ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)
            >>> conditions = OperatingConditions(speed=15)
            >>> voyage = model.voyage_simulation(ship, conditions, duration_hours=120)  # 5 days
            >>> print(f"Total fuel: {voyage['fuel_consumed']:.1f} tonnes")
        """
        # Get performance for these conditions
        result = self.predict(ship, conditions, rated_power)

        # Calculate totals
        fuel_consumed_tonnes = result.fuel_rate * duration_hours / 1000
        average_fuel_rate_per_day = result.fuel_rate_per_day

        co2_emitted_tonnes = None
        if result.co2_rate is not None:
            co2_emitted_tonnes = result.co2_rate * duration_hours / 1000

        # Distance covered (speed × time)
        distance_nm = result.speed_knots * duration_hours

        return {
            "fuel_consumed": fuel_consumed_tonnes,
            "average_fuel_rate": average_fuel_rate_per_day,
            "co2_emitted": co2_emitted_tonnes,
            "average_power": result.brake_power,
            "distance": distance_nm,
            "duration_hours": duration_hours,
            "average_speed": result.speed_knots,
            "fuel_type": result.fuel_type.value,
        }
