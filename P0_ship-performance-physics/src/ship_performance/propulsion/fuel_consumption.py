"""Fuel consumption model based on engine power and SFOC.

This module implements fuel consumption calculation based on:
- Brake power from propulsion model
- SFOC (Specific Fuel Oil Consumption) in g/kWh
- Engine load factor effects
- Different fuel types (HFO, MDO, LNG)

SFOC varies with engine load:
- Optimum load: 75-85% → minimum SFOC (170-185 g/kWh for modern engines)
- Low load: <50% → higher SFOC (+10-20%)
- High load: >90% → slightly higher SFOC (+5-10%)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FuelType(Enum):
    """Fuel type enumeration."""

    HFO = "Heavy Fuel Oil"  # IFO 380
    MDO = "Marine Diesel Oil"
    MGO = "Marine Gas Oil"
    LNG = "Liquefied Natural Gas"


@dataclass(frozen=True)
class FuelProperties:
    """Fuel properties for different fuel types.

    Attributes:
        fuel_type: Type of fuel
        lower_heating_value: LHV in kJ/kg
        density: Density in kg/m³ at 15°C
        carbon_content: Carbon content fraction (for emissions)
        sulfur_content: Sulfur content fraction
        typical_sfoc: Typical SFOC in g/kWh
    """

    fuel_type: FuelType
    lower_heating_value: float  # kJ/kg
    density: float  # kg/m³
    carbon_content: float  # fraction
    sulfur_content: float  # fraction
    typical_sfoc: float  # g/kWh


# Standard fuel properties
FUEL_PROPERTIES = {
    FuelType.HFO: FuelProperties(
        fuel_type=FuelType.HFO,
        lower_heating_value=40200,  # kJ/kg
        density=991,  # kg/m³
        carbon_content=0.85,
        sulfur_content=0.005,  # 0.5% (IMO 2020 compliant)
        typical_sfoc=185,  # g/kWh
    ),
    FuelType.MDO: FuelProperties(
        fuel_type=FuelType.MDO,
        lower_heating_value=42700,  # kJ/kg
        density=890,  # kg/m³
        carbon_content=0.87,
        sulfur_content=0.001,  # 0.1%
        typical_sfoc=180,  # g/kWh (slightly better)
    ),
    FuelType.MGO: FuelProperties(
        fuel_type=FuelType.MGO,
        lower_heating_value=43000,  # kJ/kg
        density=850,  # kg/m³
        carbon_content=0.87,
        sulfur_content=0.001,
        typical_sfoc=178,  # g/kWh
    ),
    FuelType.LNG: FuelProperties(
        fuel_type=FuelType.LNG,
        lower_heating_value=50000,  # kJ/kg (methane)
        density=450,  # kg/m³ (liquid)
        carbon_content=0.75,  # Lower carbon
        sulfur_content=0.0,  # Negligible
        typical_sfoc=155,  # g/kWh (better efficiency, dual-fuel)
    ),
}


@dataclass(frozen=True)
class FuelConsumptionResult:
    """Fuel consumption calculation result.

    Attributes:
        fuel_rate: Fuel consumption rate (kg/h)
        fuel_rate_per_day: Daily fuel consumption (tonnes/day)
        specific_consumption: SFOC used (g/kWh)
        brake_power: Engine brake power (kW)
        engine_load_factor: Engine load factor (0-1)
        fuel_type: Type of fuel
        co2_rate: CO2 emission rate (kg/h), if calculated
    """

    fuel_rate: float
    fuel_rate_per_day: float
    specific_consumption: float
    brake_power: float
    engine_load_factor: float
    fuel_type: FuelType
    co2_rate: Optional[float] = None


class FuelConsumptionModel:
    """Fuel consumption model based on brake power and SFOC.

    This model calculates fuel consumption from brake power using:
    - Base SFOC (Specific Fuel Oil Consumption)
    - Load factor corrections (optimal at 75-85% load)
    - Fuel type properties

    Example:
        >>> fuel_model = FuelConsumptionModel(fuel_type=FuelType.HFO)
        >>> result = fuel_model.calculate_consumption(
        ...     brake_power=5000,
        ...     rated_power=8000
        ... )
        >>> print(f"Fuel rate: {result.fuel_rate:.1f} kg/h")
        >>> print(f"Daily consumption: {result.fuel_rate_per_day:.1f} tonnes/day")
    """

    def __init__(
        self,
        fuel_type: FuelType = FuelType.HFO,
        base_sfoc: Optional[float] = None,
        rated_power: Optional[float] = None,
    ):
        """Initialize fuel consumption model.

        Args:
            fuel_type: Type of fuel to use
            base_sfoc: Base SFOC in g/kWh (uses typical if not provided)
            rated_power: Engine rated power in kW (for load factor calculation)

        Example:
            >>> fuel_model = FuelConsumptionModel(
            ...     fuel_type=FuelType.MDO,
            ...     base_sfoc=180,
            ...     rated_power=10000
            ... )
        """
        self._fuel_type = fuel_type
        self._fuel_properties = FUEL_PROPERTIES[fuel_type]

        # Use provided SFOC or typical value for fuel type
        self._base_sfoc = base_sfoc if base_sfoc is not None else self._fuel_properties.typical_sfoc

        self._rated_power = rated_power

    @property
    def fuel_type(self) -> FuelType:
        """Get current fuel type."""
        return self._fuel_type

    @property
    def base_sfoc(self) -> float:
        """Get base SFOC (g/kWh)."""
        return self._base_sfoc

    def calculate_consumption(
        self,
        brake_power: float,
        rated_power: Optional[float] = None,
    ) -> FuelConsumptionResult:
        """Calculate fuel consumption for given brake power.

        Args:
            brake_power: Engine brake power (kW)
            rated_power: Engine rated power (kW), uses instance value if not provided

        Returns:
            FuelConsumptionResult with consumption and related data

        Example:
            >>> fuel_model = FuelConsumptionModel(rated_power=10000)
            >>> result = fuel_model.calculate_consumption(brake_power=7500)
            >>> print(f"Fuel rate: {result.fuel_rate:.1f} kg/h")
        """
        # Determine rated power (for load factor calculation)
        P_rated = rated_power if rated_power is not None else self._rated_power

        # Calculate engine load factor
        if P_rated is not None and P_rated > 0:
            load_factor = brake_power / P_rated
        else:
            # Assume typical load if rated power not provided
            load_factor = 0.80  # Assume 80% load

        # Apply load factor correction to SFOC
        sfoc_corrected = self._apply_load_factor_correction(self._base_sfoc, load_factor)

        # Calculate fuel consumption rate
        # FC = P_B × SFOC / 1000  (convert g/h to kg/h)
        fuel_rate_kg_h = brake_power * sfoc_corrected / 1000

        # Daily consumption (tonnes/day)
        fuel_rate_per_day = fuel_rate_kg_h * 24 / 1000

        # Calculate CO2 emissions if carbon content available
        co2_rate = None
        if self._fuel_properties.carbon_content > 0:
            # CO2 = fuel_rate × carbon_content × (44/12)
            # 44/12 is molecular weight ratio CO2/C
            co2_rate = fuel_rate_kg_h * self._fuel_properties.carbon_content * (44 / 12)

        return FuelConsumptionResult(
            fuel_rate=fuel_rate_kg_h,
            fuel_rate_per_day=fuel_rate_per_day,
            specific_consumption=sfoc_corrected,
            brake_power=brake_power,
            engine_load_factor=load_factor,
            fuel_type=self._fuel_type,
            co2_rate=co2_rate,
        )

    def _apply_load_factor_correction(self, base_sfoc: float, load_factor: float) -> float:
        """Apply load factor correction to SFOC.

        Modern diesel engines have optimal SFOC at 75-85% load.
        Below and above this range, SFOC increases.

        Args:
            base_sfoc: Base SFOC at optimal load (g/kWh)
            load_factor: Engine load factor (0-1)

        Returns:
            Corrected SFOC (g/kWh)
        """
        # Ensure load factor is reasonable
        load_factor = max(0.1, min(1.1, load_factor))

        # Optimal load range
        if 0.75 <= load_factor <= 0.85:
            # Optimal - use base SFOC
            correction_factor = 1.0

        elif load_factor < 0.75:
            # Low load - efficiency drops
            # Linear penalty below 75% load
            # At 50% load: +5% SFOC
            # At 25% load: +15% SFOC
            correction_factor = 1.0 + (0.75 - load_factor) * 0.4

        else:  # load_factor > 0.85
            # High load - slight efficiency drop
            # At 100% load: +8% SFOC
            # At 110% load (overload): +15% SFOC
            correction_factor = 1.0 + (load_factor - 0.85) * 0.35

        return base_sfoc * correction_factor

    def calculate_voyage_consumption(
        self,
        brake_power: float,
        duration_hours: float,
        rated_power: Optional[float] = None,
    ) -> dict:
        """Calculate total fuel consumption for a voyage.

        Args:
            brake_power: Average brake power during voyage (kW)
            duration_hours: Voyage duration (hours)
            rated_power: Engine rated power (kW)

        Returns:
            Dictionary with:
                - fuel_rate: Consumption rate (kg/h)
                - total_fuel: Total fuel consumed (tonnes)
                - duration_hours: Voyage duration (hours)
                - average_power: Average brake power (kW)
                - co2_total: Total CO2 emissions (tonnes), if available

        Example:
            >>> fuel_model = FuelConsumptionModel()
            >>> voyage = fuel_model.calculate_voyage_consumption(
            ...     brake_power=6000,
            ...     duration_hours=120  # 5 days
            ... )
            >>> print(f"Total fuel: {voyage['total_fuel']:.1f} tonnes")
        """
        result = self.calculate_consumption(brake_power, rated_power)

        total_fuel_tonnes = result.fuel_rate * duration_hours / 1000
        co2_total_tonnes = None
        if result.co2_rate is not None:
            co2_total_tonnes = result.co2_rate * duration_hours / 1000

        return {
            "fuel_rate": result.fuel_rate,
            "total_fuel": total_fuel_tonnes,
            "duration_hours": duration_hours,
            "average_power": brake_power,
            "sfoc": result.specific_consumption,
            "co2_total": co2_total_tonnes,
            "fuel_type": self._fuel_type.value,
        }
