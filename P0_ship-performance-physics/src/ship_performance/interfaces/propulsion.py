"""Interfaces for propulsion system components.

This module defines protocols for propulsion models, enabling flexible
propulsion system implementations.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PropulsionModel(Protocol):
    """Protocol for propulsion models.

    Propulsion models calculate power requirements from resistance and speed,
    accounting for propeller efficiency, shaft losses, and other transmission
    factors.
    """

    def calculate_power(self, resistance: float, speed_ms: float) -> dict[str, float]:
        """Calculate power requirements from resistance.

        Args:
            resistance: Total resistance force in Newtons (N)
            speed_ms: Ship speed in meters per second (m/s)

        Returns:
            Dictionary containing power values in kilowatts (kW):
                - 'effective_power': P_e = R × V (power to overcome resistance)
                - 'delivered_power': P_d = P_e / η_propeller (power at propeller)
                - 'brake_power': P_b = P_d / η_shaft (power at engine)

        Raises:
            ValueError: If resistance or speed is negative

        Example:
            >>> model = SimplePropulsionModel()
            >>> power = model.calculate_power(resistance=200000, speed_ms=7.72)
            >>> print(f"Brake power: {power['brake_power']:.1f} kW")
        """
        ...


@runtime_checkable
class FuelConsumptionModel(Protocol):
    """Protocol for fuel consumption models.

    Fuel consumption models calculate fuel rate from power requirements,
    accounting for engine efficiency and fuel properties.
    """

    def calculate_consumption(self, brake_power: float) -> float:
        """Calculate fuel consumption rate from brake power.

        Args:
            brake_power: Brake power at engine in kilowatts (kW)

        Returns:
            Fuel consumption rate in kilograms per hour (kg/h)

        Raises:
            ValueError: If brake power is negative

        Example:
            >>> model = SimpleFuelModel()
            >>> fuel_rate = model.calculate_consumption(brake_power=2500)
            >>> print(f"Fuel rate: {fuel_rate:.1f} kg/h")
        """
        ...
