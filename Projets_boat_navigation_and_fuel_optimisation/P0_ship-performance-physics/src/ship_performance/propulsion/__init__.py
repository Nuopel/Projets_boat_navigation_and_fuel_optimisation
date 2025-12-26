"""Propulsion and fuel consumption models.

This module provides models for:
- Propulsion power chain (effective to delivered to brake power)
- Fuel consumption based on SFOC
- Different fuel types and properties
"""

from .fuel_consumption import (
    FUEL_PROPERTIES,
    FuelConsumptionModel,
    FuelConsumptionResult,
    FuelProperties,
    FuelType,
)
from .propulsion_model import (
    PowerBreakdown,
    PropulsionModel,
    estimate_propeller_efficiency,
)

__all__ = [
    "PropulsionModel",
    "PowerBreakdown",
    "estimate_propeller_efficiency",
    "FuelConsumptionModel",
    "FuelConsumptionResult",
    "FuelType",
    "FuelProperties",
    "FUEL_PROPERTIES",
]
