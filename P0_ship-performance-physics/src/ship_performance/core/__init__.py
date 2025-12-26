"""Core data models for ship performance calculations.

This module exports the fundamental data structures used throughout
the ship performance prediction package.
"""

from .operating_conditions import InvalidOperatingConditionsError, OperatingConditions
from .ship_parameters import InvalidShipParametersError, ShipParameters

__all__ = [
    "ShipParameters",
    "InvalidShipParametersError",
    "OperatingConditions",
    "InvalidOperatingConditionsError",
]
