"""Interfaces and protocols for ship performance components.

This module exports all protocol definitions used for polymorphic composition
of ship performance calculation components.
"""

from .propulsion import FuelConsumptionModel, PropulsionModel
from .resistance import ResistanceBreakdown, ResistanceCalculator

__all__ = [
    "ResistanceCalculator",
    "ResistanceBreakdown",
    "PropulsionModel",
    "FuelConsumptionModel",
]
