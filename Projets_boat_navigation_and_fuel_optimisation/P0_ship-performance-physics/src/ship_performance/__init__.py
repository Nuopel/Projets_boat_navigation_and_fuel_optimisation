"""Ship Performance Prediction Model.

A comprehensive Python package for predicting ship fuel consumption
under varying operational conditions.

This package implements physics-based models for:
- Calm water resistance (Holtrop-Mennen method)
- Wind resistance (windage)
- Added resistance in waves
- Propulsion system efficiency
- Fuel consumption estimation

Example:
    >>> from ship_performance.core import ShipParameters, OperatingConditions
    >>> from ship_performance.models import ShipPerformanceModel
    >>>
    >>> ship = ShipParameters(
    ...     length=150, beam=25, draft=8,
    ...     displacement=15000, block_coefficient=0.7
    ... )
    >>> conditions = OperatingConditions(speed=15, wave_height=2)
    >>> model = ShipPerformanceModel()
    >>> result = model.predict_fuel_consumption(ship, conditions)
    >>> print(f"Fuel rate: {result.fuel_rate:.1f} kg/h")
"""

__version__ = "0.1.0"
__author__ = "Ship Performance Team"

# Version info
VERSION = tuple(map(int, __version__.split(".")))

# Public API will be expanded as MVPs are completed
__all__ = [
    "__version__",
    "VERSION",
]
