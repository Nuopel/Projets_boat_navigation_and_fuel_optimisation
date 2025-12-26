"""Physical constants used in ship performance calculations.

This module contains physical constants and typical values used throughout
the ship performance prediction model.

All values are in SI units unless otherwise specified.
"""

# Fluid properties (seawater at 15°C, 35 PSU salinity)
SEAWATER_DENSITY = 1025.0  # kg/m³
SEAWATER_KINEMATIC_VISCOSITY = 1.19e-6  # m²/s

# Air properties (standard atmosphere at sea level, 15°C)
AIR_DENSITY = 1.225  # kg/m³
AIR_KINEMATIC_VISCOSITY = 1.46e-5  # m²/s

# Gravitational acceleration
GRAVITY = 9.81  # m/s²

# Fuel properties - Lower Heating Values (LHV)
HFO_LOWER_HEATING_VALUE = 42700.0  # kJ/kg (Heavy Fuel Oil)
MDO_LOWER_HEATING_VALUE = 42700.0  # kJ/kg (Marine Diesel Oil)
MGO_LOWER_HEATING_VALUE = 43000.0  # kJ/kg (Marine Gas Oil)
LNG_LOWER_HEATING_VALUE = 50000.0  # kJ/kg (Liquefied Natural Gas)

# Typical efficiency values (for reference and defaults)
TYPICAL_PROPELLER_EFFICIENCY = 0.65  # dimensionless
TYPICAL_SHAFT_EFFICIENCY = 0.98  # dimensionless
TYPICAL_ENGINE_SFOC = 185.0  # g/kWh (Specific Fuel Oil Consumption)
TYPICAL_ENGINE_EFFICIENCY = 0.45  # dimensionless (thermal efficiency)

# Unit conversion factors
KNOTS_TO_MS = 0.514444  # m/s per knot
MS_TO_KNOTS = 1.0 / KNOTS_TO_MS  # knots per m/s
DEGREES_TO_RADIANS = 0.017453292519943295  # π/180
RADIANS_TO_DEGREES = 57.29577951308232  # 180/π

# Typical ship parameter ranges (for validation warnings)
TYPICAL_BLOCK_COEFFICIENT_MIN = 0.4
TYPICAL_BLOCK_COEFFICIENT_MAX = 0.9
TYPICAL_LENGTH_BEAM_RATIO_MIN = 5.0
TYPICAL_LENGTH_BEAM_RATIO_MAX = 10.0
TYPICAL_BEAM_DRAFT_RATIO_MIN = 2.0
TYPICAL_BEAM_DRAFT_RATIO_MAX = 4.0
