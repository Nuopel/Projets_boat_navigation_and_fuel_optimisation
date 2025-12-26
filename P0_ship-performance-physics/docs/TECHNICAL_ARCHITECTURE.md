# Technical Architecture & Python Best Practices Guide

**Project:** Ship Performance Prediction Model (Navig_P0)
**Target:** Production-quality Python package
**Python Version:** 3.9+

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Principles](#design-principles)
3. [Package Structure](#package-structure)
4. [Code Standards](#code-standards)
5. [Testing Strategy](#testing-strategy)
6. [Documentation Standards](#documentation-standards)
7. [Development Workflow](#development-workflow)
8. [Dependencies & Tools](#dependencies--tools)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface Layer                   │
│  (Notebooks, Scripts, CLI - for demos/analysis)         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Application/Model Layer                     │
│  ┌─────────────────────────────────────────┐           │
│  │   ShipPerformanceModel                  │           │
│  │  (Main orchestrator)                    │           │
│  └────────┬────────────────────┬───────────┘           │
│           │                    │                        │
│  ┌────────▼─────────┐  ┌──────▼──────────────┐        │
│  │ ResistanceModel  │  │  PropulsionModel    │        │
│  │  (Composition)   │  │   (Composition)     │        │
│  └────────┬─────────┘  └──────┬──────────────┘        │
└───────────┼────────────────────┼─────────────────────────┘
            │                    │
┌───────────▼────────────────────▼─────────────────────────┐
│               Component Layer                            │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────────┐   │
│  │ CalmWater    │  │ Windage  │  │ AddedWaves      │   │
│  │ Resistance   │  │          │  │ Resistance      │   │
│  └──────────────┘  └──────────┘  └─────────────────┘   │
│  ┌──────────────┐  ┌──────────┐  ┌─────────────────┐   │
│  │ Propeller    │  │ Engine   │  │ FuelModel       │   │
│  └──────────────┘  └──────────┘  └─────────────────┘   │
└───────────┬──────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────┐
│                  Core/Domain Layer                       │
│  ┌────────────────┐  ┌──────────────────────┐           │
│  │ ShipParameters │  │ OperatingConditions  │           │
│  └────────────────┘  └──────────────────────┘           │
│  ┌────────────────────────────────────────────┐         │
│  │          Interfaces & Protocols            │         │
│  │  (ResistanceCalculator, PropulsionModel)   │         │
│  └────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────┐
│              Utilities & Constants                       │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐         │
│  │  Units   │  │ Constants │  │ Visualization│         │
│  │Conversion│  │ (Physical)│  │   Helpers    │         │
│  └──────────┘  └───────────┘  └──────────────┘         │
└──────────────────────────────────────────────────────────┘
```

### Architectural Patterns

1. **Layered Architecture:** Clear separation of concerns (Core → Components → Models → Interface)
2. **Composition over Inheritance:** Models compose calculators rather than inherit
3. **Dependency Injection:** Components receive dependencies, don't create them
4. **Strategy Pattern:** Different resistance/propulsion strategies swappable
5. **Facade Pattern:** `ShipPerformanceModel` provides simplified interface

---

## Design Principles

### SOLID Principles

#### 1. Single Responsibility Principle (SRP)
Each class has one reason to change.

```python
# ✅ GOOD: Each class has single responsibility
class FrictionResistance:
    """Calculates only friction resistance."""
    def calculate(self, ship, conditions): ...

class WaveMakingResistance:
    """Calculates only wave-making resistance."""
    def calculate(self, ship, conditions): ...

class CalmWaterResistance:
    """Composes friction + wave-making."""
    def __init__(self):
        self.friction = FrictionResistance()
        self.wave_making = WaveMakingResistance()

# ❌ BAD: One class does too much
class ResistanceCalculator:
    def calculate_friction(self): ...
    def calculate_waves(self): ...
    def calculate_wind(self): ...
    def calculate_propulsion(self): ...  # Wrong responsibility!
```

#### 2. Open/Closed Principle (OCP)
Open for extension, closed for modification.

```python
# ✅ GOOD: New resistance types added without modifying existing code
from typing import Protocol

class ResistanceCalculator(Protocol):
    """Interface for resistance calculators."""
    def calculate(self, ship, conditions) -> float: ...

class TotalResistance:
    """Composes multiple resistance calculators."""
    def __init__(self, calculators: list[ResistanceCalculator]):
        self.calculators = calculators

    def calculate(self, ship, conditions) -> float:
        return sum(calc.calculate(ship, conditions) for calc in self.calculators)

# Add new resistance type without changing TotalResistance
class AirResistance:
    def calculate(self, ship, conditions) -> float: ...

resistance_model = TotalResistance([
    CalmWaterResistance(),
    WindResistance(),
    AddedWaveResistance(),
    AirResistance()  # New type, no code changes needed
])
```

#### 3. Liskov Substitution Principle (LSP)
Subtypes must be substitutable for their base types.

```python
# ✅ GOOD: All resistance calculators follow same contract
def analyze_resistance(
    calculator: ResistanceCalculator,
    ship: ShipParameters,
    conditions: OperatingConditions
) -> float:
    """Works with ANY ResistanceCalculator implementation."""
    return calculator.calculate(ship, conditions)

# All of these work identically
analyze_resistance(CalmWaterResistance(), ship, conditions)
analyze_resistance(WindResistance(), ship, conditions)
analyze_resistance(AddedWaveResistance(), ship, conditions)
```

#### 4. Interface Segregation Principle (ISP)
Clients shouldn't depend on interfaces they don't use.

```python
# ✅ GOOD: Small, focused protocols
class ResistanceCalculator(Protocol):
    def calculate(self, ship, conditions) -> float: ...

class ResistanceBreakdown(Protocol):
    def breakdown(self, ship, conditions) -> dict: ...

# Classes implement only what they need
class SimpleWindResistance:
    """Simple wind resistance without breakdown."""
    def calculate(self, ship, conditions) -> float: ...

class DetailedCalmWater:
    """Calm water with detailed breakdown."""
    def calculate(self, ship, conditions) -> float: ...
    def breakdown(self, ship, conditions) -> dict: ...

# ❌ BAD: Fat interface forces unnecessary implementation
class ResistanceCalculatorFat(Protocol):
    def calculate(self, ship, conditions) -> float: ...
    def breakdown(self, ship, conditions) -> dict: ...  # Not all need this
    def optimize(self, ship) -> ShipParameters: ...      # Unrelated!
```

#### 5. Dependency Inversion Principle (DIP)
Depend on abstractions, not concretions.

```python
# ✅ GOOD: Depend on Protocol, not concrete class
class ShipPerformanceModel:
    def __init__(
        self,
        resistance_model: ResistanceCalculator,  # Protocol, not concrete
        propulsion_model: PropulsionCalculator    # Protocol, not concrete
    ):
        self.resistance = resistance_model
        self.propulsion = propulsion_model

# Can inject different implementations
model1 = ShipPerformanceModel(
    SimpleResistance(),
    SimplePropulsion()
)

model2 = ShipPerformanceModel(
    DetailedHoltropResistance(),
    WageningenPropulsion()
)

# ❌ BAD: Depend on concrete classes
class ShipPerformanceModelBad:
    def __init__(self):
        self.resistance = CalmWaterResistance()  # Hardcoded!
        self.propulsion = SimplePropulsion()     # Can't swap!
```

---

### Additional Design Principles

#### Don't Repeat Yourself (DRY)
```python
# ✅ GOOD: Reusable utility
def calculate_reynolds_number(length: float, speed: float, kinematic_viscosity: float) -> float:
    """Calculate Reynolds number (reused across modules)."""
    return (speed * length) / kinematic_viscosity

# ❌ BAD: Same calculation duplicated in multiple classes
class FrictionResistance:
    def calculate(self, ship, conditions):
        reynolds = (conditions.speed * ship.length) / 1.19e-6  # Duplicated!

class SomeOtherClass:
    def analyze(self, ship, conditions):
        reynolds = (conditions.speed * ship.length) / 1.19e-6  # Duplicated!
```

#### Separation of Concerns (SOC)
```python
# ✅ GOOD: Separate calculation from presentation
class CalmWaterResistance:
    """Pure calculation, no printing/plotting."""
    def calculate(self, ship, conditions) -> float: ...

class ResistanceVisualizer:
    """Separate visualization logic."""
    def plot_resistance_curve(self, resistance_calc, ship, speed_range): ...

# ❌ BAD: Mixed concerns
class CalmWaterResistanceBad:
    def calculate(self, ship, conditions) -> float:
        result = ...  # calculation
        print(f"Resistance: {result}")  # Presentation mixed in!
        return result
```

#### Composition over Inheritance
```python
# ✅ GOOD: Compose behaviors
class TotalResistance:
    def __init__(self):
        self.calm = CalmWaterResistance()
        self.wind = WindResistance()
        self.waves = AddedWaveResistance()

    def calculate(self, ship, conditions):
        return (
            self.calm.calculate(ship, conditions) +
            self.wind.calculate(ship, conditions) +
            self.waves.calculate(ship, conditions)
        )

# ❌ BAD: Deep inheritance hierarchies
class Resistance: ...
class WaterResistance(Resistance): ...
class CalmWaterResistance(WaterResistance): ...
class HoltropMennenResistance(CalmWaterResistance): ...  # Too deep!
```

---

## Package Structure

```
ship_performance/
│
├── pyproject.toml           # Project metadata, dependencies, tool configs
├── setup.py                 # Optional: for backwards compatibility
├── README.md                # User-facing documentation
├── LICENSE                  # MIT or Apache 2.0
├── .gitignore
├── .pre-commit-config.yaml  # Pre-commit hooks
│
├── src/
│   └── ship_performance/    # Main package
│       ├── __init__.py      # Package initialization, version
│       ├── py.typed         # PEP 561 type marker
│       │
│       ├── core/            # Core domain models
│       │   ├── __init__.py
│       │   ├── ship_parameters.py
│       │   ├── operating_conditions.py
│       │   └── results.py   # Result dataclasses
│       │
│       ├── interfaces/      # Protocols and ABCs
│       │   ├── __init__.py
│       │   ├── resistance.py
│       │   └── propulsion.py
│       │
│       ├── resistance/      # Resistance components
│       │   ├── __init__.py
│       │   ├── friction.py
│       │   ├── wave_making.py
│       │   ├── calm_water.py
│       │   ├── windage.py
│       │   ├── added_waves.py
│       │   └── total.py
│       │
│       ├── propulsion/      # Propulsion components
│       │   ├── __init__.py
│       │   ├── propeller.py
│       │   ├── engine.py
│       │   └── fuel.py
│       │
│       ├── models/          # Integrated models
│       │   ├── __init__.py
│       │   ├── resistance_model.py
│       │   ├── propulsion_model.py
│       │   └── performance_model.py
│       │
│       ├── utils/           # Utilities
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── units.py
│       │   ├── validation.py
│       │   └── visualization.py
│       │
│       └── validation/      # Validation tools
│           ├── __init__.py
│           ├── benchmarks.py
│           └── reference_data.py
│
├── tests/                   # Test suite (mirrors src structure)
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures
│   ├── core/
│   │   ├── test_ship_parameters.py
│   │   └── test_operating_conditions.py
│   ├── resistance/
│   │   ├── test_friction.py
│   │   ├── test_wave_making.py
│   │   └── test_calm_water.py
│   ├── propulsion/
│   │   └── test_propeller.py
│   ├── models/
│   │   └── test_performance_model.py
│   └── property_tests/      # Property-based tests
│       └── test_invariants.py
│
├── docs/                    # Documentation
│   ├── index.md
│   ├── user_guide.md
│   ├── theory_reference.md
│   ├── api_reference/       # Auto-generated
│   └── mvp_reports/         # MVP completion reports
│
├── examples/                # Example scripts
│   ├── basic_usage.py
│   ├── sensitivity_analysis.py
│   └── advanced_customization.py
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_foundation_demo.ipynb
│   ├── 02_calm_water_validation.ipynb
│   ├── 03_wind_effects.ipynb
│   ├── 04_complete_resistance_model.ipynb
│   └── 05_final_validation.ipynb
│
├── data/                    # Data files
│   ├── validation/
│   │   └── benchmark_results.csv
│   └── reference/
│       └── ship_database.json
│
└── scripts/                 # Development scripts
    ├── run_tests.sh
    ├── generate_docs.sh
    └── benchmark.py
```

---

## Code Standards

### Type Hints

**All public functions must have complete type hints.**

```python
from typing import Optional, Protocol
from dataclasses import dataclass

# ✅ GOOD: Full type hints
@dataclass
class ShipParameters:
    length: float
    beam: float
    draft: float
    displacement: float
    block_coefficient: float
    wetted_surface: Optional[float] = None

def calculate_resistance(
    ship: ShipParameters,
    speed: float,
    include_windage: bool = False
) -> float:
    """Calculate ship resistance.

    Args:
        ship: Ship parameters
        speed: Speed in m/s
        include_windage: Whether to include wind resistance

    Returns:
        Total resistance in Newtons
    """
    ...

# Use Protocol for structural typing
class ResistanceCalculator(Protocol):
    def calculate(self, ship: ShipParameters, conditions: OperatingConditions) -> float: ...

# ❌ BAD: No type hints
def calculate_resistance(ship, speed, include_windage=False):
    ...
```

### Docstrings

**Use Google style or NumPy style consistently.**

```python
# ✅ GOOD: Google style docstring
def ittc_friction_coefficient(reynolds_number: float) -> float:
    """Calculate ITTC 1957 friction coefficient.

    Uses the ITTC 1957 friction line formula:
    Cf = 0.075 / (log10(Re) - 2)²

    Args:
        reynolds_number: Reynolds number (dimensionless), must be > 0

    Returns:
        Friction coefficient (dimensionless)

    Raises:
        ValueError: If reynolds_number <= 0

    Example:
        >>> ittc_friction_coefficient(1e7)
        0.00246

    References:
        ITTC (1957). "Report of Resistance Committee."
    """
    if reynolds_number <= 0:
        raise ValueError("Reynolds number must be positive")

    import numpy as np
    return 0.075 / (np.log10(reynolds_number) - 2) ** 2
```

### Error Handling

```python
# ✅ GOOD: Specific exceptions with helpful messages
class InvalidShipParametersError(ValueError):
    """Raised when ship parameters are invalid."""
    pass

@dataclass
class ShipParameters:
    length: float
    beam: float

    def __post_init__(self):
        if self.length <= 0:
            raise InvalidShipParametersError(
                f"Ship length must be positive, got {self.length}"
            )
        if self.beam <= 0:
            raise InvalidShipParametersError(
                f"Ship beam must be positive, got {self.beam}"
            )
        if self.length / self.beam < 3:
            raise InvalidShipParametersError(
                f"L/B ratio too small ({self.length/self.beam:.2f}), "
                f"expected ≥ 3 for typical ships"
            )

# ❌ BAD: Generic exceptions, no message
def validate(self):
    if self.length <= 0:
        raise Exception  # What went wrong? How to fix?
```

### Immutability Where Possible

```python
from dataclasses import dataclass

# ✅ GOOD: Frozen dataclass (immutable)
@dataclass(frozen=True)
class ShipParameters:
    length: float
    beam: float
    draft: float

    @property
    def length_beam_ratio(self) -> float:
        """Calculate L/B ratio (derived property)."""
        return self.length / self.beam

# Ship parameters can't be accidentally modified
ship = ShipParameters(150, 25, 8)
# ship.length = 200  # Raises FrozenInstanceError ✅

# ❌ BAD: Mutable data that could be changed unexpectedly
class ShipParametersBad:
    def __init__(self, length, beam):
        self.length = length  # Can be changed anytime
        self.beam = beam
```

### Use Context Managers

```python
# ✅ GOOD: Context manager for resource management
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f}s")

# Usage
with timer("Resistance calculation"):
    resistance = model.calculate(ship, conditions)
```

### Logging, Not Printing

```python
import logging

logger = logging.getLogger(__name__)

# ✅ GOOD: Use logging
def calculate_resistance(ship, conditions):
    logger.debug(f"Calculating resistance for {ship.length}m ship")
    try:
        result = ...
        logger.info(f"Resistance calculated: {result:.2f} N")
        return result
    except Exception as e:
        logger.error(f"Failed to calculate resistance: {e}")
        raise

# ❌ BAD: Use print statements
def calculate_resistance_bad(ship, conditions):
    print(f"Calculating resistance...")  # Can't be disabled/redirected
    result = ...
    print(f"Result: {result}")
    return result
```

---

## Testing Strategy

### Test Structure

```python
# tests/resistance/test_friction.py
import pytest
from ship_performance.resistance.friction import FrictionResistance, ittc_friction_coefficient
from ship_performance.core import ShipParameters, OperatingConditions

class TestITTCFrictionCoefficient:
    """Tests for ITTC friction coefficient calculation."""

    def test_typical_reynolds_number(self):
        """Test with typical ship Reynolds number."""
        cf = ittc_friction_coefficient(1e7)
        assert pytest.approx(cf, rel=1e-4) == 0.00246

    def test_reynolds_number_range(self):
        """Test across typical range."""
        for re in [1e6, 1e7, 1e8, 1e9]:
            cf = ittc_friction_coefficient(re)
            assert 0.001 < cf < 0.01  # Reasonable range

    def test_invalid_reynolds_number(self):
        """Test that negative Reynolds number raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ittc_friction_coefficient(-1000)

    def test_zero_reynolds_number(self):
        """Test that zero Reynolds number raises error."""
        with pytest.raises(ValueError):
            ittc_friction_coefficient(0)


class TestFrictionResistance:
    """Tests for friction resistance calculator."""

    @pytest.fixture
    def ship(self):
        """Fixture providing standard ship parameters."""
        return ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
            wetted_surface=4000
        )

    @pytest.fixture
    def conditions(self):
        """Fixture providing standard operating conditions."""
        return OperatingConditions(speed=15)  # knots

    def test_calculate_friction_resistance(self, ship, conditions):
        """Test friction resistance calculation."""
        calc = FrictionResistance()
        R_f = calc.calculate(ship, conditions)

        assert R_f > 0
        assert isinstance(R_f, float)

    def test_resistance_increases_with_speed(self, ship):
        """Test that resistance increases with speed."""
        calc = FrictionResistance()

        speeds = [10, 15, 20]  # knots
        resistances = [
            calc.calculate(ship, OperatingConditions(speed=s))
            for s in speeds
        ]

        assert resistances[0] < resistances[1] < resistances[2]

    def test_comparison_with_reference(self, ship, conditions):
        """Test against known reference value."""
        calc = FrictionResistance()
        R_f = calc.calculate(ship, conditions)

        # Compare with PyResis or published data
        expected = 125000  # N (from reference)
        assert pytest.approx(R_f, rel=0.05) == expected
```

### Test Types

1. **Unit Tests:** Test individual functions/methods
2. **Integration Tests:** Test component interactions
3. **Property Tests:** Test invariants with random data
4. **Regression Tests:** Ensure consistency with previous versions
5. **Validation Tests:** Compare with external benchmarks

### Property-Based Testing

```python
# tests/property_tests/test_invariants.py
from hypothesis import given, strategies as st
from ship_performance.resistance import CalmWaterResistance
from ship_performance.core import ShipParameters, OperatingConditions

@given(
    length=st.floats(min_value=50, max_value=400),
    speed=st.floats(min_value=5, max_value=30)
)
def test_resistance_positive(length, speed):
    """Property: Resistance is always positive."""
    ship = ShipParameters(
        length=length,
        beam=length/7,  # Typical ratio
        draft=length/18,
        displacement=length**2.5 / 100,
        block_coefficient=0.7
    )
    conditions = OperatingConditions(speed=speed)

    calc = CalmWaterResistance()
    resistance = calc.calculate(ship, conditions)

    assert resistance > 0

@given(speed=st.floats(min_value=5, max_value=30))
def test_resistance_monotonically_increasing(speed):
    """Property: Resistance increases with speed for fixed ship."""
    ship = ShipParameters(length=150, beam=25, draft=8, displacement=15000, block_coefficient=0.7)

    calc = CalmWaterResistance()
    R1 = calc.calculate(ship, OperatingConditions(speed=speed))
    R2 = calc.calculate(ship, OperatingConditions(speed=speed * 1.1))

    assert R2 > R1
```

### Pytest Fixtures

```python
# tests/conftest.py
import pytest
from ship_performance.core import ShipParameters, OperatingConditions

@pytest.fixture
def cargo_ship():
    """Standard cargo ship for testing."""
    return ShipParameters(
        length=150,
        beam=25,
        draft=8,
        displacement=15000,
        block_coefficient=0.7,
        wetted_surface=4000,
        frontal_area=600
    )

@pytest.fixture
def tanker_ship():
    """Typical tanker for testing."""
    return ShipParameters(
        length=250,
        beam=42,
        draft=14,
        displacement=80000,
        block_coefficient=0.82,
        wetted_surface=12000,
        frontal_area=1200
    )

@pytest.fixture
def calm_conditions():
    """Calm weather operating conditions."""
    return OperatingConditions(
        speed=15,
        wind_speed=0,
        wind_angle=0,
        wave_height=0,
        wave_period=0,
        wave_angle=0
    )

@pytest.fixture
def moderate_weather():
    """Moderate weather conditions."""
    return OperatingConditions(
        speed=15,
        wind_speed=10,
        wind_angle=45,
        wave_height=2,
        wave_period=8,
        wave_angle=30
    )
```

---

## Documentation Standards

### Module Docstrings

```python
"""Friction resistance calculation module.

This module implements friction resistance calculations using the ITTC 1957
correlation line. It includes functions for Reynolds number calculation,
friction coefficient estimation, and total friction resistance.

Classes:
    FrictionResistance: Calculator for ship friction resistance

Functions:
    ittc_friction_coefficient: ITTC 1957 friction coefficient
    calculate_reynolds_number: Reynolds number for ship flow

References:
    ITTC (1957). "Report of the Resistance Committee."

Example:
    >>> from ship_performance.resistance.friction import FrictionResistance
    >>> from ship_performance.core import ShipParameters, OperatingConditions
    >>>
    >>> ship = ShipParameters(length=150, beam=25, draft=8, ...)
    >>> conditions = OperatingConditions(speed=15)
    >>>
    >>> calc = FrictionResistance()
    >>> R_f = calc.calculate(ship, conditions)
    >>> print(f"Friction resistance: {R_f:.0f} N")
"""
```

### API Reference Generation

Use Sphinx or MkDocs with automatic API extraction:

```bash
# Install documentation tools
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Or for MkDocs
pip install mkdocs mkdocs-material mkdocstrings[python]

# Generate documentation
cd docs
make html  # For Sphinx
# OR
mkdocs build  # For MkDocs
```

---

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone <repo-url>
cd ship_performance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Development Cycle

```bash
# Create feature branch
git checkout -b mvp-1-core-models

# Make changes...

# Run tests
pytest -v

# Run type checking
mypy src/

# Format code
black src/ tests/

# Check coverage
pytest --cov=ship_performance --cov-report=html

# Commit (pre-commit hooks run automatically)
git add .
git commit -m "Add ShipParameters dataclass with validation"

# Push
git push origin mvp-1-core-models
```

### 3. Pre-Commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203]
```

---

## Dependencies & Tools

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ship_performance"
version = "0.1.0"
description = "Ship performance prediction model with fuel consumption estimation"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "pydantic>=2.0.0",  # For data validation
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "hypothesis>=6.92.0",
    "black>=23.12.0",
    "mypy>=1.8.0",
    "flake8>=7.0.0",
    "isort>=5.13.0",
    "pre-commit>=3.5.0",
]

docs = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=2.0.0",
    "sphinx-autodoc-typehints>=1.25.0",
]

viz = [
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
]

notebooks = [
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
    "ipywidgets>=8.1.0",
]

all = [
    "ship_performance[dev,docs,viz,notebooks]",
]

[project.urls]
Homepage = "https://github.com/yourusername/ship_performance"
Documentation = "https://ship-performance.readthedocs.io"
Repository = "https://github.com/yourusername/ship_performance"
Changelog = "https://github.com/yourusername/ship_performance/blob/main/CHANGELOG.md"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311', 'py312']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = ["scipy.*", "matplotlib.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--cov=ship_performance",
    "--cov-report=term-missing",
    "--cov-report=html",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/conftest.py"]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

---

## Summary Checklist

For every Python file in the project:

- [ ] Type hints on all public functions and methods
- [ ] Comprehensive docstrings (Google or NumPy style)
- [ ] Unit tests with ≥ 80% coverage
- [ ] Black formatted (100 char line length)
- [ ] Mypy strict mode passes
- [ ] Follows SOLID principles
- [ ] No print statements (use logging)
- [ ] Proper error handling with specific exceptions
- [ ] Examples in docstrings
- [ ] No TODO comments (create issues instead)

**Remember:** The goal is production-quality code that you'd be proud to show in a portfolio or use in a real engineering application.
