# Work Breakdown Structure - MVP-Based Development Plan

**Project:** Ship Performance Prediction Model (Navig_P0)
**Approach:** Iterative MVP Development
**Total MVPs:** 5
**Estimated Duration:** 3-5 weeks

---

## WBS Overview

Each MVP is a **fully functional, testable, and demonstrable** increment that builds upon the previous one. Each delivers working code with tests, documentation, and example usage.

```
MVP-1: Foundation & Core Abstractions
  └─> Deliverable: Basic ship model with parameter validation

MVP-2: Calm Water Resistance
  └─> Deliverable: Working resistance calculator with Holtrop-Mennen

MVP-3: Environmental Effects (Wind)
  └─> Deliverable: Windage calculator integrated into resistance model

MVP-4: Wave Resistance & Integration
  └─> Deliverable: Complete resistance model with all components

MVP-5: Propulsion, Fuel Consumption & Validation
  └─> Deliverable: Full performance model with validation report
```

---

## MVP-1: Foundation & Core Abstractions

**Goal:** Establish the architectural foundation with core data models and interfaces

**Duration:** 3-5 days | **Effort:** 12-18 hours

### 1.1 Project Setup (WBS 1.1)
**Priority:** Critical | **Complexity:** Low

**Tasks:**
- [ ] Initialize Python package structure with `pyproject.toml`
- [ ] Configure development tools (black, mypy, pytest, pre-commit)
- [ ] Set up virtual environment and dependencies
- [ ] Create README with project overview
- [ ] Initialize git repository with .gitignore

**Deliverables:**
- `pyproject.toml` with all dependencies
- `setup.cfg` or `pyproject.toml` with tool configurations
- `.pre-commit-config.yaml`
- Working `make` or `poetry` commands for test/lint/format

**Acceptance Criteria:**
- ✅ `pytest` runs successfully (even with 0 tests)
- ✅ `black .` and `mypy .` pass without errors
- ✅ Pre-commit hooks installed and functional

---

### 1.2 Core Data Models (WBS 1.2)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Create `ShipParameters` dataclass/Pydantic model
- [ ] Create `OperatingConditions` dataclass/Pydantic model
- [ ] Implement parameter validation (ranges, units)
- [ ] Create unit conversion utilities (knots↔m/s, etc.)
- [ ] Add comprehensive docstrings with parameter descriptions

**Deliverables:**
- `src/ship_performance/core/ship_parameters.py`
- `src/ship_performance/core/operating_conditions.py`
- `src/ship_performance/utils/units.py`
- `tests/core/test_ship_parameters.py`
- `tests/core/test_operating_conditions.py`

**Acceptance Criteria:**
- ✅ All parameters have type hints and validation
- ✅ Invalid inputs raise appropriate exceptions
- ✅ Unit conversions are bidirectional and accurate
- ✅ Test coverage ≥ 90% for core models
- ✅ Examples in docstrings

**Example Code Structure:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ShipParameters:
    """Ship hull and geometric parameters.

    Attributes:
        length: Length overall (m)
        beam: Beam at waterline (m)
        draft: Draft at design condition (m)
        displacement: Displacement (tonnes)
        block_coefficient: Block coefficient Cb (0.5-0.9)
        wetted_surface: Wetted surface area (m²)
        frontal_area: Frontal windage area (m²)
    """
    length: float  # m
    beam: float    # m
    draft: float   # m
    displacement: float  # tonnes
    block_coefficient: float  # dimensionless
    wetted_surface: Optional[float] = None  # m²
    frontal_area: Optional[float] = None    # m²

    def __post_init__(self):
        """Validate parameters and compute derived values."""
        self._validate()
        if self.wetted_surface is None:
            self.wetted_surface = self._estimate_wetted_surface()
        if self.frontal_area is None:
            self.frontal_area = self._estimate_frontal_area()

    def _validate(self):
        """Validate parameter ranges."""
        # Implementation
        pass
```

---

### 1.3 Abstract Interfaces (WBS 1.3)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Define `ResistanceCalculator` protocol/ABC
- [ ] Define `PropulsionModel` protocol/ABC
- [ ] Define `PerformanceModel` interface
- [ ] Create type aliases for common types

**Deliverables:**
- `src/ship_performance/interfaces.py`
- Documentation of interface contracts
- Type stubs if needed

**Acceptance Criteria:**
- ✅ Clear interface definitions using `Protocol` or `ABC`
- ✅ Documented input/output contracts
- ✅ Mypy validates interface usage

**Example Interface:**
```python
from typing import Protocol
from .core import ShipParameters, OperatingConditions

class ResistanceCalculator(Protocol):
    """Protocol for resistance calculation components."""

    def calculate(
        self,
        ship: ShipParameters,
        conditions: OperatingConditions
    ) -> float:
        """Calculate resistance component in Newtons.

        Args:
            ship: Ship parameters
            conditions: Operating conditions

        Returns:
            Resistance in Newtons
        """
        ...

    @property
    def name(self) -> str:
        """Component name for reporting."""
        ...
```

---

### 1.4 MVP-1 Integration & Demo (WBS 1.4)
**Priority:** Critical | **Complexity:** Low

**Tasks:**
- [ ] Create example script demonstrating data models
- [ ] Write MVP-1 completion report
- [ ] Create notebook: `01_foundation_demo.ipynb`
- [ ] Update README with MVP-1 status

**Deliverables:**
- `examples/mvp1_foundation.py`
- `notebooks/01_foundation_demo.ipynb`
- `docs/mvp1_completion_report.md`

**Acceptance Criteria:**
- ✅ Example creates ship and conditions successfully
- ✅ All tests pass
- ✅ Documentation is complete
- ✅ Demo notebook runs without errors

**MVP-1 Demo Example:**
```python
from ship_performance.core import ShipParameters, OperatingConditions

# Create a typical cargo vessel
ship = ShipParameters(
    length=150,
    beam=25,
    draft=8,
    displacement=15000,
    block_coefficient=0.7
)

# Define operating conditions
conditions = OperatingConditions(
    speed=15,  # knots
    wind_speed=10,  # m/s
    wind_angle=45,  # degrees
    wave_height=2,  # m
    wave_period=8,  # s
    wave_angle=30  # degrees
)

print(f"Ship: {ship.length}m x {ship.beam}m")
print(f"Wetted surface (estimated): {ship.wetted_surface:.1f} m²")
print(f"Conditions: {conditions.speed} knots, Hs={conditions.wave_height}m")
```

---

## MVP-2: Calm Water Resistance

**Goal:** Implement and validate calm water resistance calculations

**Duration:** 4-6 days | **Effort:** 18-24 hours

### 2.1 Friction Resistance (WBS 2.1)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Implement ITTC 1957 friction line formula
- [ ] Calculate Reynolds number and friction coefficient
- [ ] Implement form factor estimation
- [ ] Add correlation allowance

**Deliverables:**
- `src/ship_performance/resistance/friction.py`
- `tests/resistance/test_friction.py`
- Validation against ITTC standard values

**Acceptance Criteria:**
- ✅ Friction resistance matches ITTC formulas
- ✅ Reynolds number calculated correctly
- ✅ Test cases cover Re range: 10⁶ - 10⁹
- ✅ Comparison with PyResis implementation (±2%)

**Key Formula:**
```python
def ittc_friction_coefficient(reynolds_number: float) -> float:
    """ITTC 1957 friction line.

    Cf = 0.075 / (log10(Re) - 2)²
    """
    return 0.075 / (np.log10(reynolds_number) - 2) ** 2
```

---

### 2.2 Wave-Making Resistance (WBS 2.2)
**Priority:** Critical | **Complexity:** High

**Tasks:**
- [ ] Implement Holtrop-Mennen wave resistance formula
- [ ] Calculate Froude number effects
- [ ] Handle bulbous bow correction (if applicable)
- [ ] Implement transom stern correction

**Deliverables:**
- `src/ship_performance/resistance/wave_making.py`
- `tests/resistance/test_wave_making.py`
- Validation notebook showing resistance hump

**Acceptance Criteria:**
- ✅ Wave resistance shows characteristic hump at Fn ≈ 0.3-0.4
- ✅ Validates against Holtrop-Mennen published results
- ✅ Comparison with Resistance_Calculation repo (±5%)
- ✅ Handles edge cases (very slow/fast speeds)

---

### 2.3 Calm Water Integration (WBS 2.3)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Create `CalmWaterResistance` composite calculator
- [ ] Integrate friction + wave-making components
- [ ] Implement appendage resistance (simplified)
- [ ] Create resistance breakdown reporting

**Deliverables:**
- `src/ship_performance/resistance/calm_water.py`
- `tests/resistance/test_calm_water.py`
- Integration tests

**Acceptance Criteria:**
- ✅ Total calm water resistance = sum of components
- ✅ Breakdown available for analysis
- ✅ Matches PyResis total resistance (±5%)

**Example:**
```python
class CalmWaterResistance:
    """Calm water resistance calculator (Holtrop-Mennen method)."""

    def __init__(self):
        self.friction = FrictionResistance()
        self.wave_making = WaveMakingResistance()

    def calculate(self, ship: ShipParameters, conditions: OperatingConditions) -> float:
        """Calculate total calm water resistance."""
        R_f = self.friction.calculate(ship, conditions)
        R_w = self.wave_making.calculate(ship, conditions)
        return R_f + R_w

    def breakdown(self, ship: ShipParameters, conditions: OperatingConditions) -> dict:
        """Return resistance component breakdown."""
        return {
            'friction': self.friction.calculate(ship, conditions),
            'wave_making': self.wave_making.calculate(ship, conditions)
        }
```

---

### 2.4 MVP-2 Validation & Demo (WBS 2.4)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Create validation notebook comparing with benchmarks
- [ ] Generate R(V) curves for different ship types
- [ ] Validate Froude number effects
- [ ] Compare with PyResis and Resistance_Calculation

**Deliverables:**
- `notebooks/02_calm_water_validation.ipynb`
- `docs/mvp2_validation_report.md`
- `examples/mvp2_resistance_curves.py`

**Acceptance Criteria:**
- ✅ R(V) curve shows expected quadratic-cubic trend
- ✅ Resistance hump visible at appropriate Froude number
- ✅ Agreement with reference implementations (±5%)
- ✅ Sensitivity analysis on key parameters (Cb, L/B ratio)

---

## MVP-3: Environmental Effects - Wind Resistance

**Goal:** Add windage/wind resistance calculation

**Duration:** 2-3 days | **Effort:** 10-14 hours

### 3.1 Wind Resistance Calculator (WBS 3.1)
**Priority:** High | **Complexity:** Low-Medium

**Tasks:**
- [ ] Implement aerodynamic drag formula
- [ ] Calculate relative wind velocity and angle
- [ ] Implement drag coefficient estimation
- [ ] Handle different wind directions

**Deliverables:**
- `src/ship_performance/resistance/windage.py`
- `tests/resistance/test_windage.py`

**Acceptance Criteria:**
- ✅ Wind resistance calculated from relative wind
- ✅ Drag coefficient reasonable (Cd ≈ 0.4-0.8 for ships)
- ✅ Validates against published windage coefficients
- ✅ Test coverage ≥ 85%

**Formula:**
```python
def calculate_wind_resistance(
    frontal_area: float,
    relative_wind_speed: float,
    drag_coefficient: float = 0.6
) -> float:
    """Calculate wind resistance.

    R_wind = 0.5 * ρ_air * Cd * A * V_rel²

    Args:
        frontal_area: Frontal windage area (m²)
        relative_wind_speed: Relative wind speed (m/s)
        drag_coefficient: Aerodynamic drag coefficient

    Returns:
        Wind resistance (N)
    """
    rho_air = 1.225  # kg/m³ at sea level
    return 0.5 * rho_air * drag_coefficient * frontal_area * relative_wind_speed**2
```

---

### 3.2 Integration & Demo (WBS 3.2)
**Priority:** High | **Complexity:** Low

**Tasks:**
- [ ] Create `TotalResistance` composite (calm + wind)
- [ ] Add wind sensitivity analysis
- [ ] Update examples and notebooks

**Deliverables:**
- `src/ship_performance/resistance/total_resistance.py`
- `notebooks/03_wind_effects.ipynb`
- `examples/mvp3_wind_sensitivity.py`

**Acceptance Criteria:**
- ✅ Total resistance = calm + wind
- ✅ Wind contribution reasonable (typically 5-15% in moderate winds)
- ✅ Sensitivity to wind angle demonstrated
- ✅ All tests pass

---

## MVP-4: Wave Resistance & Complete Integration

**Goal:** Add added resistance in waves and create complete resistance model

**Duration:** 5-7 days | **Effort:** 20-28 hours

### 4.1 Added Wave Resistance (WBS 4.1)
**Priority:** High | **Complexity:** High

**Tasks:**
- [ ] Research and select simplified method (Kwon, STAWAVE-1, or similar)
- [ ] Implement wave spectrum (Pierson-Moskowitz or JONSWAP)
- [ ] Calculate added resistance from Hs, Tp, heading
- [ ] Validate against published results

**Deliverables:**
- `src/ship_performance/resistance/added_waves.py`
- `src/ship_performance/utils/wave_spectrum.py`
- `tests/resistance/test_added_waves.py`
- `docs/wave_resistance_theory.md`

**Acceptance Criteria:**
- ✅ Added resistance increases with wave height
- ✅ Head seas > beam seas > following seas
- ✅ Reasonable magnitude (can equal calm water resistance in storm)
- ✅ Validates against literature benchmarks

**Simplified Approach:**
```python
def calculate_added_resistance_waves(
    ship: ShipParameters,
    conditions: OperatingConditions
) -> float:
    """Simplified added resistance in waves (empirical method).

    Based on Kwon (2008) or similar simplified approach.
    ΔR_waves = f(Hs², Tp, Fn, heading, ship_geometry)
    """
    # Simplified empirical correlation
    Hs = conditions.wave_height
    lambda_wave = 1.56 * conditions.wave_period ** 2  # deep water

    # Non-dimensional wave height
    Hs_L = Hs / ship.length

    # Heading factor (0-180 degrees, max at head seas)
    heading_factor = np.cos(np.radians(conditions.wave_angle))

    # Empirical coefficient (to be calibrated)
    C_aw = estimate_wave_resistance_coefficient(ship)

    # Added resistance
    R_aw = C_aw * ship.displacement * 9.81 * Hs_L**2 * heading_factor

    return max(0, R_aw)
```

---

### 4.2 Complete Resistance Model (WBS 4.2)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Create `ShipResistanceModel` integrating all components
- [ ] Implement resistance breakdown reporting
- [ ] Add visualization utilities
- [ ] Create comprehensive test suite

**Deliverables:**
- `src/ship_performance/models/resistance_model.py`
- `src/ship_performance/utils/visualization.py`
- `tests/models/test_resistance_model.py`

**Acceptance Criteria:**
- ✅ All resistance components integrated
- ✅ Breakdown available for analysis
- ✅ Visualization of resistance components vs speed
- ✅ Integration tests validate complete workflow

**Example:**
```python
class ShipResistanceModel:
    """Complete ship resistance model with all components."""

    def __init__(self):
        self.calm_water = CalmWaterResistance()
        self.windage = WindResistance()
        self.added_waves = AddedWaveResistance()

    def calculate_total_resistance(
        self,
        ship: ShipParameters,
        conditions: OperatingConditions
    ) -> float:
        """Calculate total resistance from all components."""
        R_calm = self.calm_water.calculate(ship, conditions)
        R_wind = self.windage.calculate(ship, conditions)
        R_waves = self.added_waves.calculate(ship, conditions)
        return R_calm + R_wind + R_waves

    def get_breakdown(self, ship, conditions) -> dict:
        """Get resistance breakdown by component."""
        return {
            'calm_water': self.calm_water.calculate(ship, conditions),
            'windage': self.windage.calculate(ship, conditions),
            'added_waves': self.added_waves.calculate(ship, conditions)
        }
```

---

### 4.3 MVP-4 Validation & Demo (WBS 4.3)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Create comprehensive validation notebook
- [ ] Sensitivity analysis: all environmental parameters
- [ ] Generate resistance maps (speed vs Hs, wind, etc.)
- [ ] Compare component contributions

**Deliverables:**
- `notebooks/04_complete_resistance_model.ipynb`
- `docs/mvp4_validation_report.md`
- `examples/mvp4_environmental_sensitivity.py`

**Acceptance Criteria:**
- ✅ Resistance ratios realistic (calm >> wind ≈ waves in moderate conditions)
- ✅ Environmental effects demonstrate expected trends
- ✅ Comprehensive sensitivity analysis documented
- ✅ Visualizations publication-quality

---

## MVP-5: Propulsion, Fuel Consumption & Final Validation

**Goal:** Complete performance model with propulsion and fuel consumption

**Duration:** 4-6 days | **Effort:** 18-24 hours

### 5.1 Propulsion Model (WBS 5.1)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Implement propeller efficiency model (Wageningen B-series or simple)
- [ ] Calculate delivered and brake power
- [ ] Implement shaft efficiency
- [ ] Handle bollard pull and off-design conditions

**Deliverables:**
- `src/ship_performance/propulsion/propeller.py`
- `src/ship_performance/propulsion/engine.py`
- `tests/propulsion/test_propeller.py`

**Acceptance Criteria:**
- ✅ Power chain: P_effective → P_delivered → P_brake
- ✅ Propeller efficiency reasonable (0.5-0.7 typical)
- ✅ Validates against known efficiency curves
- ✅ Test coverage ≥ 80%

**Power Chain:**
```python
def calculate_power_requirements(
    total_resistance: float,
    speed_ms: float,
    propeller_efficiency: float = 0.65,
    shaft_efficiency: float = 0.98
) -> dict:
    """Calculate power requirements.

    Returns:
        Dictionary with effective, delivered, and brake power (kW)
    """
    P_effective = total_resistance * speed_ms / 1000  # kW
    P_delivered = P_effective / propeller_efficiency
    P_brake = P_delivered / shaft_efficiency

    return {
        'effective_power': P_effective,
        'delivered_power': P_delivered,
        'brake_power': P_brake
    }
```

---

### 5.2 Fuel Consumption Model (WBS 5.2)
**Priority:** Critical | **Complexity:** Low-Medium

**Tasks:**
- [ ] Implement SFOC (Specific Fuel Oil Consumption) model
- [ ] Calculate fuel rate from brake power
- [ ] Add engine load factor effects
- [ ] Implement different fuel types (HFO, MDO, LNG)

**Deliverables:**
- `src/ship_performance/propulsion/fuel_consumption.py`
- `tests/propulsion/test_fuel_consumption.py`

**Acceptance Criteria:**
- ✅ Fuel rate calculated from power and SFOC
- ✅ Engine efficiency varies with load factor
- ✅ Realistic fuel consumption (typical SFOC: 170-220 g/kWh)
- ✅ Supports multiple fuel types

**Formula:**
```python
def calculate_fuel_consumption(
    brake_power_kw: float,
    sfoc_g_kwh: float = 185,
    fuel_lower_heating_value: float = 42700  # kJ/kg for HFO
) -> float:
    """Calculate fuel consumption rate.

    Args:
        brake_power_kw: Brake power (kW)
        sfoc_g_kwh: Specific fuel oil consumption (g/kWh)
        fuel_lower_heating_value: LHV of fuel (kJ/kg)

    Returns:
        Fuel consumption rate (kg/h)
    """
    return brake_power_kw * sfoc_g_kwh / 1000  # kg/h
```

---

### 5.3 Integrated Performance Model (WBS 5.3)
**Priority:** Critical | **Complexity:** Medium

**Tasks:**
- [ ] Create `ShipPerformanceModel` integrating resistance + propulsion
- [ ] Implement complete prediction workflow
- [ ] Add voyage simulation capabilities
- [ ] Create reporting and export utilities

**Deliverables:**
- `src/ship_performance/models/performance_model.py`
- `tests/models/test_performance_model.py`
- `examples/mvp5_complete_simulation.py`

**Acceptance Criteria:**
- ✅ End-to-end workflow: ship + conditions → fuel consumption
- ✅ All components integrated seamlessly
- ✅ Reporting includes all intermediate values
- ✅ Can simulate speed-consumption curves

**Complete Model:**
```python
class ShipPerformanceModel:
    """Complete ship performance prediction model."""

    def __init__(self):
        self.resistance_model = ShipResistanceModel()
        self.propulsion_model = PropulsionModel()
        self.fuel_model = FuelConsumptionModel()

    def predict_fuel_consumption(
        self,
        ship: ShipParameters,
        conditions: OperatingConditions
    ) -> PerformanceResult:
        """Predict fuel consumption for given ship and conditions.

        Returns:
            PerformanceResult with resistance, power, and fuel data
        """
        # Calculate total resistance
        R_total = self.resistance_model.calculate_total_resistance(ship, conditions)

        # Calculate power requirements
        power = self.propulsion_model.calculate_power(R_total, conditions.speed)

        # Calculate fuel consumption
        fuel_rate = self.fuel_model.calculate_consumption(power['brake_power'])

        return PerformanceResult(
            resistance_total=R_total,
            resistance_breakdown=self.resistance_model.get_breakdown(ship, conditions),
            effective_power=power['effective_power'],
            brake_power=power['brake_power'],
            fuel_rate=fuel_rate
        )
```

---

### 5.4 Comprehensive Validation (WBS 5.4)
**Priority:** Critical | **Complexity:** High

**Tasks:**
- [ ] Validate against Kaggle dataset (if using)
- [ ] Generate validation report with all success criteria
- [ ] Create parameter sensitivity study
- [ ] Benchmark against published results
- [ ] Statistical analysis of prediction accuracy

**Deliverables:**
- `notebooks/05_final_validation.ipynb`
- `docs/validation_report.md`
- `docs/sensitivity_analysis.md`
- Comparison tables and visualizations

**Acceptance Criteria:**
- ✅ R(V) curve follows quadratic-cubic trend
- ✅ Froude effects visible and correct
- ✅ Resistance component ratios realistic
- ✅ Fuel consumption predictions within ±15% of validation data
- ✅ Sensitivity analysis confirms expected parameter effects

---

### 5.5 Documentation & Polish (WBS 5.5)
**Priority:** High | **Complexity:** Medium

**Tasks:**
- [ ] Complete API documentation (all docstrings)
- [ ] Create user guide with examples
- [ ] Write theory documentation (equations, references)
- [ ] Create quickstart tutorial
- [ ] Add installation and contribution guides
- [ ] Generate API reference with Sphinx/MkDocs

**Deliverables:**
- `docs/user_guide.md`
- `docs/theory_reference.md`
- `docs/api_reference/` (auto-generated)
- `README.md` (comprehensive)
- `CONTRIBUTING.md`

**Acceptance Criteria:**
- ✅ All public APIs documented
- ✅ User guide covers all major use cases
- ✅ Theory document explains all formulas with references
- ✅ Quickstart gets user running in < 5 minutes
- ✅ Documentation builds without errors

---

### 5.6 Final Testing & CI/CD (WBS 5.6)
**Priority:** High | **Complexity:** Medium

**Tasks:**
- [ ] Achieve ≥ 80% test coverage
- [ ] Add property-based tests (hypothesis)
- [ ] Set up GitHub Actions CI/CD
- [ ] Add performance benchmarks
- [ ] Create release checklist

**Deliverables:**
- Comprehensive test suite
- `.github/workflows/ci.yml`
- `tests/property_tests/`
- Coverage report
- Performance benchmarks

**Acceptance Criteria:**
- ✅ Test coverage ≥ 80%
- ✅ CI/CD runs all tests, linting, type checking
- ✅ All tests pass on Python 3.9, 3.10, 3.11, 3.12
- ✅ Property tests validate invariants
- ✅ No performance regressions

---

## Development Guidelines for Each MVP

### General Workflow
1. **Read WBS for current MVP**
2. **Create feature branch:** `git checkout -b mvp-X-description`
3. **Implement tasks in order** (tests first when possible)
4. **Ensure all tests pass:** `pytest -v`
5. **Run quality checks:** `black . && mypy . && pytest`
6. **Create demo/notebook**
7. **Write completion report**
8. **Merge to main and tag:** `git tag mvp-X`

### Code Quality Checklist (Every File)
- [ ] Type hints on all functions
- [ ] Docstrings (Google or NumPy style)
- [ ] Unit tests with ≥ 80% coverage
- [ ] Black formatted
- [ ] Mypy passes
- [ ] No TODO comments (convert to issues)

### Documentation Checklist (Every MVP)
- [ ] Completion report written
- [ ] Example code working
- [ ] Notebook demonstrates functionality
- [ ] README updated with progress
- [ ] API changes documented

---

## Success Metrics per MVP

| MVP | Key Metric | Target |
|-----|------------|--------|
| MVP-1 | Test coverage | ≥ 90% |
| MVP-2 | Resistance accuracy vs PyResis | ± 5% |
| MVP-3 | Wind contribution realism | 5-15% of total in moderate winds |
| MVP-4 | Wave resistance magnitude | Can equal calm water in storms |
| MVP-5 | Fuel prediction accuracy | ± 15% vs validation data |

---

## Resource Allocation by MVP

| MVP | Effort (hours) | Priority | Dev Focus |
|-----|----------------|----------|-----------|
| MVP-1 | 12-18 | Critical | Architecture, foundations |
| MVP-2 | 18-24 | Critical | Physics, validation |
| MVP-3 | 10-14 | High | Environmental models |
| MVP-4 | 20-28 | High | Integration, complex physics |
| MVP-5 | 18-24 | Critical | Completion, validation |
| **Total** | **78-108** | | |

---

## Handoff to Dev AI

For each MVP, the manager AI should provide:

1. **Clear MVP specification** (this document section)
2. **Acceptance criteria checklist**
3. **Reference materials** (PyResis code, papers, etc.)
4. **Expected interfaces** (from previous MVPs)
5. **Validation approach** (test cases, benchmarks)

Dev AI should deliver:

1. **Working code** meeting all acceptance criteria
2. **Passing tests** with coverage report
3. **Demo notebook/script**
4. **Brief completion summary**
5. **Any issues/questions** encountered

---

## Next Steps

1. ✅ **Review and approve this WBS**
2. ✅ **Set up project structure** (see MVP-1.1)
3. ✅ **Create detailed technical architecture document**
4. ✅ **Begin MVP-1 development**
5. ⏳ **Iterate through MVPs with regular reviews**

**Recommendation:** Start with MVP-1 immediately. It's low-risk, establishes patterns, and provides immediate value.
