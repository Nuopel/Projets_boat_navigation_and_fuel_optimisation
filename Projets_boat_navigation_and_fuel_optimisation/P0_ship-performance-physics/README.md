# Ship Performance Prediction Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

A comprehensive Python package for predicting ship fuel consumption under varying operational conditions using physics-based models.

## 🎯 Project Overview

This project implements a complete ship performance prediction model that calculates fuel consumption based on:

- **Ship parameters:** Length, beam, draft, displacement, hull coefficients
- **Operating conditions:** Speed, wind, waves, sea state
- **Physics-based models:** Simplified Holtrop-Mennen resistance, propulsion efficiency, fuel consumption

**Status:** ✅ **COMPLETE** - All 5 MVPs Successfully Delivered!

### Current Progress

- [x] Project planning and architecture
- [x] WBS structure with 5 MVPs defined
- [x] **MVP-1: Foundation & Core Abstractions ✅**
- [x] **MVP-2: Calm Water Resistance ✅**
- [x] **MVP-3: Wind Resistance (Windage) ✅**
- [x] **MVP-4: Wave Resistance & Complete Integration ✅**
- [x] **MVP-5: Propulsion & Fuel Consumption ✅**

## 📚 Documentation

- **[Project Review](PROJECT0_OVERVIEW_LIMITS.md)** - Comprehensive project analysis and scope
- **[WBS Structure](docs/WBS_MVP_STRUCTURE.md)** - Detailed work breakdown with MVP plans
- **[Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md)** - Architecture, design patterns, best practices

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ship_performance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Basic Usage

```python
from ship_performance.core import ShipParameters, OperatingConditions
from ship_performance.models import ShipPerformanceModel

# Define a cargo vessel
ship = ShipParameters(
    length=150,        # meters
    beam=25,           # meters
    draft=8,           # meters
    displacement=15000,  # tonnes
    block_coefficient=0.7
)

# Define operating conditions
conditions = OperatingConditions(
    speed=15,          # knots
    wind_speed=10,     # m/s
    wind_angle=45,     # degrees
    wave_height=2,     # meters (Hs)
    wave_period=8,     # seconds
    wave_angle=30      # degrees
)

# Predict performance
model = ShipPerformanceModel()
result = model.predict_fuel_consumption(ship, conditions)

print(f"Total Resistance: {result.resistance_total/1000:.1f} kN")
print(f"Effective Power: {result.effective_power:.1f} kW")
print(f"Fuel Rate: {result.fuel_rate:.1f} kg/h")
```

*Note: Full API available after MVP-5 completion*

## 🏗️ Project Structure

```
ship_performance/
├── src/ship_performance/    # Main package
│   ├── core/                 # Core data models
│   ├── resistance/           # Resistance calculators
│   ├── propulsion/           # Propulsion models
│   ├── models/               # Integrated performance models
│   └── utils/                # Utilities and constants
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Example scripts
## 🧪 Development

```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ship_performance --cov-report=html

# Run specific test file
pytest tests/core/test_ship_parameters.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run all checks (via pre-commit)
pre-commit run --all-files
```

## 📊 Features

### Implemented (✅) vs Planned (📅)

| Feature                                | Status | MVP   |
| -------------------------------------- | ------ | ----- |
| Ship parameter validation              | ✅      | MVP-1 |
| Operating conditions model             | ✅      | MVP-1 |
| Calm water resistance (Holtrop-Mennen) | ✅      | MVP-2 |
| Wind resistance (windage)              | ✅      | MVP-3 |
| Added resistance in waves              | ✅      | MVP-4 |
| Complete resistance integration        | ✅      | MVP-4 |
| Propulsion efficiency model            | ✅      | MVP-5 |
| Fuel consumption prediction            | ✅      | MVP-5 |
|                                        |        |       |
| Comprehensive documentation            | ✅      | MVP-5 |

## 🔬 Technical Approach

### Physics Models

1. **Calm Water Resistance**
   
   - Friction resistance: ITTC 1957 correlation line
   - Wave-making resistance: Holtrop-Mennen method
   - Form factor and appendage corrections

2. **Environmental Resistance**
   
   - Wind resistance: Aerodynamic drag formula
   - Added wave resistance: Simplified Kwon method

3. **Propulsion**
   
   - Propeller efficiency: Wageningen B-series (or simplified)
   - Shaft losses and mechanical efficiency
   - Engine SFOC (Specific Fuel Oil Consumption)

### Design Principles

- **SOLID principles** for maintainable code
- **Separation of Concerns** - modular components
- **Composition over Inheritance** - flexible architecture
- **Type Safety** - comprehensive type hints with mypy
- **Test Coverage** - target ≥80% coverage
- **Documentation** - complete API documentation

## 📖 References

### Key Scientific Papers

1. Holtrop, J., & Mennen, G. G. J. (1982). "An approximate power prediction method." *International Shipbuilding Progress*, 29(335), 166-170.

2. ITTC (1957). "Report of Resistance Committee." *International Towing Tank Conference*.

3. Kwon, Y. J. (2008). "Speed loss due to added resistance in wind and waves." *The Naval Architect*, 14-16.

### Code Resources

- [PyResis](https://github.com/MaritimeRenewable/PyResis) - Ship resistance estimation package
- [Resistance_Calculation](https://github.com/vinibangera09/Resistance_Calculation) - Holtrop-Mennen implementation

## 🤝 Contributing

This project follows an MVP-based development approach. Each MVP is a complete, tested, documented increment.

### Development Workflow

1. Check current MVP in [WBS_MVP_STRUCTURE.md](WBS_MVP_STRUCTURE.md)
2. Create feature branch: `git checkout -b mvp-X-feature-name`
3. Implement with tests (TDD recommended)
4. Ensure all quality checks pass
5. Create demo/notebook
6. Submit for review

### Code Standards

- **Type hints** on all public functions
- **Docstrings** (Google or NumPy style)
- **Unit tests** with ≥80% coverage
- **Black** formatting (100 char line length)
- **Mypy** strict mode passes
- **No print statements** (use logging)

See [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) for detailed guidelines.

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙋 Support

For questions or issues:

- Check documentation in `docs/`
- Review [WBS_MVP_STRUCTURE.md](WBS_MVP_STRUCTURE.md) for current status
- Open an issue on GitHub

---

**Note:** This is a learning/portfolio project demonstrating:

- Advanced Python package development
- Physics-based modeling
- Software engineering best practices
- Test-driven development
- Comprehensive documentation

Built with ❤️ using Python, following industry best practices.
