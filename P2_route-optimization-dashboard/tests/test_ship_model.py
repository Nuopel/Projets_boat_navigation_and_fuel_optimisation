"""
Unit tests for ship dynamics and fuel consumption model.

Tests the ShipDynamics class and fuel consumption calculations.
"""

import pytest
import numpy as np
from src.models.ship_model import ShipSpecifications, ShipDynamics, create_default_ship


class TestShipSpecifications:
    """Test ShipSpecifications dataclass validation."""

    def test_valid_specifications(self):
        """Test creating valid ship specifications."""
        specs = ShipSpecifications(
            name="Test Ship",
            v_min=10.0,
            v_max=20.0,
            fuel_coef_speed=0.5,
            fuel_coef_wind=0.8,
            fuel_coef_wave=15.0,
            fuel_base=500.0
        )
        assert specs.name == "Test Ship"
        assert specs.v_min == 10.0
        assert specs.v_max == 20.0
        assert specs.emission_factor == 2.8  # default

    def test_invalid_v_min(self):
        """Test that negative v_min raises ValueError."""
        with pytest.raises(ValueError, match="v_min must be positive"):
            ShipSpecifications(
                name="Bad Ship",
                v_min=-5.0,
                v_max=20.0,
                fuel_coef_speed=0.5,
                fuel_coef_wind=0.8,
                fuel_coef_wave=15.0,
                fuel_base=500.0
            )

    def test_invalid_v_max(self):
        """Test that v_max <= v_min raises ValueError."""
        with pytest.raises(ValueError, match="v_max.*must be greater than v_min"):
            ShipSpecifications(
                name="Bad Ship",
                v_min=20.0,
                v_max=15.0,
                fuel_coef_speed=0.5,
                fuel_coef_wind=0.8,
                fuel_coef_wave=15.0,
                fuel_base=500.0
            )

    def test_invalid_fuel_coef(self):
        """Test that negative fuel_coef_speed raises ValueError."""
        with pytest.raises(ValueError, match="fuel_coef_speed must be non-negative"):
            ShipSpecifications(
                name="Bad Ship",
                v_min=10.0,
                v_max=20.0,
                fuel_coef_speed=-0.5,
                fuel_coef_wind=0.8,
                fuel_coef_wave=15.0,
                fuel_base=500.0
            )


class TestShipDynamicsFuelRate:
    """Test fuel consumption rate calculations."""

    @pytest.fixture
    def ship(self):
        """Create a test ship with known coefficients."""
        specs = ShipSpecifications(
            name="Test Ship",
            v_min=8.0,
            v_max=18.0,
            fuel_coef_speed=0.5,  # L/h per knot³
            fuel_coef_wind=1.0,   # L/h per knot²
            fuel_coef_wave=10.0,  # L/h per meter
            fuel_base=100.0       # L/h
        )
        return ShipDynamics(specs)

    def test_fuel_rate_calm_weather(self, ship):
        """Test fuel rate in calm conditions (W=0, H=0)."""
        velocity = 12.0  # knots
        fuel_rate = ship.fuel_rate(velocity, wind_speed=0.0, wave_height=0.0)

        # Expected: 0.5 * 12³ + 1.0 * 0² + 10.0 * 0 + 100 = 0.5 * 1728 + 100 = 964
        expected = 0.5 * (12 ** 3) + 100.0
        assert pytest.approx(fuel_rate, rel=0.01) == expected

    def test_fuel_rate_cubic_relationship(self, ship):
        """Test that fuel increases cubically with speed (W=0, H=0)."""
        v1 = 10.0
        v2 = 18.0  # Use v_max instead of 20.0

        f1 = ship.fuel_rate(v1, 0.0, 0.0)
        f2 = ship.fuel_rate(v2, 0.0, 0.0)

        # In calm weather: f(V) = 0.5*V³ + 100
        # Ratio should be approximately (V2³)/(V1³) for large speeds
        # f2/f1 ≈ (0.5*18³ + 100) / (0.5*10³ + 100) = 2824 / 600 ≈ 4.7
        # Note: not exactly (18/10)³ = 5.83 because of base term

        assert f2 > f1  # Fuel increases with speed
        # 1.8× speed should increase fuel by factor > 4
        assert f2 / f1 > 4.0

    def test_weather_penalty(self, ship):
        """Test that weather increases fuel consumption."""
        velocity = 15.0

        # Calm conditions
        fuel_calm = ship.fuel_rate(velocity, wind_speed=0.0, wave_height=0.0)

        # Stormy conditions
        fuel_stormy = ship.fuel_rate(velocity, wind_speed=25.0, wave_height=5.0)

        # Stormy should consume more fuel
        assert fuel_stormy > fuel_calm

        # Calculate penalty percentage
        penalty_pct = ((fuel_stormy - fuel_calm) / fuel_calm) * 100

        # Expect 20-60% increase in stormy conditions
        # fuel_calm = 0.5*15³ + 100 = 1787.5
        # fuel_stormy = 0.5*15³ + 1.0*25² + 10.0*5 + 100 = 1787.5 + 625 + 50 = 2462.5
        # penalty = (2462.5 - 1787.5) / 1787.5 = 37.8%
        assert 20 < penalty_pct < 60

    def test_speed_bounds_enforcement(self, ship):
        """Test that speed outside [v_min, v_max] raises ValueError."""
        # Below minimum
        with pytest.raises(ValueError, match="below minimum"):
            ship.fuel_rate(5.0, 0.0, 0.0)

        # Above maximum
        with pytest.raises(ValueError, match="exceeds maximum"):
            ship.fuel_rate(25.0, 0.0, 0.0)

        # Within bounds should work
        fuel_min = ship.fuel_rate(8.0, 0.0, 0.0)
        fuel_max = ship.fuel_rate(18.0, 0.0, 0.0)
        assert fuel_min > 0
        assert fuel_max > fuel_min

    def test_non_negative_fuel(self, ship):
        """Test that fuel rate is always non-negative."""
        # Even with negative coefficients somehow, result is clamped to 0
        velocities = [8.0, 10.0, 12.0, 15.0, 18.0]
        for v in velocities:
            fuel = ship.fuel_rate(v, 0.0, 0.0)
            assert fuel >= 0


class TestShipDynamicsEmissions:
    """Test CO2 emissions calculations."""

    @pytest.fixture
    def ship(self):
        """Create a test ship."""
        return create_default_ship()

    def test_emissions_proportional_to_fuel(self, ship):
        """Test that emissions are proportional to fuel consumption."""
        velocity = 15.0
        wind = 10.0
        wave = 2.0

        fuel_rate = ship.fuel_rate(velocity, wind, wave)
        emissions_rate = ship.emissions_rate(velocity, wind, wave)

        # emissions = fuel * emission_factor
        expected_emissions = fuel_rate * ship.specs.emission_factor

        assert pytest.approx(emissions_rate, rel=0.001) == expected_emissions


class TestShipDynamicsSegmentCalculation:
    """Test fuel/time/emissions calculations for route segments."""

    @pytest.fixture
    def ship(self):
        """Create a test ship."""
        return create_default_ship()

    def test_fuel_for_segment(self, ship):
        """Test calculating fuel for a route segment."""
        distance = 100.0  # nautical miles
        velocity = 10.0  # knots

        time_h, fuel_l, co2_kg = ship.fuel_for_segment(
            distance, velocity, wind_speed=0.0, wave_height=0.0
        )

        # Time = distance / speed = 100 / 10 = 10 hours
        assert pytest.approx(time_h, rel=0.01) == 10.0

        # Fuel = fuel_rate * time
        expected_fuel_rate = ship.fuel_rate(velocity, 0.0, 0.0)
        expected_fuel = expected_fuel_rate * 10.0
        assert pytest.approx(fuel_l, rel=0.01) == expected_fuel

        # CO2 = fuel * emission_factor
        expected_co2 = expected_fuel * ship.specs.emission_factor
        assert pytest.approx(co2_kg, rel=0.01) == expected_co2

    def test_segment_with_weather(self, ship):
        """Test segment calculation with weather penalties."""
        distance = 50.0
        velocity = 12.0
        wind = 20.0
        wave = 4.0

        time_h, fuel_l, co2_kg = ship.fuel_for_segment(distance, velocity, wind, wave)

        # Time = 50 / 12 ≈ 4.17 hours
        expected_time = distance / velocity
        assert pytest.approx(time_h, rel=0.01) == expected_time

        # Fuel should be higher than calm weather
        time_calm, fuel_calm, _ = ship.fuel_for_segment(distance, velocity, 0.0, 0.0)
        assert fuel_l > fuel_calm


class TestOptimalSpeed:
    """Test optimal speed calculations."""

    @pytest.fixture
    def ship(self):
        """Create a test ship."""
        return create_default_ship()

    def test_optimal_speed_in_range(self, ship):
        """Test that optimal speed is within operational range."""
        v_opt = ship.optimal_speed_calm_weather()

        assert v_opt >= ship.specs.v_min
        assert v_opt <= ship.specs.v_max

    def test_optimal_speed_minimizes_fuel_per_distance(self, ship):
        """Test that optimal speed approximately minimizes fuel per nm."""
        v_opt = ship.optimal_speed_calm_weather()

        # Calculate fuel per nautical mile at different speeds
        distance = 100.0
        speeds = np.linspace(ship.specs.v_min, ship.specs.v_max, 20)
        fuel_per_nm = []

        for v in speeds:
            _, fuel, _ = ship.fuel_for_segment(distance, v, 0.0, 0.0)
            fuel_per_nm.append(fuel / distance)

        fuel_per_nm = np.array(fuel_per_nm)

        # Optimal speed should be near the minimum fuel per nm
        min_idx = np.argmin(fuel_per_nm)
        v_min_fuel = speeds[min_idx]

        # Allow some tolerance (within 10% of speed range)
        tolerance = (ship.specs.v_max - ship.specs.v_min) * 0.1
        assert abs(v_opt - v_min_fuel) < tolerance


class TestDefaultShip:
    """Test default ship creation."""

    def test_create_default_ship(self):
        """Test that create_default_ship() returns valid ShipDynamics."""
        ship = create_default_ship()

        assert isinstance(ship, ShipDynamics)
        assert ship.specs.name == "Generic Cargo Vessel"
        assert ship.specs.v_min > 0
        assert ship.specs.v_max > ship.specs.v_min
        assert ship.specs.fuel_coef_speed >= 0

        # Should be able to calculate fuel rate
        fuel = ship.fuel_rate(12.0, 10.0, 2.0)
        assert fuel > 0
