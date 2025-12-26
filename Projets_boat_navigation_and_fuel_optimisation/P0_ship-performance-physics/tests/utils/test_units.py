"""Tests for unit conversion utilities."""

import math

import pytest

from ship_performance.utils import units


class TestSpeedConversion:
    """Tests for speed unit conversions."""

    def test_knots_to_ms_typical(self) -> None:
        """Test conversion of typical ship speed."""
        result = units.knots_to_ms(15.0)
        assert pytest.approx(result, rel=1e-6) == 7.716666666666667

    def test_knots_to_ms_zero(self) -> None:
        """Test conversion of zero speed."""
        assert units.knots_to_ms(0.0) == 0.0

    def test_knots_to_ms_negative(self) -> None:
        """Test that negative speed raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            units.knots_to_ms(-10.0)

    def test_ms_to_knots_typical(self) -> None:
        """Test reverse conversion."""
        result = units.ms_to_knots(7.716666666666667)
        assert pytest.approx(result, rel=1e-6) == 15.0

    def test_ms_to_knots_zero(self) -> None:
        """Test conversion of zero speed."""
        assert units.ms_to_knots(0.0) == 0.0

    def test_ms_to_knots_negative(self) -> None:
        """Test that negative speed raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            units.ms_to_knots(-5.0)

    def test_bidirectional_conversion(self) -> None:
        """Test that conversions are reversible."""
        original = 20.0  # knots
        converted = units.knots_to_ms(original)
        back = units.ms_to_knots(converted)
        assert pytest.approx(back, rel=1e-10) == original


class TestAngleConversion:
    """Tests for angle unit conversions."""

    def test_degrees_to_radians_typical(self) -> None:
        """Test typical angle conversions."""
        assert pytest.approx(units.degrees_to_radians(180.0)) == math.pi
        assert pytest.approx(units.degrees_to_radians(90.0)) == math.pi / 2
        assert pytest.approx(units.degrees_to_radians(45.0)) == math.pi / 4

    def test_degrees_to_radians_zero(self) -> None:
        """Test zero angle."""
        assert units.degrees_to_radians(0.0) == 0.0

    def test_radians_to_degrees_typical(self) -> None:
        """Test reverse conversions."""
        assert pytest.approx(units.radians_to_degrees(math.pi)) == 180.0
        assert pytest.approx(units.radians_to_degrees(math.pi / 2)) == 90.0

    def test_bidirectional_angle_conversion(self) -> None:
        """Test that angle conversions are reversible."""
        original = 123.456  # degrees
        converted = units.degrees_to_radians(original)
        back = units.radians_to_degrees(converted)
        assert pytest.approx(back, rel=1e-10) == original


class TestFroudeNumber:
    """Tests for Froude number calculation."""

    def test_froude_number_typical(self) -> None:
        """Test Froude number for typical ship."""
        # 15 knots (7.72 m/s) at 150m length
        fn = units.calculate_froude_number(7.72, 150.0)
        assert 0.20 < fn < 0.21  # Typical displacement mode

    def test_froude_number_slow_ship(self) -> None:
        """Test low Froude number."""
        fn = units.calculate_froude_number(5.0, 200.0)
        assert fn < 0.15  # Very slow

    def test_froude_number_fast_ship(self) -> None:
        """Test high Froude number."""
        fn = units.calculate_froude_number(15.0, 100.0)
        assert fn > 0.4  # Fast vessel

    def test_froude_number_negative_speed(self) -> None:
        """Test that negative speed raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            units.calculate_froude_number(-5.0, 100.0)

    def test_froude_number_zero_length(self) -> None:
        """Test that zero length raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            units.calculate_froude_number(10.0, 0.0)

    def test_froude_number_negative_length(self) -> None:
        """Test that negative length raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            units.calculate_froude_number(10.0, -100.0)


class TestReynoldsNumber:
    """Tests for Reynolds number calculation."""

    def test_reynolds_number_typical(self) -> None:
        """Test Reynolds number for typical ship."""
        # 15 knots (7.72 m/s) at 150m length
        re = units.calculate_reynolds_number(7.72, 150.0)
        assert 9e8 < re < 1e9  # Typical range for medium vessel

    def test_reynolds_number_always_turbulent(self) -> None:
        """Test that ship flows are always turbulent (Re > 5e5)."""
        # Even slow, small vessel
        re = units.calculate_reynolds_number(3.0, 50.0)
        assert re > 5e5  # Turbulent flow

    def test_reynolds_number_large_vessel(self) -> None:
        """Test Reynolds number for large vessel."""
        re = units.calculate_reynolds_number(10.0, 300.0)
        assert re > 1e9  # Very high Reynolds number

    def test_reynolds_number_negative_speed(self) -> None:
        """Test that negative speed raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            units.calculate_reynolds_number(-5.0, 100.0)

    def test_reynolds_number_zero_length(self) -> None:
        """Test that zero length raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            units.calculate_reynolds_number(10.0, 0.0)

    def test_reynolds_number_zero_viscosity(self) -> None:
        """Test that zero viscosity raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            units.calculate_reynolds_number(10.0, 100.0, 0.0)


class TestAngleNormalization:
    """Tests for angle normalization."""

    def test_normalize_positive_angle(self) -> None:
        """Test normalization of positive angles."""
        assert units.normalize_angle(45.0) == 45.0
        assert units.normalize_angle(180.0) == 180.0
        assert units.normalize_angle(359.0) == 359.0

    def test_normalize_angle_above_360(self) -> None:
        """Test normalization of angles > 360."""
        assert units.normalize_angle(450.0) == 90.0
        assert units.normalize_angle(720.0) == 0.0
        assert pytest.approx(units.normalize_angle(361.5)) == 1.5

    def test_normalize_negative_angle(self) -> None:
        """Test normalization of negative angles."""
        assert units.normalize_angle(-45.0) == 315.0
        assert units.normalize_angle(-90.0) == 270.0
        assert units.normalize_angle(-180.0) == 180.0

    def test_normalize_large_negative_angle(self) -> None:
        """Test normalization of large negative angles."""
        assert units.normalize_angle(-450.0) == 270.0


class TestRelativeAngle:
    """Tests for relative angle calculation."""

    def test_relative_angle_same_direction(self) -> None:
        """Test relative angle when heading and direction are same."""
        assert units.relative_angle(0.0, 0.0) == 0.0
        assert units.relative_angle(90.0, 90.0) == 0.0

    def test_relative_angle_opposite_direction(self) -> None:
        """Test relative angle for opposite directions."""
        assert units.relative_angle(0.0, 180.0) == 180.0
        assert units.relative_angle(90.0, 270.0) == 180.0

    def test_relative_angle_perpendicular(self) -> None:
        """Test relative angle for perpendicular directions."""
        assert units.relative_angle(0.0, 90.0) == 90.0
        assert units.relative_angle(0.0, 270.0) == 90.0

    def test_relative_angle_acute(self) -> None:
        """Test that relative angle is always acute (≤180)."""
        assert units.relative_angle(0.0, 45.0) == 45.0
        assert units.relative_angle(0.0, 315.0) == 45.0

    def test_relative_angle_symmetry(self) -> None:
        """Test that relative angle is symmetric."""
        assert units.relative_angle(30.0, 60.0) == units.relative_angle(60.0, 30.0)
        assert units.relative_angle(180.0, 270.0) == units.relative_angle(270.0, 180.0)

    def test_relative_angle_range(self) -> None:
        """Test that relative angle is always in [0, 180]."""
        for heading in [0, 45, 90, 135, 180, 225, 270, 315]:
            for direction in [0, 45, 90, 135, 180, 225, 270, 315]:
                rel_angle = units.relative_angle(heading, direction)
                assert 0 <= rel_angle <= 180
