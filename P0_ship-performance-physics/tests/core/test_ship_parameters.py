"""Tests for ship parameters dataclass."""

import warnings

import pytest

from ship_performance.core import InvalidShipParametersError, ShipParameters


class TestShipParametersConstruction:
    """Tests for creating ShipParameters instances."""

    def test_create_typical_cargo_ship(self) -> None:
        """Test creating a typical cargo ship."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )

        assert ship.length == 150
        assert ship.beam == 25
        assert ship.draft == 8
        assert ship.displacement == 15000
        assert ship.block_coefficient == 0.7
        assert ship.wetted_surface is not None  # Should be estimated
        assert ship.frontal_area is not None  # Should be estimated

    def test_create_with_all_parameters(self) -> None:
        """Test creating ship with all parameters provided."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
            wetted_surface=4000,
            frontal_area=600,
        )

        assert ship.wetted_surface == 4000
        assert ship.frontal_area == 600

    def test_create_tanker(self) -> None:
        """Test creating a typical tanker."""
        ship = ShipParameters(
            length=250,
            beam=42,
            draft=14,
            displacement=80000,
            block_coefficient=0.82,
        )

        assert ship.length == 250
        assert ship.displacement == 80000
        assert ship.block_coefficient == 0.82


class TestShipParametersValidation:
    """Tests for ship parameter validation."""

    def test_negative_length(self) -> None:
        """Test that negative length raises error."""
        with pytest.raises(InvalidShipParametersError, match="length must be positive"):
            ShipParameters(
                length=-150,
                beam=25,
                draft=8,
                displacement=15000,
                block_coefficient=0.7,
            )

    def test_zero_length(self) -> None:
        """Test that zero length raises error."""
        with pytest.raises(InvalidShipParametersError, match="length must be positive"):
            ShipParameters(
                length=0,
                beam=25,
                draft=8,
                displacement=15000,
                block_coefficient=0.7,
            )

    def test_negative_beam(self) -> None:
        """Test that negative beam raises error."""
        with pytest.raises(InvalidShipParametersError, match="beam must be positive"):
            ShipParameters(
                length=150,
                beam=-25,
                draft=8,
                displacement=15000,
                block_coefficient=0.7,
            )

    def test_negative_draft(self) -> None:
        """Test that negative draft raises error."""
        with pytest.raises(InvalidShipParametersError, match="draft must be positive"):
            ShipParameters(
                length=150,
                beam=25,
                draft=-8,
                displacement=15000,
                block_coefficient=0.7,
            )

    def test_negative_displacement(self) -> None:
        """Test that negative displacement raises error."""
        with pytest.raises(
            InvalidShipParametersError, match="displacement must be positive"
        ):
            ShipParameters(
                length=150,
                beam=25,
                draft=8,
                displacement=-15000,
                block_coefficient=0.7,
            )

    def test_block_coefficient_too_low(self) -> None:
        """Test that Cb < 0.4 raises error."""
        with pytest.raises(
            InvalidShipParametersError, match="Block coefficient must be between"
        ):
            ShipParameters(
                length=150,
                beam=25,
                draft=8,
                displacement=15000,
                block_coefficient=0.3,
            )

    def test_block_coefficient_too_high(self) -> None:
        """Test that Cb > 0.9 raises error."""
        with pytest.raises(
            InvalidShipParametersError, match="Block coefficient must be between"
        ):
            ShipParameters(
                length=150,
                beam=25,
                draft=8,
                displacement=15000,
                block_coefficient=0.95,
            )

    def test_negative_wetted_surface(self) -> None:
        """Test that negative wetted surface raises error."""
        with pytest.raises(
            InvalidShipParametersError, match="Wetted surface must be positive"
        ):
            ShipParameters(
                length=150,
                beam=25,
                draft=8,
                displacement=15000,
                block_coefficient=0.7,
                wetted_surface=-4000,
            )

    def test_negative_frontal_area(self) -> None:
        """Test that negative frontal area raises error."""
        with pytest.raises(
            InvalidShipParametersError, match="Frontal area must be positive"
        ):
            ShipParameters(
                length=150,
                beam=25,
                draft=8,
                displacement=15000,
                block_coefficient=0.7,
                frontal_area=-600,
            )


class TestShipParametersRatioWarnings:
    """Tests for warnings on unusual dimensional ratios."""

    def test_low_length_beam_ratio_warning(self) -> None:
        """Test warning for unusually low L/B ratio."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ShipParameters(
                length=100,
                beam=30,  # L/B = 3.33, below typical 5-10
                draft=8,
                displacement=15000,
                block_coefficient=0.7,
            )
            assert len(w) >= 1
            assert "L/B ratio" in str(w[0].message)

    def test_high_length_beam_ratio_warning(self) -> None:
        """Test warning for unusually high L/B ratio."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ShipParameters(
                length=250,
                beam=20,  # L/B = 12.5, above typical 5-10
                draft=8,
                displacement=15000,
                block_coefficient=0.7,
            )
            assert len(w) >= 1
            assert "L/B ratio" in str(w[0].message)

    def test_low_beam_draft_ratio_warning(self) -> None:
        """Test warning for unusually low B/T ratio."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ShipParameters(
                length=150,
                beam=25,
                draft=20,  # B/T = 1.25, below typical 2-4
                displacement=15000,
                block_coefficient=0.7,
            )
            assert len(w) >= 1
            assert "B/T ratio" in str(w[0].message)


class TestDerivedProperties:
    """Tests for computed properties."""

    def test_length_beam_ratio(self) -> None:
        """Test L/B ratio calculation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        assert ship.length_beam_ratio == 6.0

    def test_beam_draft_ratio(self) -> None:
        """Test B/T ratio calculation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        assert pytest.approx(ship.beam_draft_ratio) == 3.125

    def test_length_draft_ratio(self) -> None:
        """Test L/T ratio calculation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        assert pytest.approx(ship.length_draft_ratio) == 18.75

    def test_volumetric_displacement(self) -> None:
        """Test volumetric displacement calculation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        # 15000 tonnes / 1.025 t/m³ ≈ 14634 m³
        assert pytest.approx(ship.volumetric_displacement, rel=1e-3) == 14634.146

    def test_prismatic_coefficient(self) -> None:
        """Test prismatic coefficient estimation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        # Cp ≈ Cb / 0.99 for typical ships
        assert 0.69 < ship.prismatic_coefficient < 0.72

    def test_waterplane_coefficient(self) -> None:
        """Test waterplane coefficient estimation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        # Cwp ≈ Cb + 0.12 for typical ships
        assert 0.8 < ship.waterplane_coefficient < 0.85


class TestEstimations:
    """Tests for wetted surface and frontal area estimations."""

    def test_wetted_surface_estimated(self) -> None:
        """Test that wetted surface is estimated when not provided."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        assert ship.wetted_surface is not None
        assert ship.wetted_surface > 0
        # Typical wetted surface for this size: 2500-4000 m²
        assert 2500 < ship.wetted_surface < 4000

    def test_frontal_area_estimated(self) -> None:
        """Test that frontal area is estimated when not provided."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        assert ship.frontal_area is not None
        assert ship.frontal_area > 0
        # Typical frontal area: B × (0.5-1.0 × T)
        assert 100 < ship.frontal_area < 250

    def test_provided_values_not_overwritten(self) -> None:
        """Test that provided wetted surface and frontal area are used."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
            wetted_surface=4000,
            frontal_area=600,
        )
        assert ship.wetted_surface == 4000
        assert ship.frontal_area == 600


class TestImmutability:
    """Tests that ShipParameters is immutable (frozen dataclass)."""

    def test_cannot_modify_length(self) -> None:
        """Test that length cannot be modified after creation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        with pytest.raises((AttributeError, TypeError)):
            ship.length = 200  # type: ignore

    def test_cannot_modify_beam(self) -> None:
        """Test that beam cannot be modified after creation."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        with pytest.raises((AttributeError, TypeError)):
            ship.beam = 30  # type: ignore


class TestStringRepresentation:
    """Tests for string representation."""

    def test_repr_contains_key_info(self) -> None:
        """Test that __repr__ contains essential ship info."""
        ship = ShipParameters(
            length=150,
            beam=25,
            draft=8,
            displacement=15000,
            block_coefficient=0.7,
        )
        repr_str = repr(ship)

        assert "150" in repr_str  # length
        assert "25" in repr_str  # beam
        assert "0.7" in repr_str or "0.700" in repr_str  # Cb
        assert "ShipParameters" in repr_str
