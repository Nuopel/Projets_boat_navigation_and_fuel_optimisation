"""Ship hull and geometric parameters.

This module defines the ShipParameters dataclass which encapsulates all
geometric and physical properties of a ship hull.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

from ..utils.constants import (
    SEAWATER_DENSITY,
    TYPICAL_BEAM_DRAFT_RATIO_MAX,
    TYPICAL_BEAM_DRAFT_RATIO_MIN,
    TYPICAL_BLOCK_COEFFICIENT_MAX,
    TYPICAL_BLOCK_COEFFICIENT_MIN,
    TYPICAL_LENGTH_BEAM_RATIO_MAX,
    TYPICAL_LENGTH_BEAM_RATIO_MIN,
)


class InvalidShipParametersError(ValueError):
    """Raised when ship parameters are invalid or out of realistic ranges."""

    pass


@dataclass(frozen=True)
class ShipParameters:
    """Ship hull and geometric parameters.

    All dimensions are in SI units (meters, tonnes).

    Attributes:
        length: Length overall (m), must be > 0
        beam: Beam at waterline (m), must be > 0
        draft: Draft at design condition (m), must be > 0
        displacement: Displacement (tonnes), must be > 0
        block_coefficient: Block coefficient Cb (dimensionless), range: 0.4-0.9
        wetted_surface: Wetted surface area (m²), optional (will be estimated)
        frontal_area: Frontal windage area (m²), optional (will be estimated)

    Raises:
        InvalidShipParametersError: If parameters are invalid or out of realistic ranges

    Example:
        >>> ship = ShipParameters(
        ...     length=150, beam=25, draft=8,
        ...     displacement=15000, block_coefficient=0.7
        ... )
        >>> ship.length_beam_ratio
        6.0
        >>> ship.volumetric_displacement
        14634.146341463415
    """

    length: float  # m
    beam: float  # m
    draft: float  # m
    displacement: float  # tonnes
    block_coefficient: float  # dimensionless
    wetted_surface: Optional[float] = None  # m²
    frontal_area: Optional[float] = None  # m²

    def __post_init__(self) -> None:
        """Validate parameters and compute derived values."""
        self._validate_basic_parameters()
        self._validate_ratios()

        # Estimate missing values
        if self.wetted_surface is None:
            estimated_wetted_surface = self._estimate_wetted_surface()
            # Use object.__setattr__ since dataclass is frozen
            object.__setattr__(self, "wetted_surface", estimated_wetted_surface)

        if self.frontal_area is None:
            estimated_frontal_area = self._estimate_frontal_area()
            object.__setattr__(self, "frontal_area", estimated_frontal_area)

    def _validate_basic_parameters(self) -> None:
        """Validate basic parameter ranges."""
        if self.length <= 0:
            raise InvalidShipParametersError(
                f"Ship length must be positive, got {self.length} m"
            )

        if self.beam <= 0:
            raise InvalidShipParametersError(
                f"Ship beam must be positive, got {self.beam} m"
            )

        if self.draft <= 0:
            raise InvalidShipParametersError(
                f"Ship draft must be positive, got {self.draft} m"
            )

        if self.displacement <= 0:
            raise InvalidShipParametersError(
                f"Ship displacement must be positive, got {self.displacement} tonnes"
            )

        if not (TYPICAL_BLOCK_COEFFICIENT_MIN <= self.block_coefficient <= TYPICAL_BLOCK_COEFFICIENT_MAX):
            raise InvalidShipParametersError(
                f"Block coefficient must be between {TYPICAL_BLOCK_COEFFICIENT_MIN} "
                f"and {TYPICAL_BLOCK_COEFFICIENT_MAX}, got {self.block_coefficient}"
            )

        if self.wetted_surface is not None and self.wetted_surface <= 0:
            raise InvalidShipParametersError(
                f"Wetted surface must be positive if provided, got {self.wetted_surface} m²"
            )

        if self.frontal_area is not None and self.frontal_area <= 0:
            raise InvalidShipParametersError(
                f"Frontal area must be positive if provided, got {self.frontal_area} m²"
            )

    def _validate_ratios(self) -> None:
        """Validate dimensional ratios and issue warnings for unusual values."""
        lb_ratio = self.length_beam_ratio
        if not (TYPICAL_LENGTH_BEAM_RATIO_MIN <= lb_ratio <= TYPICAL_LENGTH_BEAM_RATIO_MAX):
            warnings.warn(
                f"L/B ratio ({lb_ratio:.2f}) is outside typical range "
                f"[{TYPICAL_LENGTH_BEAM_RATIO_MIN}, {TYPICAL_LENGTH_BEAM_RATIO_MAX}]. "
                f"This may indicate an unusual vessel type or data error.",
                UserWarning,
            )

        bt_ratio = self.beam_draft_ratio
        if not (TYPICAL_BEAM_DRAFT_RATIO_MIN <= bt_ratio <= TYPICAL_BEAM_DRAFT_RATIO_MAX):
            warnings.warn(
                f"B/T ratio ({bt_ratio:.2f}) is outside typical range "
                f"[{TYPICAL_BEAM_DRAFT_RATIO_MIN}, {TYPICAL_BEAM_DRAFT_RATIO_MAX}]. "
                f"This may indicate an unusual vessel type or data error.",
                UserWarning,
            )

        # Volumetric consistency check
        volume_from_cb = self.length * self.beam * self.draft * self.block_coefficient
        volume_from_displacement = self.volumetric_displacement

        volume_ratio = volume_from_cb / volume_from_displacement
        if not (0.85 <= volume_ratio <= 1.15):
            warnings.warn(
                f"Volumetric displacement inconsistency detected: "
                f"L×B×T×Cb = {volume_from_cb:.1f} m³, "
                f"displacement/ρ = {volume_from_displacement:.1f} m³ "
                f"(ratio: {volume_ratio:.2f}). This may indicate inconsistent parameters.",
                UserWarning,
            )

    def _estimate_wetted_surface(self) -> float:
        """Estimate wetted surface area using Schneekluth formula.

        S ≈ 1.7 * L * T + ∇^(2/3)

        where ∇ is volumetric displacement.

        Returns:
            Estimated wetted surface area in m²
        """
        volumetric_disp = self.volumetric_displacement
        # Schneekluth formula (simplified)
        surface = 1.7 * self.length * self.draft + volumetric_disp ** (2 / 3)
        return surface

    def _estimate_frontal_area(self) -> float:
        """Estimate frontal windage area.

        A_frontal ≈ B × (H_above_water)

        where H_above_water is estimated as a fraction of draft
        (typical ships have ~0.5-1.0 times draft above waterline).

        Returns:
            Estimated frontal windage area in m²
        """
        # Estimate height above water as 0.7 * draft (typical for cargo ships)
        height_above_water = 0.7 * self.draft
        frontal_area = self.beam * height_above_water
        return frontal_area

    @property
    def length_beam_ratio(self) -> float:
        """Calculate length-to-beam ratio (L/B).

        Typical values:
        - Cargo ships: 6-8
        - Tankers: 6-7
        - Container ships: 7-9
        - Fast vessels: 8-12

        Returns:
            L/B ratio (dimensionless)
        """
        return self.length / self.beam

    @property
    def beam_draft_ratio(self) -> float:
        """Calculate beam-to-draft ratio (B/T).

        Typical values:
        - Cargo ships: 2.5-3.5
        - Tankers: 2.0-2.5
        - Container ships: 3.0-4.0

        Returns:
            B/T ratio (dimensionless)
        """
        return self.beam / self.draft

    @property
    def length_draft_ratio(self) -> float:
        """Calculate length-to-draft ratio (L/T).

        Returns:
            L/T ratio (dimensionless)
        """
        return self.length / self.draft

    @property
    def volumetric_displacement(self) -> float:
        """Calculate volumetric displacement (∇).

        ∇ = Δ / ρ_seawater

        where:
        - Δ is mass displacement (tonnes)
        - ρ_seawater ≈ 1.025 t/m³

        Returns:
            Volumetric displacement in m³
        """
        return self.displacement / (SEAWATER_DENSITY / 1000.0)  # Convert to t/m³

    @property
    def prismatic_coefficient(self) -> float:
        """Calculate prismatic coefficient (Cp).

        Cp = ∇ / (A_midship × L)
        where A_midship = B × T × Cm (midship coefficient)

        For typical ships: Cp ≈ Cb / Cm, and Cm ≈ 0.98-1.0
        Simplified: Cp ≈ Cb / 0.99

        Returns:
            Estimated prismatic coefficient (dimensionless)
        """
        # Simplified estimation: Cp ≈ Cb / Cm, assuming Cm ≈ 0.99
        return self.block_coefficient / 0.99

    @property
    def waterplane_coefficient(self) -> float:
        """Estimate waterplane area coefficient (Cwp).

        Cwp = A_waterplane / (L × B)

        Typical relationship: Cwp ≈ Cb + 0.1 to 0.15

        Returns:
            Estimated waterplane coefficient (dimensionless)
        """
        # Empirical relationship
        return min(0.95, self.block_coefficient + 0.12)

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"ShipParameters("
            f"length={self.length:.1f}m, "
            f"beam={self.beam:.1f}m, "
            f"draft={self.draft:.1f}m, "
            f"displacement={self.displacement:.0f}t, "
            f"Cb={self.block_coefficient:.3f}, "
            f"L/B={self.length_beam_ratio:.2f}"
            f")"
        )
