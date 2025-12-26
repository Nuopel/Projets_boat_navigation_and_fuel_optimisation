"""Interpolation methods for ship performance prediction.

This package provides three interpolation methods:
- RBF: Radial Basis Functions (fast, flexible)
- Spline: Bivariate splines (very fast, smooth)
- Kriging: Gaussian Process Regression (uncertainty quantification)

All interpolators follow the BaseInterpolator API with fit() and predict() methods.

Example:
    >>> from src.interpolators import RBFInterpolator, SplineInterpolator, KrigingInterpolator
    >>> # Create interpolator
    >>> rbf = RBFInterpolator(kernel='thin_plate_spline')
    >>> # Train
    >>> rbf.fit(X_train, y_train)
    >>> # Predict
    >>> predictions = rbf.predict(X_test)
"""

from .base import BaseInterpolator
from .rbf import RBFInterpolator
from .spline import SplineInterpolator
from .kriging import KrigingInterpolator

__all__ = [
    'BaseInterpolator',
    'RBFInterpolator',
    'SplineInterpolator',
    'KrigingInterpolator'
]
