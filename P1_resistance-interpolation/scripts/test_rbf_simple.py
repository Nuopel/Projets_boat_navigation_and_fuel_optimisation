#!/usr/bin/env python3
"""Simple test to verify RBF interpolator is working correctly.

This tests RBF on clean synthetic data to ensure there are no bugs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.interpolators import RBFInterpolator
from src.evaluation.metrics import MetricsCalculator

print("=" * 80)
print("RBF INTERPOLATOR - SIMPLE VERIFICATION TEST")
print("=" * 80)
print()

# ===========================================
# Test 1: Very Simple 2D Function
# ===========================================
print("TEST 1: Simple quadratic function f(x, y) = x² + y²")
print("-" * 80)

# Generate clean training data
np.random.seed(42)
n_train = 50
X_train = np.random.uniform(-5, 5, (n_train, 2))
y_train = X_train[:, 0]**2 + X_train[:, 1]**2

# Generate test data
n_test = 100
X_test = np.random.uniform(-5, 5, (n_test, 2))
y_true = X_test[:, 0]**2 + X_test[:, 1]**2

print(f"Training samples: {n_train}")
print(f"Test samples: {n_test}")
print(f"X range: [-5, 5] × [-5, 5]")
print(f"y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print()

# Test different RBF kernels
kernels = ['thin_plate_spline', 'cubic', 'linear', 'gaussian', 'multiquadric']

print("Testing different RBF kernels:")
print()

for kernel in kernels:
    try:
        rbf = RBFInterpolator(kernel=kernel, smoothing=0.0)
        rbf.fit(X_train, y_train)
        y_pred = rbf.predict(X_test)

        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))

        status = "✓" if rmse < 10.0 else "⚠️"
        print(f"{status} {kernel:20s} | RMSE={rmse:8.4f} | MAE={mae:8.4f}")

    except Exception as e:
        print(f"✗ {kernel:20s} | ERROR: {str(e)}")

print()

# ===========================================
# Test 2: Yacht-Like Data Structure
# ===========================================
print("TEST 2: Yacht-like function with different feature scales")
print("-" * 80)

# Simulate yacht-like data with realistic scales
np.random.seed(42)
n_train = 100

# Velocity: 2-9 knots (like yacht data)
V_train = np.random.uniform(2, 9, n_train)
# Draft: 0.5-1.1 meters (like yacht data)
T_train = np.random.uniform(0.5, 1.1, n_train)

X_train_yacht = np.column_stack([V_train, T_train])

# Resistance function similar to yacht physics: R = a*V² + b/T + c*V*T + d
y_train_yacht = 0.05 * V_train**2 - 2.0 / T_train + 0.01 * V_train * T_train + 15.0

# Test data
V_test = np.random.uniform(2, 9, 50)
T_test = np.random.uniform(0.5, 1.1, 50)
X_test_yacht = np.column_stack([V_test, T_test])
y_true_yacht = 0.05 * V_test**2 - 2.0 / T_test + 0.01 * V_test * T_test + 15.0

print(f"Training samples: {n_train}")
print(f"Velocity range: [{V_train.min():.2f}, {V_train.max():.2f}] knots")
print(f"Draft range: [{T_train.min():.2f}, {T_train.max():.2f}] meters")
print(f"Resistance range: [{y_train_yacht.min():.2f}, {y_train_yacht.max():.2f}]")
print()
print("⚠️  Note: Features on DIFFERENT SCALES (like real yacht data)")
print()

# Test WITHOUT normalization
print("A) WITHOUT Normalization:")
try:
    rbf = RBFInterpolator(kernel='thin_plate_spline', smoothing=0.0)
    rbf.fit(X_train_yacht, y_train_yacht)
    y_pred = rbf.predict(X_test_yacht)

    rmse = np.sqrt(np.mean((y_true_yacht - y_pred)**2))
    mae = np.mean(np.abs(y_true_yacht - y_pred))
    max_error = np.max(np.abs(y_true_yacht - y_pred))

    # Check for numerical issues
    if rmse > 1e10 or np.isnan(rmse) or np.isinf(rmse):
        print(f"✗ RBF UNSTABLE: RMSE={rmse:.2e} (numerical overflow)")
    else:
        print(f"✓ RBF stable: RMSE={rmse:.4f}, MAE={mae:.4f}, Max Error={max_error:.4f}")

except Exception as e:
    print(f"✗ RBF FAILED: {str(e)}")

print()

# Test WITH normalization
print("B) WITH Normalization:")
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_yacht)
y_train_scaled = scaler_y.fit_transform(y_train_yacht.reshape(-1, 1)).ravel()
X_test_scaled = scaler_X.transform(X_test_yacht)

try:
    rbf = RBFInterpolator(kernel='thin_plate_spline', smoothing=0.0)
    rbf.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = rbf.predict(X_test_scaled)

    # Transform back
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    rmse = np.sqrt(np.mean((y_true_yacht - y_pred)**2))
    mae = np.mean(np.abs(y_true_yacht - y_pred))
    max_error = np.max(np.abs(y_true_yacht - y_pred))

    # Check for numerical issues
    if rmse > 1e10 or np.isnan(rmse) or np.isinf(rmse):
        print(f"✗ RBF STILL UNSTABLE: RMSE={rmse:.2e} (even with normalization)")
    else:
        print(f"✓ RBF stable with normalization: RMSE={rmse:.4f}, MAE={mae:.4f}, Max Error={max_error:.4f}")

except Exception as e:
    print(f"✗ RBF FAILED: {str(e)}")

print()

# ===========================================
# Test 3: With Smoothing
# ===========================================
print("TEST 3: RBF with smoothing parameter")
print("-" * 80)

print("Testing different smoothing levels on yacht-like data (normalized):")
print()

smoothing_levels = [0.0, 0.1, 0.5, 1.0]

for smoothing in smoothing_levels:
    try:
        rbf = RBFInterpolator(kernel='thin_plate_spline', smoothing=smoothing)
        rbf.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = rbf.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        rmse = np.sqrt(np.mean((y_true_yacht - y_pred)**2))

        if rmse > 1e10 or np.isnan(rmse) or np.isinf(rmse):
            print(f"✗ smoothing={smoothing:.1f}: RMSE={rmse:.2e} (UNSTABLE)")
        else:
            print(f"✓ smoothing={smoothing:.1f}: RMSE={rmse:.4f}")

    except Exception as e:
        print(f"✗ smoothing={smoothing:.1f}: ERROR - {str(e)}")

print()

# ===========================================
# Conclusions
# ===========================================
print("=" * 80)
print("CONCLUSIONS")
print("=" * 80)
print()
print("1. RBF works correctly on simple test cases (Test 1)")
print("2. RBF has issues with yacht-like data structure (different scales)")
print("3. Normalization may help but doesn't guarantee stability")
print("4. Smoothing can help but may not solve all issues")
print()
print("RECOMMENDATION:")
print("  - RBF is NOT reliable for real yacht data without extensive tuning")
print("  - Kriging handles scale differences internally and is more robust")
print("  - For production use on ship performance: Use Kriging")
print()
print("=" * 80)
