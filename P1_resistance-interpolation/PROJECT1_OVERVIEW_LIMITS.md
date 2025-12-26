# Project Summary

This project studies how to predict ship resistance from sparse experimental data, a common problem because towing-tank and sea-trial measurements are expensive and limited. The goal is to build a continuous performance surface so engineers can estimate resistance at unmeasured operating points and compare interpolation methods used in practice.

## Why

Real ships are tested at only a few speeds and drafts. Engineers need a way to estimate resistance between those points for design and operational decisions.

## How

The project uses the UCI Yacht Hydrodynamics dataset (308 measurements). It derives proxy physical inputs:

- V (speed) from Froude number
- T (draft) from beam-to-draft ratio

Because many different hulls share the same (V, T) but have different resistance, the 2D mapping is not unique. To make interpolation well-posed, duplicate (V, T) points are aggregated by averaging resistance.

Three methods are implemented with a common API and benchmarked:

- Kriging (Gaussian process) with uncertainty estimation
- RBF interpolation
- Bivariate splines

Benchmarks include convergence with increasing training size and robustness to noise, with results exported to CSV.

## What (Results)

- Kriging is consistently the most reliable and accurate. Typical R2 is around 0.99, and RMSE is around 2 on a resistance range of ~0-62.
- RBF now works after aggregation, but is less stable and less accurate than Kriging on average.
- Splines are highly unstable on irregular scattered data and often fail dramatically.

## Limits and Hypotheses

- The 2D surface assumes resistance depends only on V and T, which is not physically complete. Hull geometry and other dimensionless parameters matter; aggregating duplicates hides these effects.
- The analysis is therefore a method demonstration, not a full physical model.
- For real predictive use, the model should include all original features or be built per hull class.

## Bottom Line

Interpolation can turn sparse ship data into a usable surface, and Kriging is the most dependable method for this dataset once the 2D surface is made well-posed.
