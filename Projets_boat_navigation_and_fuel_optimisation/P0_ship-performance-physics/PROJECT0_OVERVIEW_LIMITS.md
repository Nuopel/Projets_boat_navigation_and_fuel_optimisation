# Project Overview, Assumptions, Validity, and Limits

This project builds a physics-based ship performance model that estimates total resistance, required power, fuel consumption, and emissions from basic ship geometry and operating conditions. It integrates calm-water resistance, wind resistance, added wave resistance, propulsion efficiency, and fuel consumption into a single prediction workflow. The examples demonstrate behavior across typical merchant ship sizes and speeds.

## What it does (and why)
- Inputs: ship geometry (L, B, T, displacement, Cb), speed, wind, and waves.
- Outputs: resistance breakdown (calm + wind + waves), effective/delivered/brake power, fuel rate, and CO2.
- Goal: provide a consistent, transparent model for trend analysis, scenario comparison, and educational demos.

## Core hypotheses and domain of validity
1) Ship parameters and hydrostatics
- Hypothesis: L, B, T, Cb, and displacement represent the same loading condition.
- Validity: displacement hulls; consistent geometry (L*B*T*Cb ~ displacement/ρ).
- Limit: if inputs are inconsistent (e.g., LOA vs LWL, lightship vs full load), results skew.

2) Calm-water resistance
- Hypothesis: ITTC 1957 friction formula with form factor and simplified wave-making.
- Validity: merchant displacement hulls, Fn ~ 0.05-0.35; fully turbulent flow (Re >> 1e5).
- Limit: not a full Holtrop-Mennen implementation; wave-making term is a simplified hump shape and is not accurate for high-speed or planing regimes (Fn > ~0.4).

3) Wind resistance (windage)
- Hypothesis: aerodynamic drag with empirical Cd and frontal area estimate.
- Validity: moderate wind, conventional superstructure shapes.
- Limit: Cd is approximate; actual exposed area varies with load and deck configuration.

4) Added resistance in waves
- Hypothesis: simplified Kwon-type formula with head-sea factor.
- Validity: regular sea state descriptors (Hs, Tp) and displacement hulls.
- Limit: does not capture full seakeeping dynamics, directional spreading, or resonance effects.

5) Propulsion and fuel
- Hypothesis: fixed efficiencies (propeller, shaft) and SFOC by fuel type with a nominal load factor.
- Validity: typical slow-speed diesel merchant ships at steady cruise.
- Limit: real engines vary with RPM, sea margin, fouling, and routing; transient behavior ignored.

## Do the results make sense?
- After aligning displacement with geometry and adjusting wave-making, results show non-zero wave resistance and a reasonable hump trend across Fn ~ 0.2-0.35.
- Power scales roughly with V^3 and fuel rises steeply with speed, matching standard marine behavior.
- Wind and wave contributions are modest in moderate seas and grow in storms, consistent with engineering expectations.

## Practical limits
- Not suitable for planing craft, high-speed vessels, or highly unconventional hulls.
- Not a replacement for detailed resistance tests, CFD, or validated Holtrop-Mennen calculations.
- Best used for trend analysis, comparison, and education, not absolute contractual performance.
