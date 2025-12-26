# Ship navigation and fuel optimization projects

Four short sprints covering ship performance modeling, interpolation, routing, and hybrid ML. Each was run with a WBS/MVP plan,  AI help for planning/writing, and hands-on coding for the core models and evaluation around (3 day).

## How to read this

- Skim the project snapshots for focus, highlights, and limits.
- For a demo, open P2 (routing dashboard). For modeling depth, see P0 and P3.
- Want the takeaway? Jump to "What this shows" and "Reality checks".

## Project snapshots

### P0 — Physics-based ship performance model

- Focus: calm-water, wind, and wave resistance; power, fuel, and CO2 from L, B, T, Cb, displacement, speed, wind, and waves.
- Highlights: ITTC 1957 friction + form factor with a simplified wave hump; power scales ~V^3 with reasonable resistance breakdowns for merchant displacement hulls (Fn ~0.05-0.35).
- Limits: not a full Holtrop-Mennen; not for planing/high-speed craft; aero and sea effects simplified; steady-state only.

### P1 — Resistance interpolation from sparse data

- Focus: turn sparse UCI Yacht Hydrodynamics points into a continuous surface via Kriging, RBF, and splines after aggregating duplicate (V, T).
- Highlights: Kriging delivers ~0.99 R²; RBF is usable but less stable; shows how aggregation makes the 2D problem well-posed.
- Limits: 2D surface omits hull geometry and other drivers; aggregation hides inter-hull variability, so this is a method demo, not a full predictor.

### P2 — Multi-objective route optimization under weather

- Focus: minimize time, fuel, and CO2 with progressive MVPs (A* with weather penalties, then NSGA-II on speed, plus a Streamlit dashboard).
- Highlights: demonstrates time–fuel trade-offs and Pareto fronts on synthetic weather; dashboard to explore scenarios and constraint effects.
- Limits: calibration on real data failed (heterogeneous); A* uses soft weather penalties; NSGA-II fixes the route shape; weather is synthetic.

### P3 — Hybrid physics + ML fuel prediction

- Focus: physics prior + ML residuals on 1,440 operational rows (120 vessels) with uncertainty quantification.
- Highlights: ridge and XGBoost reach ~0.95 R²; hybrids are competitive but do not beat ridge; clear distance–fuel signal and domain-driven features.
- Limits: small regional dataset; physics prior is coarse (no hull/speed profile/sea state); no cross-validation; UQ intervals under-cover (PICP < 0.85).

## What this shows

- Models: physics-based resistance/power, interpolation on sparse data, hybrid ML with uncertainty, and multi-objective optimization.
- Workflow: WBS/MVP planning, quick iterations, explicit assumptions and limits, reproducible scripts.
- Domain: naval hydrodynamics awareness (Fn ranges, wind/wave effects) and operational trade-offs (time vs fuel vs CO2).

## Reality checks

- Built fast by design: expect rough edges and simplifications.
- P2 uses synthetic weather; P0/P3 simplify physics; P1/P3 have limited generalization due to dataset scope.
- Each README and `PROJECT*_OVERVIEW_LIMITS.md` spells out the assumptions, validity ranges, and gaps.
