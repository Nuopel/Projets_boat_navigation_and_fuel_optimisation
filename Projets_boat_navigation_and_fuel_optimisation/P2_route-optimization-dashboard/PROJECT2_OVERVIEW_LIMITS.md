# Project Summary: Multi-Objective Ship Route Optimization

## Context

This project explores maritime route planning under weather constraints with competing objectives: minimize voyage time, fuel consumption, and CO2 emissions. It is structured as progressive MVPs to validate the simulation foundation, optimization approaches, and user-facing exploration tools.

## Aim

- Build a reproducible optimization pipeline for ship routing under synthetic weather.
- Quantify trade-offs between time, fuel, and emissions.
- Provide tools (scripts + dashboard) to explore Pareto-optimal solutions.

## Results

- **MVP-1 (Foundation)**: Implemented ship fuel model, synthetic weather fields, navigation grid, and tests. Calibration to real data failed due to dataset heterogeneity, so validated manual coefficients are used.
- **MVP-2 (Single-objective optimizer)**: A* route planning with weather penalties, route evaluation, and constraint checking. Demonstrates time–fuel trade-offs and time-window effects. Storm scenario highlights constraint violations and planner limitations.
- **MVP-3 (Multi-objective optimizer)**: NSGA-II generates a Pareto front (time, fuel, CO2) for constant-speed solutions on a fixed route. Weighted-sum baseline provides a narrower coverage.
- **MVP-4 (Dashboard)**: Streamlit UI to configure scenarios, run optimizers, and explore Pareto fronts interactively.

## Hypotheses

- A simplified cubic speed–fuel model with weather penalties is sufficient to demonstrate route optimization trade-offs.
- Multi-objective methods (NSGA-II) provide broader and more diverse Pareto coverage than weighted-sum scalarization.
- Weather-driven constraints (storms, wave limits, time windows) materially shift optimal decisions.

## Limits

- **Calibration**: Real-world dataset is heterogeneous and weather is categorical; a single global model underfits (R² ≈ 0).
- **Routing constraints**: A* uses soft weather penalties; storm avoidance is validated after planning unless explicitly enforced.
- **MVP-3 scope**: NSGA-II optimizes constant speed on a fixed route; route shape is not optimized.
- **Performance**: NSGA-II and route-level optimization can be slow without compiled libraries or parallel evaluation.
- **Data realism**: Weather is synthetic and grid-based; integration with live data is out of scope.
