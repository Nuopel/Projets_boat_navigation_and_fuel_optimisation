# 🚢 Ship Route Optimizer - Interactive Dashboard

Interactive web-based dashboard for exploring multi-objective ship route optimization results.

## Features

- **Scenario Configuration**: Customize routes, ship parameters, and weather conditions
- **Multi-Algorithm Comparison**: Run NSGA-II and Weighted Sum optimizers
- **Interactive Pareto Front**: Explore time-fuel-emissions trade-offs
- **3D Visualization**: View complete objective space
- **Trade-off Analysis**: Understand speed-fuel-time relationships
- **Real-time Results**: See optimization progress and metrics

## Quick Start

### 1. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `streamlit`: Web dashboard framework
- `plotly`: Interactive visualizations
- `numpy`, `scipy`, `pandas`: Core computation

### 3. Run Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

### 4. Configure & Optimize

1. **Select Scenario**: Choose "Calm Weather", "Storm Detour", or "Custom"
2. **Set Route**: Configure start/goal positions
3. **Choose Algorithm**: NSGA-II, Weighted Sum, or Both
4. **Adjust Parameters**: Population size, generations, weight samples
5. **Add Constraints**: Enable time windows if needed
6. **Run**: Click "🚀 Run Optimization"

## Route Optimization Demo (NSGA-II over Route Geometry)

This repo includes an **experimental** route-geometry demo that extends MVP-3
to optimize waypoint positions under the storm scenario. It is a didactic
example: the results are expected to show the obvious trade-off (shorter time
vs. higher fuel) to validate that the multi-objective machinery works.

```bash
# Default run
python Exemples/run_mvp3_route_demo.py
```

Outputs:
- `outputs/mvp3_route_pareto.png` (Pareto front: time vs fuel)
- `outputs/mvp3_route_paths.png` (3 representative routes over the storm field)

Optional tuning:
```bash
MVP3_POP=30 MVP3_GEN=40 MVP3_ROUTE_WP=2 MVP3_ROUTE_SPEED=12 python Exemples/run_mvp3_route_demo.py
```

Notes:
- Uses **constant speed**; only route geometry is optimized.
- Storm avoidance relies on navigability checks and constraints; it is not a fully
  realistic routing engine.

## Dashboard Tabs

### 📊 Overview
- Solution metrics (count, ranges, statistics)
- Solution table with top 20 results
- Quick comparison between algorithms

### 🎯 Pareto Front
- Interactive 2D scatter plot (Time vs. Fuel)
- Hover for detailed solution information
- Optional 3D visualization (Time, Fuel, CO2)
- Color-coded by speed

### 📈 Analysis
- Speed vs. Time relationship
- Speed vs. Fuel relationship (cubic)
- Distribution comparisons (when running both algorithms)

### ⚙️ Details
- Full scenario configuration
- Ship specifications
- Optimization parameters
- Technical details

## Example Usage

### Scenario 1: Fuel-Optimal Route

```
Scenario: Calm Weather
Route: (50, 50) → (450, 450)
Algorithm: NSGA-II (pop=50, gen=50)
Constraint: None

Result: 50 Pareto solutions exploring 8-18 knots
Best Fuel: 60,530 L @ 8.45 kn (67.0 hours)
Best Time: 110,576 L @ 18.0 kn (31.4 hours)
```

### Scenario 2: Time-Constrained Optimization

```
Scenario: Calm Weather
Route: (50, 50) → (450, 450)
Algorithm: Weighted Sum (15 samples)
Constraint: Max 35 hours

Result: 15 solutions, 8 feasible
Optimal: 94,491 L @ 16.1 kn (35.1 hours)
```

### Scenario 3: Method Comparison

```
Scenario: Storm Detour
Algorithm: Both (NSGA-II + Weighted Sum)

Result:
- NSGA-II: 50 solutions (better diversity)
- Weighted Sum: 15 solutions (faster computation)
- Coverage: NSGA-II finds 3.3x more solutions
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Solutions** | Number of Pareto-optimal solutions found |
| **Time Range** | Min-max voyage time (hours) |
| **Fuel Range** | Min-max fuel consumption (liters) |
| **Speed Range** | Min-max optimal speeds (knots) |

## Tips

1. **Start Small**: Use pop=20, gen=10 for NSGA-II to test quickly
2. **Compare Methods**: Run "Both" to see NSGA-II vs. Weighted Sum differences
3. **Explore Trade-offs**: Click points on Pareto front to see detailed solutions
4. **Use Constraints**: Enable time windows to simulate realistic scenarios
5. **Save Results**: Take screenshots or export data for reports

## Algorithm Comparison

| Aspect | NSGA-II | Weighted Sum |
|--------|---------|--------------|
| **Solutions** | 20-100 | 5-30 |
| **Time** | 2-5 minutes | 30-60 seconds |
| **Diversity** | High | Medium |
| **Weight Tuning** | Not needed | Required |
| **Best For** | Exploration | Quick decisions |

## Troubleshooting

### Dashboard won't start
```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Check port availability
streamlit run dashboard.py --server.port 8502
```

### Optimization slow
- Reduce population size (20-30)
- Reduce generations (10-20)
- Use Weighted Sum with fewer samples (5-10)

### No solutions found
- Check route is feasible (not blocked by storms)
- Relax time window constraints
- Verify ship speed range is reasonable

## Architecture

```
dashboard.py (main app)
├── Sidebar: Configuration & Controls
├── Tab 1: Overview & Metrics
├── Tab 2: Pareto Front Visualization
├── Tab 3: Trade-off Analysis
└── Tab 4: Technical Details

Backend:
├── src/optimizers/nsga2.py: NSGA-II algorithm
├── src/optimizers/weighted_sum.py: Weighted sum method
└── src/planning/*: Route planning & evaluation
```

## Performance

| Configuration | Time | Solutions |
|---------------|------|-----------|
| NSGA-II (20×10) | ~30s | 20 |
| NSGA-II (50×50) | ~3min | 50 |
| Weighted Sum (15) | ~45s | 15 |
| Both | ~4min | 65 total |

## Future Enhancements

- [ ] Save/load scenarios
- [ ] Export Pareto front to CSV
- [ ] Multi-route comparison
- [ ] Real-time weather integration
- [ ] Fleet optimization
- [ ] Custom objective functions

## Support

For issues or questions:
1. Check logs in terminal where streamlit is running
2. Verify all dependencies installed: `pip list | grep streamlit`
3. Try clearing cache: `streamlit cache clear`

## Citation

If using this dashboard in research:
```
Multi-Objective Ship Route Optimizer Dashboard
NSGA-II and Weighted Sum comparison tool
November 2025
```

---

**Dashboard Version**: 1.0
**Compatible with**: Python 3.11+
**License**: MIT
