"""
Interactive dashboard for ship route optimization.

Streamlit-based web application for:
- Scenario configuration and optimization
- Pareto front visualization and exploration
- Solution comparison and trade-off analysis
- Parameter tuning and sensitivity analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.models.ship_model import create_default_ship, ShipSpecifications, ShipDynamics
from src.models.weather_field import create_calm_scenario, create_storm_scenario, WeatherField
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.optimizers.nsga2 import NSGA2Optimizer
from src.optimizers.weighted_sum import WeightedSumOptimizer
from src.planning.constraints import TimeWindow
from src.utils.geometry import Point

# Page configuration
st.set_page_config(
    page_title="Ship Route Optimizer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🚢 Multi-Objective Ship Route Optimizer</div>',
            unsafe_allow_html=True)

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Scenario selection
scenario_type = st.sidebar.selectbox(
    "Scenario Type",
    ["Calm Weather", "Storm Detour", "Custom"]
)

# Route configuration
st.sidebar.subheader("📍 Route")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_x = st.number_input("Start X (nm)", value=50, min_value=0, max_value=500)
    start_y = st.number_input("Start Y (nm)", value=50, min_value=0, max_value=500)
with col2:
    goal_x = st.number_input("Goal X (nm)", value=450, min_value=0, max_value=500)
    goal_y = st.number_input("Goal Y (nm)", value=450, min_value=0, max_value=500)

start = Point(start_x, start_y)
goal = Point(goal_x, goal_y)
direct_distance = start.distance_to(goal)

st.sidebar.metric("Direct Distance", f"{direct_distance:.1f} nm")

# Ship configuration
st.sidebar.subheader("🚢 Ship Parameters")
ship_speed_range = st.sidebar.slider(
    "Speed Range (knots)",
    min_value=5.0,
    max_value=25.0,
    value=(8.0, 18.0),
    step=0.5
)

# Optimizer selection
st.sidebar.subheader("🔧 Optimizer")
optimizer_type = st.sidebar.radio(
    "Algorithm",
    ["NSGA-II", "Weighted Sum", "Both"]
)

# NSGA-II parameters
if optimizer_type in ["NSGA-II", "Both"]:
    st.sidebar.subheader("NSGA-II Parameters")
    nsga_pop = st.sidebar.slider("Population Size", 20, 100, 50, 10)
    nsga_gen = st.sidebar.slider("Generations", 10, 100, 50, 10)

# Weighted Sum parameters
if optimizer_type in ["Weighted Sum", "Both"]:
    st.sidebar.subheader("Weighted Sum Parameters")
    ws_samples = st.sidebar.slider("Weight Samples", 5, 30, 15, 5)

# Constraints
st.sidebar.subheader("⚠️ Constraints")
use_time_window = st.sidebar.checkbox("Enable Time Window")
if use_time_window:
    max_time = st.sidebar.number_input("Max Time (hours)", value=35.0, min_value=10.0, max_value=100.0)
    time_window = TimeWindow(max_hours=max_time)
else:
    time_window = None

# Run optimization button
run_optimization = st.sidebar.button("🚀 Run Optimization", type="primary")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Pareto Front", "📈 Analysis", "⚙️ Details"])

# Tab 1: Overview
with tab1:
    st.header("Optimization Overview")

    if run_optimization:
        with st.spinner("Optimizing routes..."):
            # Setup environment
            grid_shape = (50, 50)
            cell_size = 10.0

            if scenario_type == "Calm Weather":
                weather = create_calm_scenario(grid_shape, cell_size)
            elif scenario_type == "Storm Detour":
                storm_center = Point(250, 250)
                weather = create_storm_scenario(grid_shape, cell_size, storm_center, 80.0)
            else:
                weather = create_calm_scenario(grid_shape, cell_size)

            constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
            env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

            # Create ship with custom speed range
            ship_specs = ShipSpecifications(
                name="Custom Vessel",
                v_min=ship_speed_range[0],
                v_max=ship_speed_range[1],
                fuel_coef_speed=0.5,
                fuel_coef_wind=0.8,
                fuel_coef_wave=15.0,
                fuel_base=500.0,
                emission_factor=2.8
            )
            ship = ShipDynamics(ship_specs)

            # Run optimization
            results = {}

            if optimizer_type in ["NSGA-II", "Both"]:
                st.info("Running NSGA-II...")
                nsga2_opt = NSGA2Optimizer(ship, env, nsga_pop, nsga_gen)
                pareto_nsga2 = nsga2_opt.optimize_pareto(start, goal, use_astar=True, verbose=False)
                results['nsga2'] = pareto_nsga2
                st.success(f"✓ NSGA-II complete: {len(pareto_nsga2)} solutions")

            if optimizer_type in ["Weighted Sum", "Both"]:
                st.info("Running Weighted Sum...")
                ws_opt = WeightedSumOptimizer(ship, env)
                ws_results = ws_opt.scan_weight_space(start, goal, num_samples=ws_samples)
                from src.optimizers.pareto_utils import ParetoSolution
                pareto_ws = [
                    ParetoSolution(
                        objectives=(r.objectives.time_hours, r.objectives.fuel_liters,
                                   r.objectives.emissions_kg),
                        result=r
                    )
                    for r in ws_results if r.success
                ]
                results['weighted_sum'] = pareto_ws
                st.success(f"✓ Weighted Sum complete: {len(pareto_ws)} solutions")

            # Store results in session state
            st.session_state['results'] = results
            st.session_state['scenario'] = {
                'type': scenario_type,
                'start': start,
                'goal': goal,
                'ship': ship,
                'env': env
            }

    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']

        # Metrics
        st.subheader("📊 Solution Metrics")
        cols = st.columns(4)

        if 'nsga2' in results:
            pareto = results['nsga2']
            times = [s.objectives[0] for s in pareto]
            fuels = [s.objectives[1] for s in pareto]
            speeds = [s.result.speed for s in pareto]

            with cols[0]:
                st.metric("NSGA-II Solutions", len(pareto))
            with cols[1]:
                st.metric("Time Range", f"{min(times):.1f} - {max(times):.1f}h")
            with cols[2]:
                st.metric("Fuel Range", f"{min(fuels)/1000:.0f} - {max(fuels)/1000:.0f}k L")
            with cols[3]:
                st.metric("Speed Range", f"{min(speeds):.1f} - {max(speeds):.1f} kn")

        if 'weighted_sum' in results:
            pareto = results['weighted_sum']
            st.divider()
            cols2 = st.columns(4)

            times = [s.objectives[0] for s in pareto]
            fuels = [s.objectives[1] for s in pareto]
            speeds = [s.result.speed for s in pareto]

            with cols2[0]:
                st.metric("Weighted Sum Solutions", len(pareto))
            with cols2[1]:
                st.metric("Time Range", f"{min(times):.1f} - {max(times):.1f}h")
            with cols2[2]:
                st.metric("Fuel Range", f"{min(fuels)/1000:.0f} - {max(fuels)/1000:.0f}k L")
            with cols2[3]:
                st.metric("Speed Range", f"{min(speeds):.1f} - {max(speeds):.1f} kn")

        # Solution table
        st.subheader("📋 Solution Table")

        if 'nsga2' in results:
            pareto = results['nsga2']
            df_data = []
            for i, sol in enumerate(pareto[:20]):  # Show first 20
                df_data.append({
                    'ID': i,
                    'Speed (kn)': f"{sol.result.speed:.2f}",
                    'Time (h)': f"{sol.objectives[0]:.2f}",
                    'Fuel (L)': f"{sol.objectives[1]:,.0f}",
                    'CO2 (kg)': f"{sol.objectives[2]:,.0f}",
                    'Distance (nm)': f"{sol.result.objectives.distance_nm:.1f}"
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("👈 Configure parameters and click 'Run Optimization' to begin")

# Tab 2: Pareto Front
with tab2:
    st.header("🎯 Pareto Front Explorer")

    if 'results' in st.session_state:
        results = st.session_state['results']

        # Interactive Pareto front plot
        fig = go.Figure()

        if 'nsga2' in results:
            pareto = results['nsga2']
            times = [s.objectives[0] for s in pareto]
            fuels = [s.objectives[1] for s in pareto]
            speeds = [s.result.speed for s in pareto]

            fig.add_trace(go.Scatter(
                x=times,
                y=fuels,
                mode='markers',
                name='NSGA-II',
                marker=dict(
                    size=12,
                    color=speeds,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed (kn)"),
                    line=dict(width=2, color='DarkBlue')
                ),
                text=[f"Speed: {s:.2f} kn<br>Time: {t:.2f}h<br>Fuel: {f:,.0f}L"
                      for s, t, f in zip(speeds, times, fuels)],
                hovertemplate='%{text}<extra></extra>'
            ))

        if 'weighted_sum' in results:
            pareto = results['weighted_sum']
            times = [s.objectives[0] for s in pareto]
            fuels = [s.objectives[1] for s in pareto]
            speeds = [s.result.speed for s in pareto]

            fig.add_trace(go.Scatter(
                x=times,
                y=fuels,
                mode='markers',
                name='Weighted Sum',
                marker=dict(
                    size=14,
                    color=speeds,
                    colorscale='Plasma',
                    symbol='square',
                    line=dict(width=2, color='DarkRed')
                ),
                text=[f"Speed: {s:.2f} kn<br>Time: {t:.2f}h<br>Fuel: {f:,.0f}L"
                      for s, t, f in zip(speeds, times, fuels)],
                hovertemplate='%{text}<extra></extra>'
            ))

        fig.update_layout(
            title="Pareto Front: Time vs. Fuel Trade-off",
            xaxis_title="Voyage Time (hours)",
            yaxis_title="Fuel Consumption (liters)",
            height=600,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # 3D Pareto Front
        if 'nsga2' in results and st.checkbox("Show 3D Pareto Front"):
            pareto = results['nsga2']
            times = [s.objectives[0] for s in pareto]
            fuels = [s.objectives[1] for s in pareto]
            emissions = [s.objectives[2] for s in pareto]
            speeds = [s.result.speed for s in pareto]

            fig_3d = go.Figure(data=[go.Scatter3d(
                x=times,
                y=fuels,
                z=emissions,
                mode='markers',
                marker=dict(
                    size=8,
                    color=speeds,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed (kn)")
                ),
                text=[f"Speed: {s:.2f} kn" for s in speeds],
                hovertemplate='Time: %{x:.1f}h<br>Fuel: %{y:,.0f}L<br>CO2: %{z:,.0f}kg<br>%{text}<extra></extra>'
            )])

            fig_3d.update_layout(
                title="3D Pareto Front",
                scene=dict(
                    xaxis_title="Time (hours)",
                    yaxis_title="Fuel (liters)",
                    zaxis_title="CO2 (kg)"
                ),
                height=700
            )

            st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Run optimization first to view Pareto front")

# Tab 3: Analysis
with tab3:
    st.header("📈 Trade-off Analysis")

    if 'results' in st.session_state:
        results = st.session_state['results']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Time vs. Speed")
            if 'nsga2' in results:
                pareto = results['nsga2']
                times = [s.objectives[0] for s in pareto]
                speeds = [s.result.speed for s in pareto]

                fig_ts = px.scatter(x=speeds, y=times,
                                   labels={'x': 'Speed (kn)', 'y': 'Time (h)'},
                                   title="Speed vs. Time Relationship")
                st.plotly_chart(fig_ts, use_container_width=True)

        with col2:
            st.subheader("Fuel vs. Speed")
            if 'nsga2' in results:
                pareto = results['nsga2']
                fuels = [s.objectives[1] for s in pareto]
                speeds = [s.result.speed for s in pareto]

                fig_fs = px.scatter(x=speeds, y=fuels,
                                   labels={'x': 'Speed (kn)', 'y': 'Fuel (L)'},
                                   title="Speed vs. Fuel Relationship (Cubic)")
                st.plotly_chart(fig_fs, use_container_width=True)

        # Distribution comparison
        if len(results) == 2:
            st.subheader("Distribution Comparison")

            # Time distributions
            fig_dist = go.Figure()

            pareto_n = results['nsga2']
            times_n = [s.objectives[0] for s in pareto_n]
            fig_dist.add_trace(go.Histogram(x=times_n, name='NSGA-II', opacity=0.6))

            pareto_w = results['weighted_sum']
            times_w = [s.objectives[0] for s in pareto_w]
            fig_dist.add_trace(go.Histogram(x=times_w, name='Weighted Sum', opacity=0.6))

            fig_dist.update_layout(
                title="Time Distribution Comparison",
                xaxis_title="Voyage Time (hours)",
                yaxis_title="Count",
                barmode='overlay'
            )

            st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Run optimization first to view analysis")

# Tab 4: Details
with tab4:
    st.header("⚙️ Technical Details")

    if 'scenario' in st.session_state:
        scenario = st.session_state['scenario']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Scenario Configuration")
            st.json({
                'Type': scenario['type'],
                'Start': f"({scenario['start'].x}, {scenario['start'].y})",
                'Goal': f"({scenario['goal'].x}, {scenario['goal'].y})",
                'Distance': f"{direct_distance:.1f} nm"
            })

            st.subheader("Ship Specifications")
            ship = scenario['ship']
            st.json({
                'Name': ship.specs.name,
                'Speed Range': f"{ship.specs.v_min} - {ship.specs.v_max} knots",
                'Fuel Coefficients': {
                    'Speed': ship.specs.fuel_coef_speed,
                    'Wind': ship.specs.fuel_coef_wind,
                    'Wave': ship.specs.fuel_coef_wave,
                    'Base': ship.specs.fuel_base
                }
            })

        with col2:
            if 'results' in st.session_state:
                st.subheader("Optimization Results")
                results = st.session_state['results']

                for method, pareto in results.items():
                    st.write(f"**{method.upper()}**")
                    st.write(f"- Solutions: {len(pareto)}")
                    st.write(f"- All non-dominated: ✓")

                    if method == 'nsga2':
                        st.write(f"- Parameters: pop={nsga_pop}, gen={nsga_gen}")
                    else:
                        st.write(f"- Weight samples: {ws_samples}")
    else:
        st.info("Run optimization to view details")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
**Ship Route Optimizer Dashboard**

Multi-objective optimization using:
- NSGA-II (evolutionary)
- Weighted Sum (scalarization)

Optimizes: Time, Fuel, Emissions
""")
