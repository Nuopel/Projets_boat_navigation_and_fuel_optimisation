# Ship Fuel Efficiency Dataset - Data Dictionary

## Dataset Overview

**Source:** Nigerian maritime operational data
**Size:** 1,440 observations
**Time Period:** 12 months (January - December)
**Ships:** 120 unique vessels
**Purpose:** Predict fuel consumption for ship voyages with uncertainty quantification

---

## Feature Descriptions

### Identifier Features

#### 1. `ship_id` (Categorical)

- **Description:** Unique identifier for each vessel
- **Data Type:** String (format: NG###)
- **Unique Values:** 120
- **Example Values:** NG001, NG002, NG003, ...
- **Usage:** Can be used for ship-specific modeling or removed for generalization
- **Notes:** Each ship appears 12 times (one observation per month)

---

### Operational Features

#### 2. `ship_type` (Categorical)

- **Description:** Type/class of vessel
- **Data Type:** String
- **Unique Values:** 4
- **Categories:**
  - Oil Service Boat (408 observations, 28.3%)
  - Tanker Ship (408 observations, 28.3%)
  - Surfer Boat (324 observations, 22.5%)
  - Fishing Trawler (300 observations, 20.8%)
- **Usage:** Critical feature - different ship types have different fuel consumption patterns
- **Domain Knowledge:** Tankers typically consume more fuel due to larger size and displacement

#### 3. `route_id` (Categorical)

- **Description:** Voyage route between Nigerian ports
- **Data Type:** String
- **Unique Values:** 4
- **Categories:**
  - Port Harcourt-Lagos (389 observations, 27.0%)
  - Lagos-Apapa (388 observations, 26.9%)
  - Escravos-Lagos (369 observations, 25.6%)
  - Warri-Bonny (294 observations, 20.4%)
- **Usage:** Route characteristics (currents, traffic, port congestion) affect fuel consumption
- **Notes:** Some routes may be more fuel-efficient due to favorable currents or shorter distances

#### 4. `month` (Categorical/Temporal)

- **Description:** Month of voyage
- **Data Type:** String
- **Unique Values:** 12 (January - December)
- **Distribution:** Uniform (120 observations per month)
- **Usage:** Captures seasonal effects (monsoons, weather patterns, sea conditions)
- **Feature Engineering:** Can be converted to cyclical features (sin/cos) to capture seasonality

#### 5. `distance` (Numerical)

- **Description:** Voyage distance in nautical miles (nm)
- **Data Type:** Float
- **Unit:** Nautical miles (1 nm = 1.852 km)
- **Range:** [29.38, 197.46] nm
- **Mean:** 114.3 nm
- **Std Dev:** 43.2 nm
- **Outliers:** 145 observations (10.07%) based on IQR method
- **Usage:** PRIMARY predictor - fuel consumption strongly correlates with distance
- **Domain Knowledge:** Fuel consumption typically scales with distance, but not linearly due to speed variations

#### 6. `fuel_type` (Categorical)

- **Description:** Type of fuel used for voyage
- **Data Type:** String
- **Unique Values:** 2
- **Categories:**
  - Diesel (899 observations, 62.4%)
  - HFO - Heavy Fuel Oil (541 observations, 37.6%)
- **Usage:** Critical feature - fuel types have different energy densities and combustion efficiencies
- **Domain Knowledge:**
  - Diesel: Cleaner, more expensive, higher energy density (~42.7 MJ/kg)
  - HFO: Cheaper, dirtier, lower energy density (~40.5 MJ/kg)
  - Expect different fuel consumption rates even for same voyage

---

### Target Features

#### 7. `fuel_consumption` (Numerical) 🎯 PRIMARY TARGET

- **Description:** Total fuel consumed during voyage
- **Data Type:** Float
- **Unit:** Tonnes (metric tons)
- **Range:** [855.01, 5,695.04] tonnes
- **Mean:** 3,162.5 tonnes
- **Std Dev:** 1,105.3 tonnes
- **Outliers:** 226 observations (15.69%) based on IQR method
- **Usage:** PRIMARY target variable for regression models
- **Notes:** Outliers likely represent extreme weather conditions or operational inefficiencies
- **Derived Feature:** Can compute `fuel_rate = fuel_consumption / distance` for efficiency metric

#### 8. `CO2_emissions` (Numerical)

- **Description:** Carbon dioxide emissions produced during voyage
- **Data Type:** Float
- **Unit:** Tonnes CO2
- **Range:** [2,195.90, 16,991.82] tonnes
- **Mean:** 8,565.9 tonnes
- **Std Dev:** 3,389.2 tonnes
- **Outliers:** 230 observations (15.97%)
- **Usage:**
  - Can be used as alternative target for emissions prediction
  - Highly correlated with fuel_consumption (CO2 = fuel × emission_factor)
  - **NOT recommended as predictor** (leakage risk - CO2 directly derived from fuel)
- **Domain Knowledge:**
  - HFO emission factor: ~3.114 tonnes CO2/tonne fuel
  - Diesel emission factor: ~3.206 tonnes CO2/tonne fuel

---

### Environmental Features

#### 9. `weather_conditions` (Categorical)

- **Description:** Prevailing weather conditions during voyage
- **Data Type:** String
- **Unique Values:** 3
- **Categories:**
  - Calm (516 observations, 35.8%)
  - Stormy (462 observations, 32.1%)
  - Moderate (462 observations, 32.1%)
- **Usage:** Critical feature - weather significantly impacts fuel consumption
- **Domain Knowledge:**
  - Calm: Minimal wind/waves, optimal fuel efficiency
  - Moderate: Some resistance, moderate fuel increase (~15-20%)
  - Stormy: High resistance, significant fuel penalty (~40-60%)
- **Feature Engineering:** Can be ordinal encoded (Calm=0, Moderate=1, Stormy=2) for physics models

---

### Performance Features

#### 10. `engine_efficiency` (Numerical)

- **Description:** Engine operating efficiency during voyage
- **Data Type:** Float
- **Unit:** Percentage (%)
- **Range:** [70.62, 94.68]%
- **Mean:** 83.6%
- **Std Dev:** 6.8%
- **Outliers:** 0 observations (no outliers detected)
- **Usage:** Critical feature - lower efficiency → more fuel for same distance
- **Domain Knowledge:**
  - > 90%: Excellent condition (new engines, optimal load)
  - 80-90%: Good condition (typical operational range)
  - <80%: Degraded condition (maintenance needed, hull fouling, sub-optimal loading)
- **Physics Relationship:** fuel_consumption ∝ 1 / engine_efficiency

---

## Data Quality Summary

### Completeness

- **Missing Values:** 0% - EXCELLENT
- All 1,440 observations have complete data across all 10 features
- No imputation required

### Consistency

- **Duplicates:** 0 rows
- **Data Types:** Appropriate (categorical as strings, numerical as floats)
- **Value Ranges:** All within realistic operational bounds

### Outliers

| Feature           | Outlier Count | Percentage | Action                                                                   |
| ----------------- | ------------- | ---------- | ------------------------------------------------------------------------ |
| distance          | 145           | 10.07%     | **Keep** - long voyages are legitimate                                   |
| fuel_consumption  | 226           | 15.69%     | **Investigate** - may indicate data quality issues OR extreme conditions |
| CO2_emissions     | 230           | 15.97%     | **Keep** - correlated with fuel outliers                                 |
| engine_efficiency | 0             | 0.00%      | None needed                                                              |

**Recommendation:** Cap extreme outliers at 99th percentile or use robust models (tree-based) that handle outliers well.

---

## Feature Engineering Opportunities

### Physics-Based Features

1. **fuel_rate** = fuel_consumption / distance
   
   - Fuel efficiency metric (tonnes/nm)
   - Normalizes for distance variation

2. **fuel_intensity** = CO2_emissions / fuel_consumption
   
   - Proxy for fuel quality/type

3. **efficiency_deviation** = engine_efficiency - mean(engine_efficiency by ship_type)
   
   - Captures relative performance vs. fleet average

4. **weather_penalty_factor**:
   
   - Calm: 1.0
   - Moderate: 1.2
   - Stormy: 1.5
   - Based on domain knowledge

### Interaction Features

1. **distance × weather_conditions** - Longer voyages amplify weather impact
2. **engine_efficiency × fuel_type** - Diesel engines may respond differently
3. **ship_type × route_id** - Some ships optimized for specific routes

### Temporal Features

1. **month_sin, month_cos** - Cyclical encoding for seasonality
2. **is_monsoon** - Binary flag for wet season months (May-September in Nigeria)

### Categorical Encoding Strategies

- **ship_id:** Drop (too many unique values) OR target encode
- **ship_type:** One-hot encode (only 4 categories)
- **route_id:** One-hot encode OR frequency encode
- **fuel_type:** Binary encode (HFO=1, Diesel=0)
- **weather_conditions:** Ordinal encode (Calm=0, Moderate=1, Stormy=2)

---

## Train/Validation/Test Split Strategy

### Ratios

- **Train:** 70% (1,008 observations)
- **Validation:** 15% (216 observations)
- **Test:** 15% (216 observations)

### Stratification

Stratify by **ship_type** to ensure:

- All 4 ship types represented in each split
- Proportional distribution maintained
- Reduces evaluation variance

### Random Seed

- **Seed:** 42 (reproducibility)

---

## Domain Knowledge & Physics Relationships

### Expected Relationships

1. **fuel_consumption ∝ distance** (strong positive)
2. **fuel_consumption ∝ 1 / engine_efficiency** (inverse)
3. **fuel_consumption higher for:**
   - Stormy > Moderate > Calm weather
   - HFO vs. Diesel (slightly)
   - Tanker Ship > Oil Service Boat > Fishing Trawler (due to size)

### Physics Model (Simplified)

```
fuel_consumption ≈ k × distance × weather_factor / (engine_efficiency / 100)

where:
- k = ship-specific constant (calibrated per ship_type)
- weather_factor: Calm=1.0, Moderate=1.2, Stormy=1.5
- engine_efficiency normalized to decimal (e.g., 85% → 0.85)
```

This simplified model will serve as the **physics baseline** in MVP-3.

---

## Data Usage Guidelines

### Model Training

- **Predictors:** All features EXCEPT ship_id, fuel_consumption, CO2_emissions
- **Target:** fuel_consumption
- **Exclude CO2_emissions:** Directly derived from target (data leakage)

### Feature Selection Priority

**Tier 1 (Must Include):**

- distance
- ship_type
- weather_conditions
- engine_efficiency

**Tier 2 (Recommended):**

- fuel_type
- route_id
- month

**Tier 3 (Optional):**

- ship_id (high cardinality, use with caution)

### Validation Checks

1. **Sanity Check:** Fuel consumption increases with distance (positive correlation)
2. **Physics Check:** Stormy weather → higher fuel than Calm for same distance
3. **Efficiency Check:** Lower engine_efficiency → higher fuel consumption

---

## References & Data Lineage

**Source:** Nigerian maritime operational records
**Collection Period:** 12 months
**Data Quality:** Production-grade operational data
**Preprocessing:** Minimal (no missing values, validated ranges)
**Last Updated:** 2025-11-18

---

## Contact & Questions

For questions about this dataset or feature definitions, refer to:

- Project documentation: `WBS_ML_Ship_Fuel_Prediction.md`
- EDA notebook: `notebooks/01_eda.ipynb`
- Profiling script: `src/data_profiler.py`?
