# Coefficient Analysis Results
**LinearRegression Model - Energy Consumption Prediction**

---

## Executive Summary

This document analyzes the trained LinearRegression model coefficients to understand **what each feature means** and **how predictions are made**.

**Key Finding**: The model has significant **multicollinearity** - several features are highly correlated (r > 0.9), which affects coefficient interpretation. Ridge regression is recommended to stabilize coefficients.

---

## Model Equation

```
heating_load = 22.16 + 6.87×cooling_load + 2.08×overall_height
               + 1.26×glazing_area - 1.09×relative_compactness
               - 0.93×roof_area + 0.70×wall_area - 0.60×surface_area
               + 0.29×glazing_area_distribution - 0.08×orientation
```

*Note: All features are **scaled** (mean=0, std=1) before applying coefficients*

---

## 1. Intercept (β₀ = 22.16 kWh)

**What it means**: The predicted heating load when all features are at their **mean value**.

- This is the "baseline" heating load for an "average" building
- When a scaled feature = 0, it means the original feature is at its mean
- Example: Average building height, average surface area, etc.

**Physical interpretation**: The typical building in this dataset requires about 22 kWh of heating.

---

## 2. Individual Coefficients (Sorted by Magnitude)

### Rank 1: cooling_load (β = +6.87)
**Effect**: +6.87 kWh per 1 standard deviation increase

**What it means**:
- **Strongest predictor by far** (3.3× more important than next feature)
- Positive relationship: Higher cooling load → Higher heating load
- Both loads measure the same thing: **thermal inefficiency**

**Physical interpretation**:
- Buildings that gain heat easily (high cooling) also lose heat easily (high heating)
- Poor insulation affects both directions
- `cooling_load` acts as a **proxy for overall thermal performance**

**Why it dominates**: Captures the building's fundamental thermal behavior

---

### Rank 2: overall_height (β = +2.08)
**Effect**: +2.08 kWh per 1 standard deviation increase

**What it means**:
- Taller buildings need more heating
- Positive relationship: Height ↑ → Heating ↑

**Physical interpretation**:
- **Volume effect**: More cubic meters of air to heat
- **Stack effect**: Hot air rises and escapes from top
- **Surface-to-volume ratio**: Taller buildings have more exterior surface per volume

**Correlation note**: Highly correlated with `cooling_load` (r=0.895) and `roof_area` (r=-0.973)

---

### Rank 3: glazing_area (β = +1.26)
**Effect**: +1.26 kWh per 1 standard deviation increase

**What it means**:
- More windows → More heating needed
- Positive relationship: Glass area ↑ → Heating ↑

**Physical interpretation**:
- **Thermal weakness**: Glass R-value ~1 vs wall R-value ~20
- Windows are the **weak link** in building envelope
- Heat escapes 20× faster through glass than walls

**Practical impact**: Every 1-std increase in glazing area (≈0.13 m²) adds 1.26 kWh to heating load

---

### Rank 4: relative_compactness (β = -1.09)
**Effect**: -1.09 kWh per 1 standard deviation increase

**What it means**:
- More compact buildings need **less** heating
- Negative relationship: Compactness ↑ → Heating ↓

**Physical interpretation**:
- **Shape efficiency**: Spheres and cubes minimize surface area per volume
- Compact shapes → Less exterior surface → Less heat loss
- Formula: `relative_compactness = volume / surface_area`

**Why negative?**: Higher compactness = better efficiency = lower heating

**Multicollinearity warning**: r = -0.992 with `surface_area` (nearly perfect inverse!)

---

### Rank 5: roof_area (β = -0.93)
**Effect**: -0.93 kWh per 1 standard deviation increase

**What it means**:
- Surprisingly, **larger roof decreases heating**
- Negative relationship: Roof area ↑ → Heating ↓

**Why negative? (Counterintuitive!)**:
This coefficient is **misleading due to multicollinearity**:
- `roof_area` correlates r=-0.973 with `overall_height`
- `roof_area` correlates r=-0.865 with `cooling_load`
- `roof_area` correlates r=+0.882 with `surface_area`

**True interpretation**:
- The negative sign is an **artifact of correlated features**
- Physically, larger roof **should** increase heating (more surface to lose heat)
- But the model sees: "tall buildings (small roof) need more heating"
- Coefficient captures the **residual effect after accounting for height and cooling_load**

**Takeaway**: Don't trust individual coefficients when multicollinearity is high!

---

### Rank 6: wall_area (β = +0.70)
**Effect**: +0.70 kWh per 1 standard deviation increase

**What it means**:
- More wall surface → More heating
- Positive relationship: Wall area ↑ → Heating ↑

**Physical interpretation**:
- Walls are the **primary heat transfer surface**
- Heat loss through walls via conduction and convection
- Larger wall area = more heat escape route

**Why moderate importance?**: Other features (cooling_load, surface_area) already capture much of this information

---

### Rank 7: surface_area (β = -0.60)
**Effect**: -0.60 kWh per 1 standard deviation increase

**What it means**:
- Larger total surface **decreases** heating (counterintuitive!)
- Negative relationship: Surface area ↑ → Heating ↓

**Why negative? (Multicollinearity again!)**:
- `surface_area` correlates r=-0.992 with `relative_compactness`
- `surface_area` correlates r=-0.861 with `overall_height`

**True interpretation**:
- Physically, more surface area **should** increase heating
- Negative coefficient is due to **confounding with compactness**
- Model captures: "For a given volume, higher surface area (lower compactness) increases heating"
- But direct relationship is distorted by correlations

**Takeaway**: Another victim of multicollinearity

---

### Rank 8: glazing_area_distribution (β = +0.29)
**Effect**: +0.29 kWh per 1 standard deviation increase

**What it means**:
- How windows are distributed across building faces
- Values: 0 (no windows), 1-4 (window distribution patterns)
- Positive relationship: Higher distribution value → Slightly more heating

**Physical interpretation**:
- **Secondary effect**: Window placement matters less than total window area
- Uniform distribution (value=4) may create more thermal bridges
- But effect is small compared to total glazing area

**Why low importance?**: `glazing_area` (total) is 4.4× more important than distribution

---

### Rank 9: orientation (β = -0.08)
**Effect**: -0.08 kWh per 1 standard deviation increase

**What it means**:
- Building orientation (N/S/E/W) has **minimal impact**
- Weakest predictor in the model

**Why so small?**:
- Dataset is from **simulations**, not real buildings
- May not account for realistic solar gains
- In real buildings, south-facing (N hemisphere) reduces heating via passive solar

**Practical implication**: Orientation is nearly irrelevant for this dataset

---

## 3. Prediction Examples

Let's see how coefficients work together to make predictions:

### Example 1: Low Heating Building (6.01 kWh actual)

```
Prediction = 22.16  (base)
           - 9.77  (cooling_load: well insulated, low thermal loss)
           - 2.22  (glazing_area: small windows)
           - 2.05  (overall_height: short building)
           - 1.19  (wall_area: small wall surface)
           - 0.89  (roof_area: small roof)
           - 0.42  (other features)
           -------
           = 5.63 kWh (predicted)
```

**Error**: 6.3% - Excellent prediction!

**Key insight**: `cooling_load` dominates (-9.77 kWh), showing this building is thermally efficient

---

### Example 2: Medium Heating Building (18.31 kWh actual)

```
Prediction = 22.16  (base)
           + 2.51  (cooling_load: moderate thermal efficiency)
           - 2.22  (glazing_area: still small windows)
           + 2.11  (overall_height: taller than Example 1)
           - 1.01  (relative_compactness: less compact)
           + 0.63  (roof_area)
           - 0.44  (other features)
           -------
           = 23.73 kWh (predicted)
```

**Error**: 29.6% - Larger error, model overestimates

**Key insight**: Features push in opposite directions, creating more uncertainty

---

### Example 3: High Heating Building (43.10 kWh actual)

```
Prediction = 22.16  (base)
           +10.88  (cooling_load: poor insulation, high thermal loss)
           + 2.11  (overall_height: tall)
           + 1.55  (glazing_area: large windows)
           + 0.63  (roof_area)
           + 0.39  (wall_area)
           - 0.15  (other features)
           -------
           = 37.56 kWh (predicted)
```

**Error**: 12.8% - Underestimates, but still reasonable

**Key insight**: `cooling_load` dominates (+10.88 kWh), showing this building is thermally inefficient

---

## 4. The Multicollinearity Problem

### High Correlations Found (|r| > 0.7):

| Feature 1              | Feature 2              | Correlation | Issue                        |
|------------------------|------------------------|-------------|------------------------------|
| relative_compactness   | surface_area           | r = -0.992  | Nearly perfect inverse!      |
| roof_area              | overall_height         | r = -0.973  | Tall → small roof            |
| overall_height         | cooling_load           | r = +0.895  | Height drives thermal load   |
| surface_area           | roof_area              | r = +0.882  | Both measure size            |
| relative_compactness   | roof_area              | r = -0.870  | Compact → large roof/volume  |
| roof_area              | cooling_load           | r = -0.865  | Confounding relationship     |
| surface_area           | overall_height         | r = -0.861  | Size-height tradeoff         |
| relative_compactness   | overall_height         | r = +0.831  | Both capture shape           |

### Consequences:

1. **Unstable coefficients**: Small data changes → Large coefficient changes
2. **Sign flips**: Coefficients may have wrong sign (roof_area, surface_area)
3. **Misleading importance**: Individual feature impact is confounded
4. **High variance**: Coefficients have high standard errors

### Example of Instability:

`roof_area` coefficient is **-0.93** (negative), suggesting larger roofs reduce heating.

**But physically**: Larger roofs should increase heating (more surface to lose heat)

**Why the flip?**: `roof_area` correlates with:
- `overall_height` (r=-0.973): Tall buildings have proportionally smaller roofs
- `cooling_load` (r=-0.865): Buildings with large roofs have lower cooling needs

The model "sees": "When roof_area increases (controlling for other features), heating decreases"

But this is because **roof_area is a proxy for building shape**, not a causal effect.

---

## 5. Solutions to Multicollinearity

### Option 1: Ridge Regression (Recommended)

**What it does**: Adds penalty to large coefficients, shrinking them toward zero

**Benefits**:
- Reduces coefficient variance
- More stable predictions
- Better generalization

**How to implement**:
```yaml
# config/config.yaml
model:
  type: Ridge
  params:
    alpha: 1.0  # Regularization strength
    fit_intercept: true
```

**Expected result**: Coefficients will be smaller but more interpretable

---

### Option 2: Remove Highly Correlated Features

**Strategy**: Drop one feature from each highly correlated pair

**Features to drop**:
- `surface_area` (keeps `relative_compactness`)
- `roof_area` (keeps `overall_height`)

**Benefits**:
- Eliminates multicollinearity
- Simpler model
- More interpretable coefficients

**Drawback**: May lose some predictive power (but likely minimal)

---

### Option 3: Principal Component Analysis (PCA)

**What it does**: Creates uncorrelated linear combinations of features

**Benefits**:
- Zero multicollinearity
- Reduces dimensionality

**Drawbacks**:
- **Loses interpretability** (coefficients are for "PC1", "PC2", not physical features)
- Not recommended if interpretation is important

---

### Option 4: Remove `cooling_load`

**Rationale**: Since `cooling_load` dominates and correlates with many features, removing it might reveal other relationships

**Expected outcome**: Other features (height, glazing, compactness) will become more important

**Trade-off**: Lower R² (maybe 0.85 instead of 0.96)

**When to use**: If you want to predict heating_load **without knowing** cooling_load

---

## 6. Why Coefficient ≈ Importance (After Scaling)

**Before scaling**: Features have different scales (88 vs 0.11), so raw coefficients are misleading

**After StandardScaler**: All features have mean=0, std=1

**Result**: 1-unit change in scaled feature = 1-std change in original feature

**Therefore**: Coefficient directly represents the effect of a **typical variation** in the feature

**Comparison**:

| Feature                  | Coefficient | Std (scaled) | Importance | Difference |
|--------------------------|-------------|--------------|------------|------------|
| cooling_load             | 6.8678      | 1.0008       | 6.8734     | 0.0056     |
| overall_height           | 2.0804      | 1.0008       | 2.0821     | 0.0017     |
| glazing_area             | 1.2624      | 1.0008       | 1.2634     | 0.0010     |
| relative_compactness     | 1.0866      | 1.0008       | 1.0874     | 0.0008     |

**Conclusion**: After scaling, coefficient magnitude ≈ importance (difference < 0.1%)

**Why the tiny difference?**: Scaled features have std slightly above 1.0 due to rounding

---

## 7. Key Takeaways

### ✅ What We Learned:

1. **Intercept (22.16 kWh)**: Baseline heating for "average" building
2. **cooling_load dominates**: 3.3× more important than any other feature
3. **Positive coefficients**: cooling_load, height, glazing, wall_area increase heating
4. **Negative coefficients**: compactness, roof_area, surface_area decrease heating
5. **After scaling**: coefficient magnitude = feature importance
6. **Prediction = linear combination**: Sum of (coefficient × scaled_feature)

### ⚠️ Critical Issues:

1. **Multicollinearity is severe**: 8 pairs of features with |r| > 0.8
2. **Some coefficients have wrong signs**: roof_area, surface_area (due to confounding)
3. **Don't trust individual coefficients**: They capture residual effects, not causal effects
4. **Ridge regression recommended**: Will stabilize coefficients

### 🔬 Physical Insights:

1. **cooling_load is a thermal efficiency proxy**: Captures overall insulation quality
2. **Height matters**: Taller buildings = more volume = more heating
3. **Glazing is critical**: Windows are thermal weak points
4. **Shape efficiency**: Compact buildings minimize heat loss
5. **Orientation is irrelevant**: Simulated data may not reflect real solar gains

### 📊 Model Performance Context:

- **R² = 0.96**: Excellent fit
- **No negative predictions**: Unlike many models
- **No overfitting**: Train and test metrics nearly identical
- **But**: Multicollinearity limits coefficient interpretation

---

## 8. Next Steps

### Immediate Actions:

1. **Try Ridge Regression**
   - Handles multicollinearity
   - More stable coefficients
   - Still interpretable

2. **Remove `cooling_load` and retrain**
   - See how other features compensate
   - More realistic for prediction scenarios where cooling_load is unknown

3. **Analyze residuals**
   - Check if linear model assumptions hold
   - Identify patterns in prediction errors

### Future Analysis:

4. **Compare with Lasso Regression**
   - L1 regularization performs feature selection
   - May automatically drop correlated features

5. **Investigate interaction terms**
   - Example: `glazing_area × orientation` (windows facing south)
   - May improve predictions

6. **Try polynomial features**
   - Example: `surface_area²`, `height²`
   - Capture non-linear relationships

7. **Build model without highly correlated features**
   - Drop surface_area and roof_area
   - See if performance remains high

---

## Appendix: How to Read Coefficients

### Formula:
```
heating_load = β₀ + β₁×x₁ + β₂×x₂ + ... + βₙ×xₙ
```

### Interpretation:
- **β₀ (intercept)**: Predicted value when all features = 0 (mean after scaling)
- **βᵢ (coefficient)**: Change in prediction per 1-unit change in feature xᵢ
- **After scaling**: 1-unit = 1 standard deviation of original feature

### Example:
`glazing_area` coefficient = +1.26

**Meaning**:
- 1 standard deviation increase in glazing area (≈0.13 m²)
- Increases heating load by 1.26 kWh
- All other features held constant

### Caution with Multicollinearity:
- "All other features held constant" may be impossible
- If features are correlated, changing one changes others
- Coefficient captures **partial effect**, not total effect

---

**Document Version**: 1.0
**Date**: 2025-12-02
**Model**: LinearRegression with StandardScaler
**Performance**: R² = 0.96, RMSE = 1.94 kWh
**Status**: Multicollinearity detected, Ridge regression recommended
