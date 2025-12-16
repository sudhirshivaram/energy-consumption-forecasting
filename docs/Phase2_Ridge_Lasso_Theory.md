# Phase 2: Ridge and Lasso Regression Theory
**L1 and L2 Regularization Explained**

---

## Why Do We Need Regularization?

### Problem Discovered in Phase 1:

From our LinearRegression analysis, we found **severe multicollinearity**:
- `relative_compactness ↔ surface_area`: r = -0.992
- `roof_area ↔ overall_height`: r = -0.973
- `overall_height ↔ cooling_load`: r = +0.895

**Consequences**:
1. **Unstable coefficients**: Small data changes → Large coefficient changes
2. **Wrong signs**: `roof_area` and `surface_area` have counterintuitive negative coefficients
3. **High variance**: Model is overly sensitive to training data
4. **Poor generalization**: May not work well on new data

**Solution**: Regularization techniques like Ridge and Lasso

---

## What is Regularization?

**Regularization** adds a penalty to the loss function to discourage large coefficients.

### Standard LinearRegression (OLS):
```
minimize: Σ(yᵢ - ŷᵢ)²
```
Only minimizes prediction error

### Regularized Regression:
```
minimize: Σ(yᵢ - ŷᵢ)² + penalty(coefficients)
```
Minimizes both prediction error AND coefficient magnitude

**Trade-off**: Slight increase in bias (training error) for large decrease in variance (generalization)

---

## Ridge Regression (L2 Regularization)

### Formula:
```
minimize: Σ(yᵢ - ŷᵢ)² + α × Σ(βⱼ²)
           ↑                ↑
       MSE loss       L2 penalty
```

Where:
- `α` (alpha) = regularization strength (hyperparameter)
- `βⱼ` = coefficient for feature j
- `Σ(βⱼ²)` = sum of squared coefficients

### What L2 Penalty Does:

**Shrinks coefficients toward zero** (but never exactly zero)

Example:
- Without penalty: β = [10, -8, 0.1, 6]
- With L2 penalty: β = [4, -3, 0.05, 2]  ← All shrunk

### Key Properties:

1. **All features retained**: Never sets coefficients to exactly 0
2. **Shrinks correlated features together**: Distributes impact across correlated features
3. **Better with multicollinearity**: Reduces variance from correlated predictors
4. **Continuous solution**: Differentiable, unique solution

### When to Use Ridge:

- ✅ **Multicollinearity present** (our case!)
- ✅ Want to **keep all features**
- ✅ Believe **all features are somewhat useful**
- ✅ Need **stable, interpretable coefficients**

### Alpha (α) Parameter:

- `α = 0`: Same as LinearRegression (no penalty)
- `α → ∞`: All coefficients → 0 (only intercept remains)
- **Typical range**: 0.01 to 100

**Finding optimal α**: Cross-validation (we'll use GridSearch or try multiple values)

---

## Lasso Regression (L1 Regularization)

### Formula:
```
minimize: Σ(yᵢ - ŷᵢ)² + α × Σ|βⱼ|
           ↑                ↑
       MSE loss       L1 penalty
```

Where:
- `α` (alpha) = regularization strength
- `|βⱼ|` = absolute value of coefficient
- `Σ|βⱼ|` = sum of absolute coefficients

### What L1 Penalty Does:

**Shrinks coefficients toward zero AND sets many to exactly zero**

Example:
- Without penalty: β = [10, -8, 0.1, 6]
- With L1 penalty: β = [5, -4, 0, 3]  ← Third coefficient eliminated!

### Key Properties:

1. **Feature selection**: Sets unimportant coefficients to exactly 0
2. **Sparse models**: Only keeps most important features
3. **Handles multicollinearity differently**: Picks one feature from correlated group, zeros others
4. **No unique solution**: If features are highly correlated, arbitrary which one is kept

### When to Use Lasso:

- ✅ **Want automatic feature selection**
- ✅ Believe **many features are irrelevant**
- ✅ Need **sparse, simple model**
- ✅ **Interpretability** more important than perfect accuracy

### Alpha (α) Parameter:

- `α = 0`: Same as LinearRegression
- `α → ∞`: All coefficients → 0
- **Typical range**: 0.001 to 10

**Note**: Lasso is more sensitive to α than Ridge

---

## Ridge vs Lasso: Visual Comparison

### Coefficient Shrinkage Pattern:

```
Original coefficients: [10, 8, 6, 4, 2, 1, 0.5, 0.1]

Ridge (α=1):           [8, 6.5, 5, 3.5, 1.8, 0.9, 0.45, 0.09]
                       ↑ All shrunk proportionally, none eliminated

Lasso (α=1):           [7, 5, 3, 1, 0, 0, 0, 0]
                       ↑ Smaller coefficients eliminated
```

### Handling Correlated Features:

**Scenario**: `feature_A` and `feature_B` are highly correlated (r=0.95)

**LinearRegression**:
- β_A = 10, β_B = -8  ← Unstable, counterintuitive signs
- Small data change → β_A = 5, β_B = 3  ← Large swing!

**Ridge**:
- β_A = 4, β_B = 4  ← Shrinks both, distributes weight
- Small data change → β_A = 3.5, β_B = 4.5  ← Stable!

**Lasso**:
- β_A = 6, β_B = 0  ← Picks one, eliminates other
- Small data change → β_A = 0, β_B = 6  ← May flip which is kept

---

## Mathematical Intuition

### Why Does L1 Create Sparsity?

**Geometric interpretation**:

Imagine coefficients as a point in 2D space (β₁, β₂):

**L2 constraint** (Ridge): β₁² + β₂² ≤ t
- Forms a **circle**
- Solution usually NOT on axes
- Both β₁ and β₂ are non-zero

**L1 constraint** (Lasso): |β₁| + |β₂| ≤ t
- Forms a **diamond**
- Diamond has **corners on axes**
- Solution often hits a corner → one coefficient = 0

```
      β₂
       ↑
       |     ╱  ╲     ← L1 constraint (diamond)
       |    ╱    ╲
       |   ╱      ╲
-------+--●--------●---→ β₁
       | ╱  ⚫      ╲
       |╱    ↑      ╲
      ╱ ╲   hits corner (β₂=0)
```

### Why Does L2 Shrink Smoothly?

**Circle has no corners** → Solution can land anywhere on circle → Both coefficients non-zero but shrunk

---

## Expected Outcomes for Our Dataset

### Current Issues (LinearRegression):
1. `cooling_load` dominates: 6.87 importance
2. `roof_area` has wrong sign: -0.93 (should be positive)
3. `surface_area` has wrong sign: -0.60 (should be positive)
4. High multicollinearity affects interpretation

### Ridge Regression Predictions:

**Expected changes**:
- ✅ **Coefficients will shrink**: All will be smaller in magnitude
- ✅ **Wrong signs may flip**: `roof_area` and `surface_area` might become positive
- ✅ **More stable**: Less sensitive to train/test split
- ✅ **cooling_load still dominates**: But less extreme
- ⚠️ **Slight performance drop**: R² might decrease from 0.96 to ~0.95

**Why it helps**:
- Reduces impact of multicollinearity
- Distributes weight among correlated features
- More interpretable coefficients

### Lasso Regression Predictions:

**Expected changes**:
- ✅ **Feature elimination**: Some features will have β = 0
- ✅ **Likely to eliminate**: `orientation` (weakest), possibly others
- ✅ **From correlated pairs, picks one**:
  - Might keep `relative_compactness`, drop `surface_area`
  - Might keep `overall_height`, drop `roof_area`
- ⚠️ **Performance drop**: R² might decrease to ~0.94-0.95
- ⚠️ **Arbitrary selection**: Which correlated feature is kept may vary

**Why it helps**:
- Automatic feature selection
- Simpler, more interpretable model
- Removes redundant features

---

## ElasticNet: Combining Ridge and Lasso

### Formula:
```
minimize: Σ(yᵢ - ŷᵢ)² + α × (λ × Σ|βⱼ| + (1-λ) × Σ(βⱼ²))
                              ↑              ↑
                           L1 penalty    L2 penalty
```

Where:
- `α` = overall regularization strength
- `λ` (l1_ratio) = balance between L1 and L2 (0 to 1)
  - `λ = 0`: Pure Ridge
  - `λ = 1`: Pure Lasso
  - `λ = 0.5`: 50/50 mix

### When to Use ElasticNet:

- ✅ **Groups of correlated features**: Want to select groups, not individuals
- ✅ **Want both shrinkage and selection**
- ✅ **Lasso is too unstable** (many correlated features)
- ✅ **Ridge doesn't simplify enough**

**For our dataset**: May not be necessary, but worth trying if Ridge/Lasso don't satisfy needs

---

## Implementation Plan

### Step 1: Ridge Regression

1. **Choose α values to test**: [0.01, 0.1, 1.0, 10, 100]
2. **Train models** with each α
3. **Compare performance**: R², RMSE vs LinearRegression
4. **Analyze coefficients**: Check if wrong signs are fixed
5. **Select best α**: Based on test performance

### Step 2: Lasso Regression

1. **Choose α values to test**: [0.001, 0.01, 0.1, 1.0, 10]
2. **Train models** with each α
3. **Check which features are eliminated** (β = 0)
4. **Compare performance**: R², RMSE vs LinearRegression
5. **Analyze sparsity**: How many features remain?

### Step 3: Comparison

Create comparison table:

| Model          | α    | R²   | RMSE | # Features | Multicollinearity Issue? |
|----------------|------|------|------|------------|--------------------------|
| LinearReg      | N/A  | 0.96 | 1.94 | 9          | Yes (severe)             |
| Ridge          | ?    | ?    | ?    | 9          | ?                        |
| Lasso          | ?    | ?    | ?    | ?          | ?                        |

### Step 4: Coefficient Analysis

For best Ridge and Lasso models:
- Run `analyze_coefficients.py`
- Compare coefficient values
- Check if wrong signs are fixed
- Document interpretation improvements

---

## Success Criteria

### Ridge Success:
- ✅ More stable coefficients (less sensitive to data)
- ✅ Wrong signs fixed (`roof_area`, `surface_area` become positive or close to 0)
- ✅ Performance within 1-2% of LinearRegression (R² ≥ 0.94)
- ✅ Reduced coefficient variance

### Lasso Success:
- ✅ At least 2-3 features eliminated (simpler model)
- ✅ Performance within 2-3% of LinearRegression (R² ≥ 0.93)
- ✅ Retained features make physical sense
- ✅ Model is interpretable (fewer features = clearer story)

### Overall Success:
- ✅ Understand trade-off between bias and variance
- ✅ Learn how regularization handles multicollinearity
- ✅ Can explain when to use Ridge vs Lasso
- ✅ Document findings for future reference

---

## Code Changes Required

### 1. Update config.yaml:

Already model-agnostic! Just change:
```yaml
model:
  type: Ridge  # or Lasso
  params:
    alpha: 1.0
    fit_intercept: true
```

### 2. Update model_factory.py:

Check if Ridge and Lasso are already supported:
- `Ridge` from `sklearn.linear_model`
- `Lasso` from `sklearn.linear_model`

If not, add them to `SUPPORTED_MODELS` dict.

### 3. Training pipeline:

No changes needed! Ridge and Lasso have same interface as LinearRegression:
- `model.fit(X_train, y_train)`
- `model.predict(X_test)`
- `model.coef_` and `model.intercept_`

### 4. Coefficient analysis:

Works as-is! Ridge and Lasso both have `coef_` attribute.

---

## Experiment Design

### Experiment 1: Ridge with Different α Values

```bash
# α = 0.01 (very light regularization)
# α = 0.1
# α = 1.0  (default, recommended start)
# α = 10
# α = 100 (very heavy regularization)
```

**Expected**: As α increases, coefficients shrink, performance slightly decreases

### Experiment 2: Lasso with Different α Values

```bash
# α = 0.001 (very light)
# α = 0.01
# α = 0.1 (recommended start)
# α = 1.0
# α = 10 (very heavy, might eliminate all features)
```

**Expected**: As α increases, more features eliminated

### Experiment 3: Best Ridge vs Best Lasso vs LinearRegression

Compare the best performing α for each method.

---

## Learning Objectives

By the end of Phase 2, you will understand:

1. **What is regularization?** Penalty added to loss function
2. **Why regularization?** Reduce overfitting, handle multicollinearity
3. **L2 (Ridge)**: Shrinks all coefficients smoothly
4. **L1 (Lasso)**: Shrinks + feature selection (sets some to 0)
5. **Bias-variance trade-off**: Regularization increases bias, decreases variance
6. **How to choose α**: Cross-validation or manual search
7. **When to use Ridge**: Multicollinearity, want all features
8. **When to use Lasso**: Want feature selection, simpler model
9. **Coefficient interpretation**: How regularization affects meaning
10. **Multicollinearity solution**: Practical experience fixing unstable coefficients

---

## References

### Scikit-learn Documentation:
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

### Key Concepts:
- **Regularization**: Adding penalty to prevent overfitting
- **L1 norm**: Sum of absolute values (|β₁| + |β₂| + ...)
- **L2 norm**: Sum of squares (β₁² + β₂² + ...)
- **Sparsity**: Having many zero coefficients (Lasso property)
- **Hyperparameter**: Parameter not learned from data (α in this case)

---

**Next Steps**:
1. Implement Ridge regression
2. Train with different α values
3. Analyze coefficients and performance
4. Repeat for Lasso
5. Compare and document findings

Ready to implement? Let's start with Ridge!
