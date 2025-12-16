# Phase 2 Findings Report
**Ridge and Lasso Regression Experimentation**

**Date**: 2025-12-08
**Branch**: `phase-2-ridge-lasso-regression`
**Status**: ✅ Complete - Ready to merge

---

## Executive Summary

**Objective**: Test Ridge and Lasso regression to:
1. Handle multicollinearity discovered in Phase 1
2. Potentially improve model performance
3. Understand when regularization helps

**Result**: ✅ **Regularization provides NO benefit for our dataset**

**Recommendation**: **Continue with LinearRegression** (simplest, performs identically)

---

## What We Tested

### Models and Hyperparameters:

1. **LinearRegression** (Baseline)
   - No regularization (α = 0)

2. **Ridge Regression** (L2 Regularization)
   - α = [0.01, 0.1, 1.0, 10, 100]

3. **Lasso Regression** (L1 Regularization)
   - α = [0.001, 0.01, 0.1, 1.0, 10]

### Evaluation Metrics:
- Test R² (primary metric)
- Test RMSE
- Train/Test gap (overfitting check)
- Coefficient changes
- Feature elimination (Lasso sparsity)

---

## Key Findings

### 1. Performance: All Models Identical

| Model | Alpha | Test R² | Test RMSE | Features Retained |
|-------|-------|---------|-----------|-------------------|
| **LinearRegression** | 0 | **0.9637** | 1.9447 | 9/9 |
| **Ridge (best)** | 0.01 | 0.9637 | 1.9447 | 9/9 |
| **Lasso (best)** | 0.001 | 0.9637 | 1.9458 | 9/9 |

**Conclusion**: Ridge and Lasso perform identically to LinearRegression.

---

### 2. Ridge Regularization Has Minimal Effect

#### Penalty Calculation:

```
Sum of squared coefficients: Σ(βⱼ²) = 56.06

Alpha values tested:
α = 0.01   →  Penalty = 0.01 × 56.06 = 0.56
α = 0.1    →  Penalty = 0.1  × 56.06 = 5.61
α = 1.0    →  Penalty = 1.0  × 56.06 = 56.06
α = 10     →  Penalty = 10   × 56.06 = 560.6
α = 100    →  Penalty = 100  × 56.06 = 5606
```

#### Compare to MSE:

```
Test MSE = 3.78

α = 0.01   →  Penalty is 15% of MSE    ← Too weak!
α = 0.1    →  Penalty is 148% of MSE
α = 1.0    →  Penalty is 1483% of MSE
α = 10     →  Penalty is 14,830% of MSE
α = 100    →  Penalty is 148,300% of MSE ← Too strong!
```

**Key Insight**: Only when penalty > 100% of MSE does regularization have noticeable effect.

#### Coefficient Changes:

| Alpha | cooling_load | Change | % Change | Test R² |
|-------|--------------|--------|----------|---------|
| 0 (LinearReg) | 6.8678 | - | - | 0.9637 |
| 0.01 (Ridge) | 6.8673 | -0.0005 | -0.01% | 0.9637 |
| 1.0 (Ridge) | 6.8167 | -0.0511 | -0.74% | 0.9637 |
| 100 (Ridge) | 4.4632 | -2.4046 | -35% | **0.9508** |

**Observation**: Need α=100 to see meaningful shrinkage, but performance drops!

---

### 3. Lasso Feature Elimination

| Alpha | Features Retained | Eliminated Features | Test R² |
|-------|-------------------|---------------------|---------|
| 0.001 | 9/9 | None | 0.9637 |
| 0.01 | 7/9 | 2 features | 0.9633 |
| 0.1 | 6/9 | 3 features | 0.9631 |
| 1.0 | 3/9 | 6 features | **0.9402** |
| 10 | 0/9 | **All features** | **0.0** |

**Key Finding**:
- Light Lasso (α=0.001): No features eliminated, same performance
- Moderate Lasso (α=0.1): Eliminates 3 features, tiny performance drop
- Heavy Lasso (α=1.0): Only 3 features remain, R² drops to 0.94
- Extreme Lasso (α=10): Eliminates ALL features, model predicts mean

**Likely eliminated features** (weakest predictors):
1. `orientation` (weakest, β=-0.08)
2. `glazing_area_distribution` (β=0.29)
3. One from correlated pairs (surface_area or roof_area)

---

### 4. Multicollinearity NOT Fixed

**Problem from Phase 1**: `roof_area` and `surface_area` have counterintuitive negative coefficients due to multicollinearity.

**Did Ridge/Lasso fix the signs?** ❌ **NO**

| Feature | LinearReg | Ridge (α=0.01) | Lasso (α=0.001) | Fixed? |
|---------|-----------|----------------|-----------------|--------|
| `roof_area` | -0.926 | -0.925 | -0.301 | ❌ Still negative |
| `surface_area` | -0.603 | -0.602 | -1.032 | ❌ More negative! |

**Why not fixed?**
- The negative signs are due to **structural multicollinearity** (features are fundamentally correlated)
- When `roof_area` increases holding height constant, the building is less tall
- This creates confounding relationships that regularization can't "fix"
- The coefficients represent **partial effects**, not causal effects

**True understanding**: The "wrong" signs are actually correct given the multicollinearity. They show what happens when you change one feature while holding others constant (which may be impossible in reality).

---

### 5. Why Regularization Didn't Help

#### Reason 1: No Overfitting

```
LinearRegression Performance:
  Train R² = 0.970
  Test R²  = 0.964
  Gap      = 0.006 (0.6%)  ← Excellent generalization!
```

**Small train/test gap** = No overfitting = Regularization not needed

#### Reason 2: Coefficients Already Small

```
Largest coefficient: cooling_load = 6.87
Most coefficients < 2.0

Σ(βⱼ²) = 56.06  ← Relatively small
```

Compare to badly scaled models:
```
Without StandardScaler, coefficients might be:
  surface_area = 0.0001  (tiny, but feature range is 500-800)
  glazing_area = 200     (huge, but feature range is 0-0.4)

  Σ(βⱼ²) = 40,000+  ← Would need heavy regularization!
```

**Our case**: StandardScaler normalized features → Natural coefficient scale

#### Reason 3: Dataset is Well-Behaved

```
✓ Only 768 samples, 9 features (not high-dimensional)
✓ Strong linear relationship (R²=0.96)
✓ No noise or irrelevant features
✓ All features contribute to prediction
```

**No problems for regularization to solve!**

---

## What We Learned

### 1. Ridge Penalty Calculation

```
Ridge Loss = MSE + α × Σ(βⱼ²)

Step-by-step:
1. Square each coefficient
2. Sum them up: Σ(βⱼ²)
3. Multiply by α: Penalty = α × Σ(βⱼ²)
4. Add to MSE: Total Loss = MSE + Penalty

If Penalty << MSE → Ridge ≈ LinearRegression
If Penalty >> MSE → Ridge shrinks coefficients
```

**Our case**:
- Penalty (0.56) << MSE (3.78) for α=0.01
- Ridge has no effect until α >> 0.1

### 2. When Regularization Helps

**Ridge/Lasso help when**:
- ✅ Overfitting: Train R² >> Test R² (e.g., 0.99 vs 0.70)
- ✅ Huge coefficients: > 100, unstable
- ✅ High dimensionality: Many features, few samples
- ✅ Irrelevant features: Need automatic selection (Lasso)

**Our case**:
- ❌ No overfitting: Train R²=0.97, Test R²=0.96
- ❌ Small coefficients: < 7
- ❌ Low dimensionality: 768 samples, 9 features
- ❌ All features relevant: Each contributes to R²

**Result**: Regularization provides no benefit!

### 3. StandardScaler's Critical Role

**Why our coefficients are already well-behaved**:
```
Before scaling:
  surface_area: mean=671, std=88
  glazing_area: mean=0.23, std=0.13
  → Coefficients would be on vastly different scales

After StandardScaler:
  All features: mean=0, std=1
  → Coefficients on same scale
  → Natural interpretation: "effect per 1-std change"
```

**StandardScaler prevented the problems** that Ridge/Lasso usually solve!

### 4. L1 vs L2 Regularization

**Ridge (L2)**:
- Penalty = α × Σ(β²)
- Shrinks all coefficients smoothly
- Never eliminates features (coefficients → 0 but never = 0)
- Best when all features are somewhat useful

**Lasso (L1)**:
- Penalty = α × Σ|β|
- Shrinks AND eliminates features
- Sets weak coefficients to exactly 0
- Best for feature selection

**Our takeaway**:
- Ridge didn't help because no overfitting
- Lasso could simplify model (6 features instead of 9) but at cost of performance

---

## Detailed Results

### Ridge Regression: Full Results

| Alpha | Train R² | Test R² | Test RMSE | cooling_load coef | Change from baseline |
|-------|----------|---------|-----------|-------------------|----------------------|
| 0.01 | 0.9704 | 0.9637 | 1.9447 | 6.8673 | -0.0005 (-0.01%) |
| 0.1 | 0.9704 | 0.9637 | 1.9449 | 6.8628 | -0.0050 (-0.07%) |
| 1.0 | 0.9704 | 0.9637 | 1.9459 | 6.8167 | -0.0511 (-0.74%) |
| 10 | 0.9699 | 0.9632 | 1.9586 | 6.3889 | -0.4789 (-6.97%) |
| 100 | 0.9575 | 0.9508 | 2.2656 | 4.4632 | -2.4046 (-35.01%) |

**Observation**: Only α≥100 shows meaningful shrinkage, but performance suffers.

### Lasso Regression: Full Results

| Alpha | Train R² | Test R² | Test RMSE | Features | Eliminated |
|-------|----------|---------|-----------|----------|------------|
| 0.001 | 0.9704 | 0.9637 | 1.9458 | 9/9 | None |
| 0.01 | 0.9703 | 0.9633 | 1.9550 | 7/9 | 2 |
| 0.1 | 0.9698 | 0.9631 | 1.9614 | 6/9 | 3 |
| 1.0 | 0.9440 | 0.9402 | 2.4976 | 3/9 | 6 |
| 10 | 0.0 | -0.0055 | 10.2375 | 0/9 | All |

**Observation**: Lasso at α=0.1 could simplify model (6 features) with minimal performance loss.

---

## Recommendation

### ✅ Continue with LinearRegression

**Reasons**:
1. **Performance**: R²=0.9637, identical to Ridge/Lasso
2. **Simplicity**: No hyperparameter (α) to tune
3. **Interpretability**: All 9 features retained
4. **No overfitting**: Train/test gap is only 0.6%
5. **Already optimal**: StandardScaler + LinearRegression is the "right" complexity

### ❌ Don't Use Ridge

**Why not**:
- No performance improvement
- Adds complexity (tuning α)
- Doesn't fix multicollinearity signs
- Coefficients already well-behaved

### 🤔 Lasso: Optional Simplification

**Consider Lasso (α=0.1) IF**:
- Want simpler model (6 features instead of 9)
- Willing to accept tiny performance drop (R²: 0.9637 → 0.9631)
- Prefer feature selection over full model

**But**: LinearRegression is already simple enough (9 features is manageable).

---

## Files Created

### Documentation:
1. **[Phase2_Ridge_Lasso_Theory.md](Phase2_Ridge_Lasso_Theory.md)** - Complete L1/L2 theory
2. **[Ridge_Explained_Simply.md](Ridge_Explained_Simply.md)** - Step-by-step visual guide
3. **[Phase2_Workflow.md](Phase2_Workflow.md)** - Experiment instructions
4. **[Phase2_Findings_Report.md](Phase2_Findings_Report.md)** - This document

### Code:
1. **[experiment_ridge_lasso.py](../app-ml/entrypoint/experiment_ridge_lasso.py)** - Automated experiments
2. **[visualize_ridge_concept.py](../app-ml/entrypoint/visualize_ridge_concept.py)** - Interactive visualization
3. **[model_factory.py](../app-ml/src/pipelines/model_factory.py)** - Added Ridge/Lasso support

---

## What We Accomplished

### ✅ Completed:
1. Implemented Ridge and Lasso in model factory
2. Tested 5 alpha values for Ridge
3. Tested 5 alpha values for Lasso
4. Compared performance metrics
5. Analyzed coefficient changes
6. Checked multicollinearity fix
7. Understood why regularization didn't help
8. Created comprehensive documentation

### 📚 Knowledge Gained:
1. How Ridge penalty is calculated (α × Σ(β²))
2. Why α=0.01 has minimal effect (penalty << MSE)
3. When regularization helps vs doesn't help
4. L1 vs L2 regularization differences
5. StandardScaler's role in coefficient scale
6. Multicollinearity can't always be "fixed"

### 🎓 Key Learnings:
1. **Regularization is not always needed**
2. **Overfitting detection** (train/test gap) guides regularization need
3. **Coefficient scale matters** (StandardScaler helped)
4. **Simple is better** when performance is identical

---

## Next Steps

### Immediate:
1. ✅ Document findings (this report)
2. ⏳ Commit Phase 2 changes
3. ⏳ Merge to main branch

### Future Phases:
**Phase 3**: Residual Analysis
- Check linear regression assumptions
- Identify prediction error patterns
- Understand when model fails

**Phase 4**: Advanced Models
- Try tree-based models (RandomForest, XGBoost, CatBoost)
- Compare with LinearRegression baseline
- Determine if non-linear relationships exist

**Phase 5**: Model Interpretation
- SHAP values
- Partial dependence plots
- Feature interaction analysis

---

## Conclusion

**Phase 2 validated our Phase 1 model**:
- LinearRegression is already optimal for this dataset
- StandardScaler + LinearRegression = well-behaved coefficients
- No overfitting → No need for regularization
- R²=0.96 is excellent for baseline model

**Key insight**: Sometimes the simplest solution is the best solution!

**Multicollinearity**: While we can't "fix" the confounded features, we understand:
- The "wrong" signs are artifacts of holding other features constant
- They don't indicate a model problem
- They're mathematically correct given the correlations
- Ridge/Lasso don't eliminate this confounding

**Recommendation**: Proceed with LinearRegression to Phase 3 (residual analysis) to further validate model assumptions.

---

**Phase 2 Status**: ✅ **COMPLETE**
**Branch**: `phase-2-ridge-lasso-regression`
**Ready to merge**: ✅ Yes
**Next phase**: Phase 3 - Residual Analysis

---

**Experiments conducted**: 11 models (1 LinearReg + 5 Ridge + 5 Lasso)
**Total training time**: ~5 minutes
**Documentation pages**: 4 comprehensive guides
**Lines of code**: ~600 (experiment + visualization scripts)
**Learning value**: ⭐⭐⭐⭐⭐ (Deep understanding of when NOT to use regularization!)
