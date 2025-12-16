# Phase 2 Workflow: Ridge and Lasso Regression

## Current Branch: `phase-2-ridge-lasso-regression`

---

## Setup Complete ✓

### 1. Documentation Created:
- [Phase2_Ridge_Lasso_Theory.md](Phase2_Ridge_Lasso_Theory.md) - Complete theory guide

### 2. Code Changes:
- **model_factory.py** - Added Ridge and Lasso support
  - Import: `from sklearn.linear_model import Ridge, Lasso`
  - Added to SUPPORTED_MODELS dictionary
  - Added default parameters

### 3. Experiment Script:
- **experiment_ridge_lasso.py** - Automated testing script
  - Tests Ridge with α = [0.01, 0.1, 1.0, 10, 100]
  - Tests Lasso with α = [0.001, 0.01, 0.1, 1.0, 10]
  - Compares with LinearRegression baseline
  - Analyzes coefficient changes
  - Checks if multicollinearity issues are fixed

---

## How to Run Experiments

### Option 1: Run Full Automated Experiment (Recommended)

```bash
python app-ml/entrypoint/experiment_ridge_lasso.py
```

This will:
1. Train LinearRegression (baseline)
2. Train 5 Ridge models with different α values
3. Train 5 Lasso models with different α values
4. Generate comparison table
5. Show best models
6. Compare coefficients
7. Check if wrong signs are fixed

**Expected output**: Comparison table showing performance and coefficient analysis

---

### Option 2: Manual Testing (One Model at a Time)

#### Test Ridge Regression:

Edit `config/config.yaml`:
```yaml
training:
  model:
    type: Ridge
    params:
      alpha: 1.0
      fit_intercept: true
```

Then run:
```bash
python app-ml/entrypoint/train.py
python app-ml/entrypoint/analyze_coefficients.py
```

#### Test Lasso Regression:

Edit `config/config.yaml`:
```yaml
training:
  model:
    type: Lasso
    params:
      alpha: 0.1
      fit_intercept: true
      max_iter: 10000
```

Then run:
```bash
python app-ml/entrypoint/train.py
python app-ml/entrypoint/analyze_coefficients.py
```

---

## What to Look For

### Ridge Regression (Expected Outcomes):

1. **Performance**: Test R² should be close to 0.96 (within 1-2%)
2. **Coefficient shrinkage**: All coefficients smaller than LinearRegression
3. **Sign correction**: `roof_area` and `surface_area` might flip from negative to positive
4. **Stability**: Coefficients should be more stable
5. **All features retained**: Non-zero coefs = 9/9

### Lasso Regression (Expected Outcomes):

1. **Performance**: Test R² might drop to 0.93-0.95 (2-3% decrease)
2. **Feature selection**: Some coefficients = 0 (sparse model)
3. **Likely eliminated**: `orientation` (weakest feature)
4. **From correlated pairs, one eliminated**:
   - Either `relative_compactness` OR `surface_area` (r=-0.992)
   - Either `overall_height` OR `roof_area` (r=-0.973)
5. **Simpler model**: Non-zero coefs might be 5-7/9

### Success Criteria:

#### Ridge Success:
- ✓ Test R² ≥ 0.94 (within 2% of baseline)
- ✓ `roof_area` and `surface_area` coefficients closer to 0 or positive
- ✓ Coefficient magnitudes reduced (all shrunk)

#### Lasso Success:
- ✓ Test R² ≥ 0.93 (within 3% of baseline)
- ✓ At least 2-3 features eliminated
- ✓ Remaining features make physical sense
- ✓ Model is simpler and more interpretable

---

## Next Steps After Experiments

### 1. Analyze Results:
- Which α value gives best performance for Ridge?
- Which α value gives best performance for Lasso?
- Are multicollinearity issues fixed?
- Which features does Lasso eliminate?

### 2. Choose Best Model:
- If performance is similar → Choose Lasso (simpler)
- If Lasso drops performance too much → Choose Ridge
- If Ridge doesn't help → Stick with LinearRegression

### 3. Update Config:
```yaml
training:
  model:
    type: Ridge  # or Lasso or LinearRegression
    params:
      alpha: <best_alpha>
      fit_intercept: true
```

### 4. Save Final Model:
```bash
python app-ml/entrypoint/train.py
```

### 5. Document Findings:
- Update Phase2_Findings.md with results
- Create comparison charts
- Explain which model was chosen and why

### 6. Commit and Merge:
```bash
git add .
git commit -m "Phase 2: Ridge and Lasso regression implementation"
git checkout main
git merge phase-2-ridge-lasso-regression
```

---

## Quick Reference

### Alpha (α) Parameter:

**Ridge**:
- α = 0 → LinearRegression (no regularization)
- α = 0.01-0.1 → Light regularization
- α = 1 → Moderate regularization (good starting point)
- α = 10-100 → Heavy regularization

**Lasso**:
- α = 0 → LinearRegression (no regularization)
- α = 0.001-0.01 → Light regularization
- α = 0.1 → Moderate regularization (good starting point)
- α = 1-10 → Heavy regularization (may eliminate all features)

### Key Files:

- **Theory**: `docs/Phase2_Ridge_Lasso_Theory.md`
- **Experiment**: `app-ml/entrypoint/experiment_ridge_lasso.py`
- **Config**: `config/config.yaml`
- **Model Factory**: `app-ml/src/pipelines/model_factory.py`
- **Training**: `app-ml/entrypoint/train.py`
- **Analysis**: `app-ml/entrypoint/analyze_coefficients.py`

---

## Expected Timeline:

1. Run experiments: ~5-10 minutes
2. Analyze results: ~10-15 minutes
3. Choose best model: ~5 minutes
4. Document findings: ~15-20 minutes
5. **Total**: ~35-50 minutes

Ready to run the experiments!
