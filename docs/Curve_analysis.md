
# Learning Curve Analysis: Comparative Summary

This section summarizes the learning curve analyses for all models tested on the energy efficiency dataset. Use this as a reference for understanding model generalization and overfitting.

---

## CatBoostRegressor
- **Train R²:** Nearly 1.0 at all sizes (fits training data extremely well).
- **Test R²:** Rises quickly, stabilizes above 0.94.
- **Interpretation:** No overfitting, excellent generalization, high-capacity model. Best absolute R².

## Ridge
- **Train R²:** High and stable, just below 1.0.
- **Test R²:** Rises quickly, nearly matches train R² at large sizes.
- **Interpretation:** No overfitting, very small gap, robust and stable. Slightly lower R² than CatBoost, but textbook bias-variance tradeoff.

## Lasso
- **Train R²:** Starts around 0.93, rises to ~0.96.
- **Test R²:** Rises quickly, stabilizes just below train curve (~0.94).
- **Interpretation:** No overfitting, small and stable gap. Slight underfitting compared to Ridge/CatBoost. Good for feature selection and interpretability.

## GradientBoostingRegressor
- **Train R²:** Nearly 1.0 at all sizes.
- **Test R²:** Rises quickly, stabilizes above 0.97.
- **Interpretation:** No overfitting, high and stable test R². Slight gap is normal for this model type.

## RandomForestRegressor
- **Train R²:** Nearly 1.0 at all sizes.
- **Test R²:** Rises quickly, stabilizes above 0.95.
- **Interpretation:** No overfitting, high and stable test R². Slight gap is normal for this model type.

## XGBRegressor
- **Train R²:** Nearly 1.0 at all sizes.
- **Test R²:** Rises quickly, stabilizes above 0.95.
- **Interpretation:** No overfitting, high and stable test R². Slight gap is normal for this model type.

---

## Key Takeaways
- All models generalize well with no significant overfitting.
- CatBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, and XGBRegressor achieve the highest absolute R², making them best for capturing complex patterns.
- Ridge and Lasso are robust, interpretable baselines; Ridge is especially stable, while Lasso is good for feature selection.
- Use both the shape of the learning curve (gap, stability) and absolute R² to choose the best model for your needs.

---

Refer to this document as you review your results or explore further readings!
