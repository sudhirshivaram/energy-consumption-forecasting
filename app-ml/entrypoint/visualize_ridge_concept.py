"""
Visual explanation of Ridge regression and why α=0.01 has minimal effect.

This script:
1. Shows coefficient changes across different alpha values
2. Visualizes the loss function components (MSE vs Penalty)
3. Explains the math step-by-step with actual numbers
"""

import os
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root)

import pandas as pd
import numpy as np
import yaml
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline


def train_multiple_alphas():
    """Train Ridge with multiple alpha values and collect coefficients."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Alpha values to test
    alphas = [0, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]

    results = []

    for alpha in alphas:
        print(f"\nTraining Ridge with α={alpha}...")

        # Update config
        if alpha == 0:
            config['training']['model']['type'] = 'LinearRegression'
            config['training']['model']['params'] = {'fit_intercept': True}
        else:
            config['training']['model']['type'] = 'Ridge'
            config['training']['model']['params'] = {
                'alpha': alpha,
                'fit_intercept': True
            }

        # Load and prepare data
        data_manager = DataManager(config)
        raw_data = data_manager.load_raw_data(format='csv')

        preprocessing = PreprocessingPipeline(config)
        preprocessed_data = preprocessing.run(raw_data)

        feature_eng = FeatureEngineeringPipeline(config)
        engineered_data = feature_eng.run(preprocessed_data)

        # Train model
        training = TrainingPipeline(config)
        df_with_target = training.create_target_variable(engineered_data)
        X, y = training.prepare_features_and_target(df_with_target)
        X_train, X_test, y_train, y_test = training.split_data(X, y)

        model = training.train_model(X_train, y_train, X_test, y_test)

        # Evaluate
        metrics = training.evaluate_model(model, X_train, y_train, X_test, y_test)

        # Store results
        coef_dict = {f: c for f, c in zip(training.feature_names, model.coef_)}
        coef_dict['alpha'] = alpha
        coef_dict['test_r2'] = metrics['test']['R2']
        coef_dict['test_rmse'] = metrics['test']['RMSE']
        coef_dict['intercept'] = model.intercept_

        results.append(coef_dict)

    return pd.DataFrame(results), training.feature_names


def calculate_loss_components(coefficients_df, feature_names):
    """Calculate MSE and penalty components for each alpha."""
    print("\n" + "="*80)
    print("LOSS FUNCTION BREAKDOWN: MSE vs Penalty")
    print("="*80)
    print("\nRidge Loss = MSE + α × Σ(βⱼ²)")
    print("           ↑         ↑")
    print("     prediction   penalty")
    print("       error      (shrinkage)")

    print("\n" + "-"*80)
    print(f"{'Alpha':>10s} {'MSE':>12s} {'Penalty':>12s} {'Total Loss':>12s} {'Penalty %':>12s} {'Test R²':>10s}")
    print("-"*80)

    for _, row in coefficients_df.iterrows():
        alpha = row['alpha']
        test_rmse = row['test_rmse']
        mse = test_rmse ** 2

        # Calculate sum of squared coefficients
        coef_squared_sum = sum([row[f]**2 for f in feature_names])

        # Calculate penalty
        penalty = alpha * coef_squared_sum

        # Total loss
        total_loss = mse + penalty

        # Penalty as percentage of MSE
        if mse > 0:
            penalty_pct = (penalty / mse) * 100
        else:
            penalty_pct = 0

        model_type = "LinearReg" if alpha == 0 else "Ridge"

        print(f"{alpha:10.3f} {mse:12.4f} {penalty:12.4f} {total_loss:12.4f} {penalty_pct:11.1f}% {row['test_r2']:10.6f}")

    print("-"*80)
    print("\nKey insight: When penalty is < 10% of MSE, regularization has minimal effect")
    print("            When penalty is > 100% of MSE, regularization dominates\n")


def show_coefficient_changes(coefficients_df, feature_names):
    """Show how each coefficient changes with alpha."""
    print("\n" + "="*80)
    print("COEFFICIENT CHANGES ACROSS ALPHA VALUES")
    print("="*80)

    # Get baseline (alpha=0)
    baseline = coefficients_df[coefficients_df['alpha'] == 0].iloc[0]

    for feature in feature_names:
        print(f"\n{feature.upper()}")
        print("-" * 60)
        print(f"{'Alpha':>10s} {'Coefficient':>15s} {'Change from Baseline':>25s} {'% Change':>12s}")
        print("-" * 60)

        baseline_coef = baseline[feature]

        for _, row in coefficients_df.iterrows():
            alpha = row['alpha']
            coef = row[feature]
            change = coef - baseline_coef
            pct_change = (change / baseline_coef * 100) if baseline_coef != 0 else 0

            model_type = "LinearReg" if alpha == 0 else f"Ridge"

            print(f"{alpha:10.3f} {coef:15.6f} {change:+15.6f} {pct_change:+11.2f}%")


def explain_step_by_step():
    """Step-by-step explanation with actual numbers."""
    print("\n" + "="*80)
    print("STEP-BY-STEP: WHY α=0.01 HAS MINIMAL EFFECT")
    print("="*80)

    print("""
STEP 1: Understanding the Ridge Optimization
--------------------------------------------
Ridge tries to minimize:

    Loss = Σ(yᵢ - ŷᵢ)² + α × Σ(βⱼ²)
           ↑                ↑
         MSE           penalty term

- MSE: How well the model fits the data (lower = better predictions)
- Penalty: Discourages large coefficients (controlled by α)

The model finds coefficients that balance these two objectives.

STEP 2: Our LinearRegression Coefficients
-----------------------------------------
From our model (α=0, no penalty):

    cooling_load:              6.8678
    overall_height:            2.0804
    glazing_area:              1.2624
    relative_compactness:     -1.0866
    roof_area:                -0.9262
    wall_area:                 0.6953
    surface_area:             -0.6029
    glazing_area_distribution: 0.2894
    orientation:              -0.0803

STEP 3: Calculate Sum of Squared Coefficients
--------------------------------------------
Σ(βⱼ²) = 6.8678² + 2.0804² + ... + 0.0803²

Let's calculate:
    6.8678²  = 47.17
    2.0804²  = 4.33
    1.2624²  = 1.59
    1.0866²  = 1.18
    0.9262²  = 0.86
    0.6953²  = 0.48
    0.6029²  = 0.36
    0.2894²  = 0.08
    0.0803²  = 0.01
    ----------------
    Total    = 56.06

STEP 4: Calculate Penalty for Different Alpha Values
---------------------------------------------------
Penalty = α × 56.06

    α = 0.001  →  Penalty = 0.001 × 56.06 = 0.056
    α = 0.01   →  Penalty = 0.01  × 56.06 = 0.561
    α = 0.1    →  Penalty = 0.1   × 56.06 = 5.61
    α = 1.0    →  Penalty = 1.0   × 56.06 = 56.06
    α = 10     →  Penalty = 10    × 56.06 = 560.6
    α = 100    →  Penalty = 100   × 56.06 = 5606

STEP 5: Compare Penalty to MSE (Prediction Error)
------------------------------------------------
Our model's Test MSE ≈ 3.78

    α = 0.001  →  Penalty = 0.056  (1.5% of MSE)   ← Negligible!
    α = 0.01   →  Penalty = 0.561  (15% of MSE)    ← Still small
    α = 0.1    →  Penalty = 5.61   (148% of MSE)   ← Now matters
    α = 1.0    →  Penalty = 56.06  (1483% of MSE)  ← Strong effect
    α = 10     →  Penalty = 560.6  (14,830% of MSE) ← Very strong
    α = 100    →  Penalty = 5606   (148,300% of MSE) ← Extreme

STEP 6: What This Means for Optimization
----------------------------------------
When penalty is small compared to MSE:
    - Model focuses on minimizing MSE (fitting data well)
    - Penalty has minimal influence on coefficient choices
    - Coefficients barely change from LinearRegression

When penalty is large compared to MSE:
    - Model must balance fitting data vs shrinking coefficients
    - Penalty forces coefficients toward zero
    - Model accepts worse predictions to satisfy penalty

STEP 7: Why α=0.01 Changes Almost Nothing
-----------------------------------------
Total loss with α=0.01:
    Loss = MSE + Penalty
    Loss = 3.78 + 0.56 = 4.34

The penalty (0.56) is only 15% of total loss.

So the model thinks:
    "I'll reduce MSE from 3.78 to 3.75 (-0.03),
     even if it increases penalty from 0.56 to 0.59 (+0.03).

     Total loss: 3.75 + 0.59 = 4.34 (same as before)

     Net benefit: Better predictions, same total loss!"

Result: Coefficients barely change.

STEP 8: When Would Ridge Help More?
----------------------------------
Scenario 1: Coefficients were HUGE (e.g., 100, 200, -150)
    → Σ(βⱼ²) = 100² + 200² + 150² = 72,500
    → Penalty (α=0.01) = 0.01 × 72,500 = 725
    → This is 192× larger than MSE!
    → Ridge would heavily shrink these coefficients

Scenario 2: Model was OVERFITTING badly
    → Train R² = 0.99, Test R² = 0.70
    → Coefficients are unstable (fitting noise)
    → Ridge would stabilize coefficients, improve generalization

Our case: Neither applies!
    → Coefficients are small (all < 7)
    → No overfitting (Train R²=0.97, Test R²=0.96)
    → Ridge provides no benefit

CONCLUSION
----------
α=0.01 is essentially identical to LinearRegression because:

1. ✓ Our coefficients are already small (< 7)
2. ✓ StandardScaler normalized features (natural coefficient scale)
3. ✓ No overfitting (train/test gap is tiny)
4. ✓ Penalty (0.56) is tiny compared to MSE (3.78)

The model already found the "best" coefficients without regularization!
    """)


def main():
    """Main function."""
    print("\n" + "="*80)
    print("VISUALIZING RIDGE REGRESSION CONCEPT")
    print("="*80)

    # Train models with different alphas
    coefficients_df, feature_names = train_multiple_alphas()

    # Show step-by-step explanation
    explain_step_by_step()

    # Calculate loss components
    calculate_loss_components(coefficients_df, feature_names)

    # Show coefficient changes
    show_coefficient_changes(coefficients_df, feature_names)

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Ridge penalty = α × Σ(βⱼ²)
   - When α is small, penalty is weak
   - When coefficients are small, Σ(βⱼ²) is small
   - Both combine to make penalty negligible

2. Model chooses coefficients to minimize: MSE + Penalty
   - If penalty << MSE, focus on MSE (fit data well)
   - If penalty >> MSE, focus on penalty (shrink coefficients)

3. Our case: Penalty (0.56) is 15% of MSE (3.78)
   - Model prioritizes fitting data
   - Coefficients barely change
   - Ridge α=0.01 ≈ LinearRegression

4. When Ridge helps:
   - Large, unstable coefficients (overfitting)
   - Many irrelevant features
   - Need to prevent overfitting

5. Our model doesn't need Ridge because:
   - Already well-behaved coefficients
   - No overfitting
   - StandardScaler normalized everything
   - LinearRegression is already optimal!
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
