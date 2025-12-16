"""
Analyze LinearRegression coefficients to understand their meaning and impact.

This script loads the trained model and provides detailed analysis of:
1. Raw coefficient values
2. Interpretation of each coefficient
3. Impact on predictions
4. Practical examples
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


def load_trained_model_and_scaler(config):
    """Load the trained model and get the scaler from training pipeline."""
    from pipelines.postprocessing import PostprocessingPipeline

    postprocessing = PostprocessingPipeline(config)
    model = postprocessing.load_model()

    # We need to recreate the training pipeline to get the scaler
    # This is a bit hacky, but necessary to understand scaled vs unscaled coefficients
    print("\nRecreating training pipeline to get scaler...")
    data_manager = DataManager(config)
    raw_data = data_manager.load_raw_data(format='csv')

    preprocessing = PreprocessingPipeline(config)
    preprocessed_data = preprocessing.run(raw_data)

    feature_eng = FeatureEngineeringPipeline(config)
    engineered_data = feature_eng.run(preprocessed_data)

    training = TrainingPipeline(config)
    df_with_target = training.create_target_variable(engineered_data)
    X, y = training.prepare_features_and_target(df_with_target)
    X_train, X_test, y_train, y_test = training.split_data(X, y)

    return model, training.scaler, training.feature_names, X_train, y_train, X_test, y_test


def analyze_coefficients(model, feature_names, scaler):
    """Analyze and interpret model coefficients."""
    print("\n" + "="*80)
    print("LINEAR REGRESSION COEFFICIENT ANALYSIS")
    print("="*80)

    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    print(f"\n{'='*80}")
    print("1. RAW MODEL PARAMETERS")
    print(f"{'='*80}")
    print(f"\nIntercept (β₀): {intercept:.6f}")
    print("\nThis is the predicted heating_load when ALL scaled features = 0")
    print("(i.e., when all features are at their mean value in original scale)")

    # Create DataFrame for better visualization
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', key=abs, ascending=False)

    print(f"\n{'='*80}")
    print("2. COEFFICIENTS (Sorted by Absolute Value)")
    print(f"{'='*80}")
    print("\nThese are for SCALED features (mean=0, std=1)")
    print("\nFormula: heating_load = β₀ + Σ(βᵢ × feature_scaled_i)\n")

    for idx, row in coef_df.iterrows():
        feature = row['feature']
        coef = row['coefficient']
        sign = "increases" if coef > 0 else "decreases"

        print(f"{feature:30s} β = {coef:8.4f}")
        print(f"  → 1 std increase in {feature} {sign} heating_load by {abs(coef):.4f} kWh")
        print()

    return coef_df, intercept


def explain_coefficient_meaning(coef_df):
    """Provide physical interpretation of coefficients."""
    print(f"\n{'='*80}")
    print("3. PHYSICAL INTERPRETATION")
    print(f"{'='*80}")

    interpretations = {
        'cooling_load': {
            'sign': 'positive' if coef_df[coef_df['feature']=='cooling_load']['coefficient'].values[0] > 0 else 'negative',
            'meaning': 'Buildings with high cooling needs also have high heating needs',
            'physics': 'Both loads reflect thermal inefficiency - poor insulation causes both heat loss (winter) and heat gain (summer)'
        },
        'overall_height': {
            'sign': 'positive' if coef_df[coef_df['feature']=='overall_height']['coefficient'].values[0] > 0 else 'negative',
            'meaning': 'Taller buildings require more heating',
            'physics': 'Larger volume = more air to heat. Also affects stack effect and surface-to-volume ratio'
        },
        'glazing_area': {
            'sign': 'positive' if coef_df[coef_df['feature']=='glazing_area']['coefficient'].values[0] > 0 else 'negative',
            'meaning': 'More windows = higher heating load',
            'physics': 'Glass is thermally inefficient (R-value ~1 vs wall R-value ~20). More glazing = more heat loss'
        },
        'relative_compactness': {
            'sign': 'positive/negative',
            'meaning': 'Building shape efficiency affects heating',
            'physics': 'Compact shapes (high value) minimize surface area per volume, reducing heat loss'
        },
        'surface_area': {
            'sign': 'positive' if coef_df[coef_df['feature']=='surface_area']['coefficient'].values[0] > 0 else 'negative',
            'meaning': 'Larger envelope = more heating needed',
            'physics': 'Heat loss is proportional to surface area (Q = U × A × ΔT)'
        },
        'wall_area': {
            'sign': 'positive' if coef_df[coef_df['feature']=='wall_area']['coefficient'].values[0] > 0 else 'negative',
            'meaning': 'More wall surface = more heat loss',
            'physics': 'Walls are the primary heat transfer surface (conduction + convection)'
        },
        'roof_area': {
            'sign': 'positive' if coef_df[coef_df['feature']=='roof_area']['coefficient'].values[0] > 0 else 'negative',
            'meaning': 'Larger roof = more heating load',
            'physics': 'Hot air rises - roof is critical for heat retention'
        },
        'glazing_area_distribution': {
            'sign': 'varies',
            'meaning': 'How windows are distributed affects heating',
            'physics': 'Uniform distribution (4) may be more efficient than concentrated (1-3)'
        },
        'orientation': {
            'sign': 'small effect',
            'meaning': 'Building direction has minimal impact',
            'physics': 'Surprising - may be because dataset is simulations without real solar angles'
        }
    }

    for idx, row in coef_df.iterrows():
        feature = row['feature']
        coef = row['coefficient']

        if feature in interpretations:
            info = interpretations[feature]
            print(f"\n{feature.upper()}")
            print(f"  Coefficient: {coef:.4f}")
            print(f"  Direction: {'↑ Positive (increases heating)' if coef > 0 else '↓ Negative (decreases heating)'}")
            print(f"  Meaning: {info['meaning']}")
            print(f"  Physics: {info['physics']}")


def demonstrate_predictions(model, scaler, feature_names, X_train, y_train):
    """Show how coefficients are used to make predictions."""
    print(f"\n{'='*80}")
    print("4. PREDICTION EXAMPLES")
    print(f"{'='*80}")

    # Pick 3 examples: low, medium, high heating load
    low_idx = y_train.idxmin()
    high_idx = y_train.idxmax()
    median_idx = y_train.sort_values().iloc[len(y_train)//2:len(y_train)//2+1].index[0]

    examples = [
        ('Lowest heating load', low_idx),
        ('Median heating load', median_idx),
        ('Highest heating load', high_idx)
    ]

    intercept = model.intercept_

    for label, idx in examples:
        print(f"\n{'-'*80}")
        print(f"{label.upper()}")
        print(f"{'-'*80}")

        # Get feature values (scaled)
        features_scaled = X_train.loc[idx]

        # Make prediction
        prediction = model.predict(features_scaled.values.reshape(1, -1))[0]
        actual = y_train.loc[idx]

        print(f"\nActual heating load: {actual:.2f} kWh")
        print(f"Predicted heating load: {prediction:.2f} kWh")
        print(f"Error: {prediction - actual:.2f} kWh ({abs(prediction-actual)/actual*100:.1f}%)")

        print(f"\nPrediction breakdown:")
        print(f"  Base (intercept):     {intercept:8.4f} kWh")

        # Show contribution of each feature
        contributions = []
        for i, feature in enumerate(feature_names):
            coef = model.coef_[i]
            feature_val = features_scaled.iloc[i]
            contribution = coef * feature_val
            contributions.append((feature, contribution))

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        total_contribution = 0
        for feature, contribution in contributions[:5]:  # Top 5 contributors
            sign = '+' if contribution >= 0 else ''
            print(f"  {feature:25s} {sign}{contribution:8.4f} kWh")
            total_contribution += contribution

        other_contribution = sum(c for _, c in contributions[5:])
        if len(contributions) > 5:
            sign = '+' if other_contribution >= 0 else ''
            print(f"  {'Others':25s} {sign}{other_contribution:8.4f} kWh")

        print(f"  {'-'*35}")
        print(f"  {'TOTAL':25s} {prediction:8.4f} kWh")


def compare_coefficient_importance(coef_df, model, scaler, X_train):
    """Compare raw coefficients vs scaled importance."""
    print(f"\n{'='*80}")
    print("5. COEFFICIENT vs IMPORTANCE")
    print(f"{'='*80}")

    print("\nWhy we can't just use raw coefficients for importance:")
    print("Coefficients are for SCALED features (mean=0, std=1)")
    print("After scaling, all features have the same std=1")
    print("So the coefficient directly represents the impact of 1-std change\n")

    # Calculate importance (should match what training.py does)
    feature_std = X_train.std()

    # Since features are scaled, std should be ~1.0
    print("Standard deviations of scaled features:")
    for feature in coef_df['feature'][:5]:
        std = feature_std[feature]
        print(f"  {feature:30s} std = {std:.6f}")

    print("\nBecause std ≈ 1.0 for all scaled features,")
    print("coefficient magnitude ≈ importance!")

    print("\nComparison:")
    print(f"{'Feature':30s} {'Coefficient':>12s} {'Importance':>12s}")
    print("-"*60)

    for _, row in coef_df.head(10).iterrows():
        feature = row['feature']
        coef = abs(row['coefficient'])
        importance = coef * feature_std[feature]
        print(f"{feature:30s} {coef:12.4f} {importance:12.4f}")


def investigate_multicollinearity(X_train, feature_names):
    """Check for correlated features that might affect coefficient interpretation."""
    print(f"\n{'='*80}")
    print("6. FEATURE CORRELATIONS (Multicollinearity Check)")
    print(f"{'='*80}")

    print("\nHighly correlated features can make individual coefficients misleading.")
    print("Looking for |correlation| > 0.7...\n")

    corr_matrix = X_train.corr()

    # Find high correlations
    high_corr = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr.append((feature_names[i], feature_names[j], corr))

    if high_corr:
        print("High correlations found:")
        for feat1, feat2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {feat1:25s} ↔ {feat2:25s}  r = {corr:6.3f}")

        print("\n⚠ Warning: These features are highly correlated!")
        print("This means their individual coefficients may be unstable.")
        print("Consider Ridge regression to handle multicollinearity.")
    else:
        print("✓ No high correlations found (all |r| < 0.7)")
        print("Coefficients should be stable and interpretable.")

    # Show cooling_load correlation with everything
    print(f"\n{'-'*80}")
    print("Correlation of cooling_load with other features:")
    print(f"{'-'*80}")
    cooling_corr = corr_matrix['cooling_load'].sort_values(ascending=False)
    for feature, corr in cooling_corr.items():
        if feature != 'cooling_load':
            print(f"  {feature:30s} r = {corr:6.3f}")


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("LOADING DATA AND MODEL")
    print("="*80)

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load model and data
    model, scaler, feature_names, X_train, y_train, X_test, y_test = load_trained_model_and_scaler(config)

    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Training samples: {len(X_train)}")

    # Perform analysis
    coef_df, intercept = analyze_coefficients(model, feature_names, scaler)

    explain_coefficient_meaning(coef_df)

    demonstrate_predictions(model, scaler, feature_names, X_train, y_train)

    compare_coefficient_importance(coef_df, model, scaler, X_train)

    investigate_multicollinearity(X_train, feature_names)

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Coefficients show the change in heating_load for a 1-std change in feature
2. After scaling, coefficient magnitude ≈ feature importance
3. Positive coefficients increase heating load, negative decrease it
4. cooling_load dominates because it captures thermal efficiency
5. Check for multicollinearity when interpreting individual coefficients
6. The model is linear: prediction = intercept + sum(coef × feature)
    """)

    print("\nNext steps:")
    print("  - Try Ridge regression to handle multicollinearity")
    print("  - Remove cooling_load and retrain to see other features emerge")
    print("  - Analyze residuals to check linear regression assumptions")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
