"""
Experiment script to systematically test Ridge and Lasso with different alpha values.

This script:
1. Trains Ridge models with different alpha values
2. Trains Lasso models with different alpha values
3. Compares with baseline LinearRegression
4. Analyzes coefficient changes and multicollinearity improvements
5. Generates comparison report
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
from pipelines.model_factory import ModelFactory
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_and_evaluate(model_type, alpha, config):
    """Train a model and return performance metrics and coefficients."""
    print(f"\n{'='*80}")
    print(f"Training {model_type} with alpha={alpha}")
    print(f"{'='*80}")

    # Update config
    config['training']['model']['type'] = model_type
    config['training']['model']['params'] = {
        'alpha': alpha,
        'fit_intercept': True
    }
    if model_type == 'Lasso':
        config['training']['model']['params']['max_iter'] = 10000  # Lasso may need more iterations

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
    train_metrics = metrics['train']
    test_metrics = metrics['test']

    # Get coefficients
    coefficients = pd.DataFrame({
        'feature': training.feature_names,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    # Count non-zero coefficients (for Lasso sparsity)
    non_zero_coefs = (np.abs(model.coef_) > 1e-5).sum()

    return {
        'model_type': model_type,
        'alpha': alpha,
        'train_r2': train_metrics['R2'],
        'test_r2': test_metrics['R2'],
        'train_rmse': train_metrics['RMSE'],
        'test_rmse': test_metrics['RMSE'],
        'train_mae': train_metrics['MAE'],
        'test_mae': test_metrics['MAE'],
        'intercept': model.intercept_,
        'coefficients': coefficients,
        'non_zero_coefs': non_zero_coefs,
        'total_coefs': len(model.coef_)
    }


def train_baseline(config):
    """Train baseline LinearRegression for comparison."""
    print(f"\n{'='*80}")
    print(f"Training Baseline: LinearRegression")
    print(f"{'='*80}")

    # Update config
    config['training']['model']['type'] = 'LinearRegression'
    config['training']['model']['params'] = {'fit_intercept': True}

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
    train_metrics = metrics['train']
    test_metrics = metrics['test']

    # Get coefficients
    coefficients = pd.DataFrame({
        'feature': training.feature_names,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    return {
        'model_type': 'LinearRegression',
        'alpha': 0.0,
        'train_r2': train_metrics['R2'],
        'test_r2': test_metrics['R2'],
        'train_rmse': train_metrics['RMSE'],
        'test_rmse': test_metrics['RMSE'],
        'train_mae': train_metrics['MAE'],
        'test_mae': test_metrics['MAE'],
        'intercept': model.intercept_,
        'coefficients': coefficients,
        'non_zero_coefs': len(model.coef_),
        'total_coefs': len(model.coef_)
    }


def compare_results(results):
    """Create comparison table of all results."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")

    # Performance comparison
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_type'],
            'Alpha': r['alpha'],
            'Train R²': f"{r['train_r2']:.6f}",
            'Test R²': f"{r['test_r2']:.6f}",
            'Test RMSE': f"{r['test_rmse']:.4f}",
            'Non-zero Coefs': f"{r['non_zero_coefs']}/{r['total_coefs']}"
        }
        for r in results
    ])

    print(comparison_df.to_string(index=False))

    # Find best models
    best_ridge = max([r for r in results if r['model_type'] == 'Ridge'],
                     key=lambda x: x['test_r2'], default=None)
    best_lasso = max([r for r in results if r['model_type'] == 'Lasso'],
                     key=lambda x: x['test_r2'], default=None)
    baseline = [r for r in results if r['model_type'] == 'LinearRegression'][0]

    print(f"\n{'='*80}")
    print("BEST MODELS")
    print(f"{'='*80}")
    print(f"\nBaseline (LinearRegression):")
    print(f"  Test R²: {baseline['test_r2']:.6f}, RMSE: {baseline['test_rmse']:.4f}")

    if best_ridge:
        print(f"\nBest Ridge (α={best_ridge['alpha']}):")
        print(f"  Test R²: {best_ridge['test_r2']:.6f}, RMSE: {best_ridge['test_rmse']:.4f}")
        print(f"  R² change: {(best_ridge['test_r2'] - baseline['test_r2']):.6f} ({(best_ridge['test_r2'] - baseline['test_r2'])/baseline['test_r2']*100:.2f}%)")

    if best_lasso:
        print(f"\nBest Lasso (α={best_lasso['alpha']}):")
        print(f"  Test R²: {best_lasso['test_r2']:.6f}, RMSE: {best_lasso['test_rmse']:.4f}")
        print(f"  R² change: {(best_lasso['test_r2'] - baseline['test_r2']):.6f} ({(best_lasso['test_r2'] - baseline['test_r2'])/baseline['test_r2']*100:.2f}%)")
        print(f"  Features eliminated: {baseline['total_coefs'] - best_lasso['non_zero_coefs']}/{baseline['total_coefs']}")

    return baseline, best_ridge, best_lasso


def compare_coefficients(baseline, ridge, lasso):
    """Compare coefficients across models."""
    print(f"\n{'='*80}")
    print("COEFFICIENT COMPARISON")
    print(f"{'='*80}\n")

    # Merge coefficients
    baseline_coefs = baseline['coefficients'].rename(columns={'coefficient': 'LinearRegression'})

    coef_comparison = baseline_coefs.copy()

    if ridge:
        ridge_coefs = ridge['coefficients'].set_index('feature')['coefficient']
        coef_comparison['Ridge'] = coef_comparison['feature'].map(ridge_coefs)

    if lasso:
        lasso_coefs = lasso['coefficients'].set_index('feature')['coefficient']
        coef_comparison['Lasso'] = coef_comparison['feature'].map(lasso_coefs)

    print(coef_comparison.to_string(index=False))

    # Check if wrong signs are fixed
    print(f"\n{'='*80}")
    print("SIGN ANALYSIS (Checking if multicollinearity artifacts are fixed)")
    print(f"{'='*80}\n")

    problematic_features = ['roof_area', 'surface_area']
    for feature in problematic_features:
        baseline_sign = np.sign(baseline_coefs[baseline_coefs['feature'] == feature]['LinearRegression'].values[0])

        print(f"{feature}:")
        print(f"  LinearRegression: {baseline_coefs[baseline_coefs['feature'] == feature]['LinearRegression'].values[0]:.4f} ({'negative' if baseline_sign < 0 else 'positive'})")

        if ridge:
            ridge_val = ridge_coefs[feature]
            ridge_sign = np.sign(ridge_val)
            print(f"  Ridge (α={ridge['alpha']}): {ridge_val:.4f} ({'negative' if ridge_sign < 0 else 'positive'})")
            if baseline_sign != ridge_sign:
                print(f"    → ✓ Sign flipped! (multicollinearity reduced)")

        if lasso:
            lasso_val = lasso_coefs[feature]
            lasso_sign = np.sign(lasso_val) if abs(lasso_val) > 1e-5 else 0
            if abs(lasso_val) < 1e-5:
                print(f"  Lasso (α={lasso['alpha']}): 0.0000 (eliminated)")
            else:
                print(f"  Lasso (α={lasso['alpha']}): {lasso_val:.4f} ({'negative' if lasso_sign < 0 else 'positive'})")
                if baseline_sign != lasso_sign:
                    print(f"    → ✓ Sign flipped! (multicollinearity reduced)")
        print()


def main():
    """Main experiment function."""
    print("\n" + "="*80)
    print("RIDGE AND LASSO REGRESSION EXPERIMENT")
    print("="*80)

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results = []

    # Train baseline
    baseline_result = train_baseline(config.copy())
    results.append(baseline_result)

    # Ridge experiments with different alphas
    ridge_alphas = [0.01, 0.1, 1.0, 10, 100]
    print(f"\n{'='*80}")
    print(f"RIDGE REGRESSION EXPERIMENTS (α = {ridge_alphas})")
    print(f"{'='*80}")

    for alpha in ridge_alphas:
        result = train_and_evaluate('Ridge', alpha, config.copy())
        results.append(result)

    # Lasso experiments with different alphas
    lasso_alphas = [0.001, 0.01, 0.1, 1.0, 10]
    print(f"\n{'='*80}")
    print(f"LASSO REGRESSION EXPERIMENTS (α = {lasso_alphas})")
    print(f"{'='*80}")

    for alpha in lasso_alphas:
        result = train_and_evaluate('Lasso', alpha, config.copy())
        results.append(result)

    # Compare all results
    baseline, best_ridge, best_lasso = compare_results(results)

    # Compare coefficients
    if best_ridge or best_lasso:
        compare_coefficients(baseline, best_ridge, best_lasso)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("1. Review the comparison table")
    print("2. Check if Ridge/Lasso fixed multicollinearity issues (sign changes)")
    print("3. Decide which model to use for production")
    print("4. Update config.yaml with best model")
    print("5. Retrain and save final model")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
