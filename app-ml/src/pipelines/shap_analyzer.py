import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

class SHAPAnalyzer:
    """
    Utility class for SHAP explainability analysis and saving outputs.
    """
    def __init__(self, model, X, model_type='tree', output_dir='shap_outputs'):
        self.model = model
        self.X = X
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self):
        # Auto-select explainer based on model type
        model_class = self.model.__class__.__name__.lower()
        if 'catboost' in model_class or 'xgboost' in model_class or 'lgbm' in model_class or 'randomforest' in model_class or 'tree' in model_class:
            self.explainer = shap.TreeExplainer(self.model)
        elif 'linearregression' in model_class or 'ridge' in model_class or 'lasso' in model_class or 'elasticnet' in model_class:
            self.explainer = shap.Explainer(self.model, self.X)
        else:
            self.explainer = shap.Explainer(self.model, self.X)
        self.shap_values = self.explainer.shap_values(self.X)
        np.save(self.output_dir / 'shap_values.npy', self.shap_values)
        return self.shap_values


    def save_summary_plot(self, plot_type='bar'):
        if self.shap_values is None:
            self.compute_shap_values()
        # Rename feature if needed for consistency
        X_display = self.X.copy()
        if 'cooling_load' in X_display.columns:
            X_display = X_display.rename(columns={'cooling_load': 'heating_load'})
        plt.figure()
        shap.summary_plot(self.shap_values, X_display, plot_type=plot_type, show=False)
        plt.savefig(self.output_dir / f'shap_summary_{plot_type}.png', bbox_inches='tight')
        plt.close()


    def save_beeswarm_plot(self):
        if self.shap_values is None:
            self.compute_shap_values()
        X_display = self.X.copy()
        if 'cooling_load' in X_display.columns:
            X_display = X_display.rename(columns={'cooling_load': 'heating_load'})
        plt.figure()
        shap.summary_plot(self.shap_values, X_display, show=False)
        plt.savefig(self.output_dir / 'shap_beeswarm.png', bbox_inches='tight')
        plt.close()

    def save_dependence_plot(self, feature, interaction_index=None):
        if self.shap_values is None:
            self.compute_shap_values()
        plt.figure()
        shap.dependence_plot(feature, self.shap_values, self.X, interaction_index=interaction_index, show=False)
        plt.savefig(self.output_dir / f'shap_dependence_{feature}.png', bbox_inches='tight')
        plt.close()

    def save_summary_stats(self):
        if self.shap_values is None:
            self.compute_shap_values()
        # Save mean(|SHAP|) per feature
        X_display = self.X.copy()
        if 'cooling_load' in X_display.columns:
            X_display = X_display.rename(columns={'cooling_load': 'heating_load'})
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        stats_df = pd.DataFrame({'feature': X_display.columns, 'mean_abs_shap': mean_abs_shap})
        stats_df = stats_df.sort_values('mean_abs_shap', ascending=False)
        stats_df.to_csv(self.output_dir / 'shap_summary_stats.csv', index=False)

    def run_full_analysis(self, features_for_dependence=None):
        self.compute_shap_values()
        self.save_summary_plot('bar')
        self.save_beeswarm_plot()
        self.save_summary_stats()
        if features_for_dependence:
            for feat in features_for_dependence:
                self.save_dependence_plot(feat)
