import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = Path("models/experiments/catboost_optimized_model.cbm")
SHAP_VALUES_PATH = Path("shap_outputs/shap_values.npy")
SHAP_STATS_PATH = Path("shap_outputs/shap_summary_stats.csv")
SHAP_BEESWARM_PATH = Path("shap_outputs/shap_beeswarm.png")
SHAP_BAR_PATH = Path("shap_outputs/shap_summary_bar.png")

# Load model and SHAP outputs
def load_model():
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    return model

def load_shap_values():
    return np.load(SHAP_VALUES_PATH)

def load_shap_stats():
    return pd.read_csv(SHAP_STATS_PATH)

# UI functions
def show_shap_summary():
    return SHAP_BAR_PATH

def show_shap_beeswarm():
    return SHAP_BEESWARM_PATH

def show_shap_stats():
    df = load_shap_stats()
    return df.to_markdown(index=False)

# Gradio UI

def predict_and_explain(X1, X2, X3, X4, X5, X6, X7, X8):
    model = load_model()
    input_df = pd.DataFrame([[X1, X2, X3, X4, X5, X6, X7, X8]], columns=["X1","X2","X3","X4","X5","X6","X7","X8"])
    pred = model.predict(input_df)[0]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    # Force plot as image
    plt.figure(figsize=(8,2))
    shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False)
    plt.tight_layout()
    force_path = "shap_outputs/force_plot_temp.png"
    plt.savefig(force_path, bbox_inches='tight')
    plt.close()
    return pred, force_path

def main():
    with gr.Blocks() as demo:
        gr.Markdown("""
<h1 style='text-align: center; color: #2c3e50; font-size: 2.5em; margin-bottom: 0.2em;'>
    Energy Consumption Forecast
</h1>
<div style='text-align: center; color: #555; font-size: 1.2em; margin-bottom: 1.5em;'>
    Interactive Model & SHAP Explainability Dashboard
</div>
        """)
        gr.Markdown("View global SHAP explanations for the trained CatBoost model.")
        with gr.Tab("Local Prediction & Explanation"):
            gr.Markdown("Enter feature values to get a prediction and local SHAP explanation.")
            with gr.Row():
                X1 = gr.Number(label="X1 (Relative Compactness, -)")
                X2 = gr.Number(label="X2 (Surface Area, m²)")
                X3 = gr.Number(label="X3 (Wall Area, m²)")
                X4 = gr.Number(label="X4 (Roof Area, m²)")
            with gr.Row():
                X5 = gr.Number(label="X5 (Overall Height, m)")
                X6 = gr.Number(label="X6 (Orientation, categorical)")
                X7 = gr.Number(label="X7 (Glazing Area, m²)")
                X8 = gr.Number(label="X8 (Glazing Area Distribution, categorical)")
            predict_btn = gr.Button("Predict & Explain")
            pred_output = gr.Number(label="Predicted Heating Load (Y1, kWh/m²)")
            force_img = gr.Image(label="SHAP Force Plot (Local Explanation)")
            predict_btn.click(
                predict_and_explain,
                inputs=[X1, X2, X3, X4, X5, X6, X7, X8],
                outputs=[pred_output, force_img]
            )
        with gr.Tab("SHAP Summary (Bar)"):
            gr.Image(value=show_shap_summary, label="SHAP Feature Importance (Bar)")
        with gr.Tab("SHAP Beeswarm"):
            gr.Image(value=show_shap_beeswarm, label="SHAP Beeswarm Plot")
        with gr.Tab("SHAP Stats Table"):
            gr.Markdown(show_shap_stats())
        gr.Markdown("---")
        gr.Markdown("For local explanations or custom data, extend this app as needed.")
    demo.launch()

if __name__ == "__main__":
    main()
