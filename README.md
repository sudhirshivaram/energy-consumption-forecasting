# 🏠 Energy Consumption Forecasting with SHAP Explainability

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green.svg)](https://github.com/slundberg/shap)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning pipeline for predicting building heating loads with comprehensive model explainability using SHAP (SHapley Additive exPlanations).

**🔗 [Try Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/sudhirshivaram/energy-consumption-forecasting)**

## 📊 Project Overview

This project builds an interpretable machine learning system to forecast energy consumption in buildings based on architectural and thermal properties. The pipeline achieves **96% accuracy (R²=0.96)** using a simple LinearRegression baseline, validated through extensive experimentation with regularization techniques.

### Key Highlights

- ✅ **High Performance**: R²=0.96, RMSE=1.94 kWh
- ✅ **Explainable AI**: SHAP analysis for model interpretability
- ✅ **Interactive Dashboard**: Gradio UI for exploring predictions
- ✅ **Production-Ready**: Model-agnostic pipeline with comprehensive testing
- ✅ **Well-Documented**: Extensive guides and findings reports

---

## 🎯 Problem Statement

Predicting building energy consumption is crucial for:
- Designing energy-efficient buildings
- Reducing carbon footprint
- Optimizing HVAC systems
- Meeting regulatory standards

**Challenge**: Build an accurate, interpretable model to predict heating load based on building characteristics.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sudhirshivaram/energy-consumption-forecasting.git
cd energy-consumption-forecasting

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# Train LinearRegression baseline
python app-ml/entrypoint/train.py

# Output: Trained model saved to models/prod/
```

### Run SHAP Dashboard

```bash
# Launch interactive Gradio dashboard
python app-ml/entrypoint/gradio_shap_dashboard.py

# Open browser at http://localhost:7860
```

---

## 📈 Model Performance

### Baseline Model: LinearRegression

| Metric | Training | Test | Status |
|--------|----------|------|--------|
| **R²** | 0.970 | 0.964 | ✅ No overfitting |
| **RMSE** | 1.94 kWh | 1.94 kWh | ✅ Excellent |
| **MAE** | 1.48 kWh | 1.49 kWh | ✅ Low error |
| **MAPE** | 5.5% | 6.3% | ✅ Accurate |
| **Negative Predictions** | 0 | 0 | ✅ None |

### Regularization Experiments (Phase 2)

Tested Ridge and Lasso regression with various alpha values:

| Model | Alpha | Test R² | Improvement |
|-------|-------|---------|-------------|
| LinearRegression | N/A | 0.9637 | Baseline |
| Ridge | 0.01 | 0.9637 | **No change** |
| Ridge | 1.0 | 0.9637 | **No change** |
| Lasso | 0.001 | 0.9637 | **No change** |

**Key Finding**: LinearRegression is already optimal! No overfitting detected (train/test gap = 0.6%). StandardScaler normalization makes regularization unnecessary.

See [Phase 2 Findings Report](docs/Phase2_Findings_Report.md) for detailed analysis.

---

## 🧠 Model Explainability (SHAP Analysis)

### Top Feature Importance

```
1. cooling_load              6.87  ← Dominant predictor (thermal efficiency)
2. overall_height            2.08  ← Building volume
3. glazing_area              1.26  ← Window surface area
4. relative_compactness      1.09  ← Building shape efficiency
5. roof_area                 0.93  ← Top surface heat loss
```

**Insight**: `cooling_load` is the strongest predictor because it captures the building's overall thermal efficiency - buildings that gain heat easily also lose heat easily.

### SHAP Visualizations

The project includes comprehensive SHAP analysis:
- Summary plots (bee swarm, bar charts)
- Dependence plots (feature interactions)
- Force plots (individual predictions)
- Statistical summaries

Access via the [Gradio Dashboard](app-ml/entrypoint/gradio_shap_dashboard.py) for interactive exploration.

---

## 🏗️ Project Structure

```
energy-consumption-new/
├── app-ml/                          # ML Pipeline
│   ├── entrypoint/                 # Entry scripts
│   │   ├── train.py               # Model training
│   │   ├── gradio_shap_dashboard.py  # Interactive UI
│   │   ├── analyze_coefficients.py   # Coefficient analysis
│   │   └── experiment_ridge_lasso.py # Regularization experiments
│   ├── src/                        # Source code
│   │   ├── pipelines/             # ML pipeline components
│   │   │   ├── training.py        # Training pipeline
│   │   │   ├── model_factory.py   # Model creation
│   │   │   ├── shap_analyzer.py   # SHAP integration
│   │   │   └── ...
│   └── notebooks/                  # Analysis notebooks
├── config/                          # Configuration
│   └── config.yaml                 # Model & pipeline config
├── data/                            # Data storage
│   ├── raw_data/                   # UCI Energy Efficiency dataset
│   └── prod_data/                  # Processed data
├── docs/                            # Documentation 📚
│   ├── LinearRegression_Reference_Guide.md
│   ├── Ridge_Explained_Simply.md
│   ├── Phase2_Findings_Report.md
│   └── Coefficient_Analysis_Results.md
├── models/                          # Trained models
│   ├── prod/                       # Production model
│   └── experiments/                # Experimental models
├── outputs/                         # Generated outputs
│   └── shap/                       # SHAP visualizations
├── tests/                           # Unit & integration tests
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔬 Technical Approach

### Phase 1: LinearRegression Deep Dive

**Objective**: Establish baseline and understand coefficients

**Results**:
- ✅ R²=0.96 achieved with simple LinearRegression
- ✅ StandardScaler normalization applied
- ✅ Feature importance calculated: `|coefficient| × std_dev`
- ✅ No negative predictions
- ✅ Multicollinearity analyzed (8 pairs with |r| > 0.8)

**Learnings**:
- `cooling_load` dominates (importance = 6.87)
- Some coefficients have counterintuitive signs due to multicollinearity
- Model is well-behaved despite high feature correlation

📖 [Full Phase 1 Guide](docs/LinearRegression_Reference_Guide.md)

### Phase 2: Ridge & Lasso Regularization

**Objective**: Test if regularization improves performance or fixes multicollinearity

**Experiments**:
- Ridge: α = [0.01, 0.1, 1.0, 10, 100]
- Lasso: α = [0.001, 0.01, 0.1, 1.0, 10]

**Results**:
- ❌ **No performance improvement** (all models: R²=0.9637)
- ❌ **Multicollinearity NOT fixed** (signs remain)
- ✅ **Validated baseline is optimal**

**Why regularization didn't help**:
1. No overfitting (train/test gap = 0.6%)
2. Coefficients already small (< 7)
3. StandardScaler normalized features properly
4. Penalty (0.56) << MSE (3.78) for α=0.01

📖 [Ridge Explained Simply](docs/Ridge_Explained_Simply.md) | [Phase 2 Findings](docs/Phase2_Findings_Report.md)

### Phase 3: SHAP Explainability Integration

**Objective**: Make model predictions interpretable

**Implementation**:
- ✅ `SHAPAnalyzer` pipeline component
- ✅ Auto-select explainer (Tree/Linear)
- ✅ Generate SHAP values and visualizations
- ✅ Interactive Gradio dashboard

**Outputs**:
- Summary plots (importance rankings)
- Dependence plots (feature relationships)
- Force plots (individual explanations)
- Statistical summaries (CSV export)

---

## 📊 Dataset

**Source**: [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/energy+efficiency)

**Description**: Building energy analysis using 12 different building shapes simulated in Ecotect

**Features** (8 + 1 target-related):
- `relative_compactness`: Building shape efficiency (volume/surface_area)
- `surface_area`: Total building envelope (m²)
- `wall_area`: Vertical surface area (m²)
- `roof_area`: Top surface area (m²)
- `overall_height`: Building height (m)
- `orientation`: Building direction (N/S/E/W)
- `glazing_area`: Window surface area (0-0.4)
- `glazing_area_distribution`: Window distribution (1-4)
- `cooling_load`: Cooling energy need (kWh) - **Feature**
- `heating_load`: Heating energy need (kWh) - **Target**

**Size**: 768 building simulations

---

## 🛠️ Tech Stack

**Core ML**:
- Python 3.11+
- scikit-learn 1.4+ (LinearRegression, Ridge, Lasso)
- NumPy, Pandas

**Explainability**:
- SHAP (SHapley Additive exPlanations)
- Matplotlib, Seaborn (visualizations)

**UI**:
- Gradio (interactive dashboard)

**Pipeline**:
- StandardScaler (feature normalization)
- Model factory pattern (model-agnostic design)
- YAML configuration (flexible parameterization)

**Testing & Quality**:
- pytest (unit & integration tests)
- Git (version control)
- Type hints (code clarity)

---

## 📚 Documentation

### Learning Guides:
- [LinearRegression Reference Guide](docs/LinearRegression_Reference_Guide.md) - Complete theory & implementation
- [Coefficient Analysis Results](docs/Coefficient_Analysis_Results.md) - Understanding coefficients
- [Ridge Explained Simply](docs/Ridge_Explained_Simply.md) - Step-by-step L2 regularization

### Experiment Reports:
- [Phase 2 Findings Report](docs/Phase2_Findings_Report.md) - Ridge/Lasso experimentation
- [Phase 2 Theory](docs/Phase2_Ridge_Lasso_Theory.md) - L1/L2 regularization theory
- [Phase 2 Workflow](docs/Phase2_Workflow.md) - Experiment methodology

### Guides:
- [FAQ](docs/FAQ.md) - Common questions
- [Suggested Readings](docs/Suggested_Readings.md) - Further learning

Total: **1,444 lines of documentation** 📖

---

## 🧪 Running Experiments

### Compare Ridge/Lasso:

```bash
# Automated experiment across multiple alpha values
python app-ml/entrypoint/experiment_ridge_lasso.py

# Output:
# - Comparison table (all models)
# - Best model identification
# - Coefficient analysis
```

### Analyze Coefficients:

```bash
# Deep dive into coefficient interpretation
python app-ml/entrypoint/analyze_coefficients.py

# Output:
# - Coefficient values & importance
# - Physical interpretation
# - Prediction examples
# - Multicollinearity check
```

### Visualize Ridge Concept:

```bash
# Interactive visualization of Ridge penalty
python app-ml/entrypoint/visualize_ridge_concept.py

# Output:
# - Loss function breakdown (MSE vs Penalty)
# - Coefficient changes across alpha
# - Step-by-step explanation
```

---

## 🎓 Key Learnings

### 1. Simple Baselines Are Powerful
- LinearRegression achieved 96% accuracy
- No need for complex models when data has strong linear relationship
- **Lesson**: Always start with simple baseline before trying complex models

### 2. Regularization Isn't Always Needed
- Ridge/Lasso provided zero benefit
- StandardScaler + good train/test split prevented overfitting
- **Lesson**: Check for overfitting before adding regularization

### 3. Feature Scaling Is Critical
- Without StandardScaler, coefficients would be on vastly different scales
- Scaling enabled proper feature importance calculation
- **Lesson**: Always scale features for linear models

### 4. Multicollinearity Can't Always Be "Fixed"
- Ridge/Lasso didn't change counterintuitive coefficient signs
- Confounding is a data property, not a model bug
- **Lesson**: Understand structural relationships in data

### 5. Explainability Matters
- SHAP helps understand why model makes predictions
- `cooling_load` dominates because it captures thermal efficiency
- **Lesson**: Model interpretability is as important as accuracy

---

## 📝 Future Work

### Phase 4: Tree-Based Models (Next Step)
- [ ] Implement CatBoost baseline
- [ ] Compare with LinearRegression (R²=0.96)
- [ ] Test XGBoost, LightGBM, RandomForest
- [ ] Evaluate if non-linearity improves predictions

### Phase 5: Production Deployment
- [ ] Create REST API (FastAPI)
- [ ] Add input validation
- [ ] Monitor prediction drift
- [ ] A/B testing framework

### Phase 6: Advanced Features
- [ ] Polynomial features (interaction terms)
- [ ] Time-series forecasting (if temporal data available)
- [ ] Ensemble methods
- [ ] Hyperparameter optimization (Optuna)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sudhir Shivaram**
- GitHub: [@sudhirshivaram](https://github.com/sudhirshivaram)
- LinkedIn: [Your LinkedIn Profile]
- Portfolio: [Your Portfolio Link]

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Energy Efficiency dataset
- scikit-learn community for excellent documentation
- SHAP library for explainability tools
- Claude Code for development assistance

---

## 📊 Project Stats

- **Lines of Code**: ~2,500+
- **Documentation**: 1,444 lines (6 comprehensive guides)
- **Models Trained**: 11 (1 LinearReg + 5 Ridge + 5 Lasso)
- **Test Coverage**: Unit & integration tests
- **Performance**: R²=0.96, RMSE=1.94 kWh

---

**Built with ❤️ for transparent, explainable AI**

*Last updated: December 2024*
