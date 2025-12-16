## Q: What is data leakage and why is it a problem?

**A:**
Data leakage occurs when information that would not be available at prediction time is used to train the model. This can happen if you include features that are derived from, or highly correlated with, the target variable, or if you use future information that would not be known when making predictions. Leakage leads to overly optimistic performance during training and testing, but poor real-world results. Always ensure your features are available at prediction time and not derived from the target.

---

## Q: Should I drop features that are highly correlated with the target, like `cooling_load` when predicting `heating_load`?

**A:**
Yes. If a feature is highly correlated with the target and would not be available at prediction time, it should be dropped to prevent data leakage. For example, `cooling_load` is almost perfectly correlated with `heating_load` in the energy dataset. Including it as a feature would allow the model to "cheat" and produce unrealistically high performance. Drop such features before training and evaluation.

---
## Q: Why start with Ridge for hyperparameter tuning?

**A:**
Ridge regression is a simple, fast, and robust linear model that is less sensitive to multicollinearity and overfitting than plain linear regression. It’s a great starting point for demonstrating hyperparameter tuning because it has a single key hyperparameter (`alpha`), making the process easy to visualize and interpret. Once you’re comfortable with Ridge, you can apply similar techniques to more complex models.

---

## Q: Is the data linear? How do I check?

**A:**
To check linearity, use scatter plots of each feature vs. the target, a correlation matrix, and residual plots after fitting a linear model. If most relationships are roughly straight lines and residuals are randomly scattered around zero, the data is likely suitable for linear models. If you see curves or patterns, consider feature engineering or non-linear models.

---

## Q: How do I compare hyperparameter tuning techniques?

**A:**
Compare the best cross-validated scores (e.g., R²) and the chosen hyperparameters from each method (Grid Search, Random Search, Optuna). Visualize results in a bar chart for clarity. If all methods yield similar results, your search space is likely simple and the model is stable.

---

## Q: Why 5-fold cross-validation?

**A:**
5-fold CV is a good balance between bias and variance in model evaluation. It provides reliable estimates without excessive computation. More folds (e.g., 10) can give slightly more stable results but take longer to run.

---

## Q: What if you choose more folds for cross-validation?

**A:**
Using more folds (like 10) can reduce the variance of your score estimates, but increases computation time. For most datasets, 5 or 10 folds are standard. Very small datasets may benefit from leave-one-out CV, but this is rarely needed in practice.

---

## Q: What hyperparameters were tuned in Ridge regression?

**A:**
The main hyperparameter for Ridge is `alpha`, which controls the strength of regularization. Tuning `alpha` helps balance model complexity and overfitting. In your experiments, you searched over a range of `alpha` values using different tuning strategies.

---

## Q: Can I see a summary table of the tuning results?

**A:**
Yes! Here’s an example summary table:

| Method         | Best Alpha | Best CV R² |
|---------------|------------|------------|
| Grid Search   |   0.001    |   0.96     |
| Random Search |   0.001    |   0.96     |
| Optuna        |   0.0042   |   0.96     |

This shows all methods found similar optimal values and performance.

---
# Frequently Asked Questions (FAQ)

## Q: Why should I try further experiments (e.g., hyperparameter tuning, feature engineering, or error analysis) even if my test metrics are already impressive?

**A:**
Even with strong test metrics, further experiments are recommended because:
- They help confirm your model isn’t overfitting or exploiting quirks in the data.
- They improve your understanding of model behavior (feature importance, error analysis, interpretability).
- They can yield small but meaningful improvements, especially for production use.
- They prepare your model for future data changes and ensure robustness.

In summary: impressive metrics are great, but best practice is to validate, interpret, and optimize your model for reliability and future use.

---


## Q: How can I confirm that my model isn’t overfitting?

**A:**
Overfitting occurs when a model learns the training data too well, including its noise, and performs poorly on new data. To check for overfitting:

- **Compare train and test metrics:** If training and test scores (e.g., R², RMSE) are close, overfitting is less likely. A large gap suggests overfitting.
- **Use cross-validation:** Evaluate your model on multiple train/test splits. Consistent scores across splits mean good generalization.
- **Plot learning curves:** If test and train performance curves are close and stabilize as training size increases, overfitting is unlikely.

**Example (Python):**

```python
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'Mean CV R2: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}')

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
	model, X, y, cv=5, scoring='r2', n_jobs=-1,
	train_sizes=np.linspace(0.1, 1.0, 5)
)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Test')
plt.xlabel('Training Set Size')
plt.ylabel('R2 Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

If you see a small gap between train and test scores, and stable learning curves, your model is likely not overfitting.

---

*Add more questions as you go!*

## Q: Why does my residual plot look different or show patterns?

**A:**
A residual plot can look different for several reasons, and interpreting its shape is important for diagnosing your model:

- **Non-linearity:** If the relationship between features and target is not linear, residuals will show patterns (e.g., curves, clusters) instead of being randomly scattered.
- **Heteroscedasticity:** If the spread of residuals increases or decreases with the predicted value, it means the model’s errors are not constant (variance changes across predictions).
- **Outliers or influential points:** Extreme values can create visible clusters or gaps in the residual plot.
- **Feature engineering or data leakage:** If you recently dropped a feature (like `cooling_load`), the model may now be less able to “cheat,” revealing true structure or issues in the data.
- **Model misspecification:** If the model is too simple (e.g., linear for a non-linear problem), residuals will show systematic patterns.

**What should a good residual plot look like?**
- Residuals should be randomly scattered around zero, with no clear pattern.
- The spread should be roughly constant across all predicted values.

**What to do if your plot looks different?**
- Investigate for non-linearity or missing features.
- Try more complex models or add polynomial features.
- Check for outliers or data quality issues.
- Review your feature engineering and data preparation steps.

## Q: Why should I use SHAP or explainable AI (XAI) tools in my ML application?

**A:**
SHAP (SHapley Additive exPlanations) and other explainable AI (XAI) tools help you understand how your model makes predictions. They provide both global (overall feature importance) and local (individual prediction) explanations. This is important because:
- It builds trust with users and stakeholders by making model decisions transparent.
- It helps debug and improve models by revealing which features drive predictions.
- It is often required for regulatory or business reasons, especially in domains like energy, finance, and healthcare.
- It can uncover data or modeling issues (e.g., data leakage, spurious correlations).

**Recommended readings for SHAP/XAI:**
- [SHAP documentation and gallery](https://shap.readthedocs.io/en/latest/)
- [Interpretable Machine Learning (book) – Chapter on SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)
- [A Guide to Interpretable Machine Learning with SHAP](https://towardsdatascience.com/a-guide-to-interpretable-machine-learning-with-shap-6c943c6e3ea0)
- [Explainable AI: A Guide for Engineers and Data Scientists (Google)](https://cloud.google.com/explainable-ai)

## Q: If I drop a feature (like `cooling_load`) from the features DataFrame in the notebook, do I need to rerun the entire training pipeline?

**A:** No, you do not need to rerun the entire training pipeline if you drop a feature from the features DataFrame (`X`) after the pipeline has already prepared your features and target. As long as you drop the column before any model training, tuning, or evaluation steps, all subsequent analysis will use the correct, updated feature set. You only need to rerun the pipeline if you change the underlying data, preprocessing, or feature engineering logic. Dropping a column after feature preparation is efficient and correct for preventing data leakage or removing unwanted features.


## Q: Why do I sometimes get a `ModuleNotFoundError` for my own modules (like `pipelines`) in Jupyter notebooks?

**A:**
This usually happens when your notebook's working directory or `sys.path` does not include the folder where your custom modules live. Always ensure you:
- Set the working directory to your project root (e.g., with `os.chdir()` at the top of your notebook).
- Add the relevant source folders (like `app-ml/src`) to `sys.path` before importing your modules.
- Run these setup cells first after restarting the kernel.

---

## Q: Why does my notebook sometimes fail to find data files, even though they exist?

**A:**
This is usually due to the current working directory of the notebook kernel. Relative paths are resolved from the working directory, not the notebook's location. Use `os.getcwd()` to check, and set the working directory to your project root with `os.chdir()`. This ensures all relative paths work as expected.

---

## Q: How do I interpret the correlation matrix heatmap?

**A:**
The correlation matrix shows the linear relationship between each pair of variables. Values close to +1 indicate strong positive correlation, values close to -1 indicate strong negative correlation, and values near 0 indicate little or no linear relationship. For example, if `roof_area` has a correlation of -0.86 with the target, it means as `roof_area` increases, the target tends to decrease.

---

## Q: Why do all my hyperparameter tuning methods (Grid Search, Random Search, Optuna) give similar results?

**A:**
If your search space is simple and the optimal value is easy to find, all methods may converge to the same result. This is a sign of a stable model and well-behaved data. For more complex models or larger search spaces, advanced methods like Optuna may offer efficiency advantages.
