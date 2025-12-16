"""
Toy Example: Demonstrating the Scaler Bug
==========================================

This script shows what happens when you:
1. Train a model on SCALED data
2. Save only the MODEL (not the scaler)
3. Try to predict with UNSCALED data

Result: Completely wrong predictions!
"""

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("="*70)
print("TOY EXAMPLE: The Scaler Bug")
print("="*70)

# Create simple toy data
print("\n1. Create toy training data:")
X_train = np.array([[100, 200], [150, 250], [200, 300]])
y_train = np.array([50, 75, 100])
print(f"   X_train (raw):\n{X_train}")
print(f"   y_train: {y_train}")

# Scale the data
print("\n2. Scale the features:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(f"   X_train (scaled):\n{X_train_scaled}")
print(f"   Mean: {scaler.mean_}")
print(f"   Scale: {scaler.scale_}")

# Train model on SCALED data
print("\n3. Train LinearRegression on SCALED data:")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print(f"   Coefficients: {model.coef_}")
print(f"   Intercept: {model.intercept_}")

# Test prediction with SCALED data (correct)
print("\n4. Predict with SCALED data (CORRECT):")
X_test_raw = np.array([[175, 275]])
X_test_scaled = scaler.transform(X_test_raw)
prediction_correct = model.predict(X_test_scaled)[0]
print(f"   X_test (raw): {X_test_raw[0]}")
print(f"   X_test (scaled): {X_test_scaled[0]}")
print(f"   ✅ Prediction: {prediction_correct:.2f}")

# Simulate the BUG: Save only the model (not the scaler)
print("\n5. SIMULATE THE BUG: Save only the model")
print("   (This is what happens in our current code!)")
joblib.dump(model, '/tmp/model_without_scaler.pkl')
print("   Saved: /tmp/model_without_scaler.pkl")

# Load the model (without scaler)
print("\n6. Load the model (scaler lost!):")
loaded_model = joblib.load('/tmp/model_without_scaler.pkl')
print("   Loaded model successfully")

# Try to predict with UNSCALED data (BUG!)
print("\n7. Predict with UNSCALED data (BUG!):")
prediction_wrong = loaded_model.predict(X_test_raw)[0]
print(f"   X_test (raw, NOT scaled): {X_test_raw[0]}")
print(f"   ❌ Prediction: {prediction_wrong:.2f}")
print(f"   Expected: {prediction_correct:.2f}")
print(f"   Error: {abs(prediction_wrong - prediction_correct):.2f}")

# THE SOLUTION: Use sklearn Pipeline
print("\n" + "="*70)
print("THE SOLUTION: Save the Complete Pipeline")
print("="*70)

print("\n8. Create Pipeline with Scaler + Model:")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)
print("   Pipeline created: [StandardScaler -> LinearRegression]")

# Save the COMPLETE pipeline
print("\n9. Save the COMPLETE pipeline:")
joblib.dump(pipeline, '/tmp/model_with_scaler.pkl')
print("   Saved: /tmp/model_with_scaler.pkl")

# Load and predict (works correctly!)
print("\n10. Load pipeline and predict with RAW data:")
loaded_pipeline = joblib.load('/tmp/model_with_scaler.pkl')
prediction_pipeline = loaded_pipeline.predict(X_test_raw)[0]
print(f"    X_test (raw): {X_test_raw[0]}")
print(f"    ✅ Prediction: {prediction_pipeline:.2f}")
print(f"    Expected: {prediction_correct:.2f}")
print(f"    Error: {abs(prediction_pipeline - prediction_correct):.2f}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✅ Correct (scaled data):     {prediction_correct:.2f}")
print(f"❌ Bug (unscaled data):       {prediction_wrong:.2f}  (off by {abs(prediction_wrong - prediction_correct):.2f})")
print(f"✅ Pipeline (handles scaling): {prediction_pipeline:.2f}  (perfect!)")
print("\n💡 Key Lesson: Always save the COMPLETE preprocessing pipeline!")
print("="*70)
