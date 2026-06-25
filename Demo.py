# ============================================================
'''
## Model Demonstration

To demonstrate the practical application of the developed predictive maintenance system, a Random Forest model was trained using the complete dataset and the engineered features.

The demo reads machine data from a CSV file, applies the same preprocessing and feature engineering steps used during model development, and generates a prediction for a selected machine instance.

For the selected sample, the system displays:

* Failure probability
* Predicted machine status (Failure / No Failure)
* Actual machine status
* Prediction correctness

This demonstration illustrates how the trained model can be used to evaluate new machine data and support predictive maintenance decisions.
'''
# ============================================================

# ============================================================
# RANDOM FOREST DEMO - TRAIN FROM FILE & PREDICT ONE SAMPLE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

file_path = r"D:\Degendorf Institute of Technology\Semster 1\Machine Learning and Deep Learning in Production and Logistics\Project\Predictive_Maintenance_Project\ai4i2020.csv"

df_demo = pd.read_csv(file_path)


# ------------------------------------------------------------
# 2. Encode Type Feature
# ------------------------------------------------------------

le_demo = LabelEncoder()
df_demo["Type_enc"] = le_demo.fit_transform(df_demo["Type"])


# ------------------------------------------------------------
# 4. Feature Selection
# ------------------------------------------------------------

feature_cols = [
    "Air temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

X = df_demo[feature_cols]
y = df_demo["Machine failure"]

# ------------------------------------------------------------
# 5. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 6. Train Random Forest
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_scaled, y)


# ------------------------------------------------------------
# 7. Select Sample Row For Prediction
# ------------------------------------------------------------
# Change row number to test any machine
# Example:
# sample_row = 0
# sample_row = 100
# sample_row = 500
# ------------------------------------------------------------

#----------------------------------- Start : This part is to show us the failure rows----------------------------
# failure_rows = df_demo[df_demo["Machine failure"] == 1]

# print(f"Number of failure samples: {len(failure_rows)}")

# print("\nFirst 20 failure rows:")
# print(failure_rows.index.tolist()[:20])

#------------------------------------- End : This part is to show us the failure rows--------------------------------

# [50, 69, 77, 160, 161, 168, 194, 207, 242, 248, 249, 259, 327, 380, 442, 463, 586, 603, 746, 847]
sample_row = 50

sample = df_demo.iloc[[sample_row]].copy()

actual_value = sample["Machine failure"].iloc[0]

# ------------------------------------------------------------
# 8. Prepare Sample
# ------------------------------------------------------------

sample_X = sample[feature_cols]

sample_X_scaled = scaler.transform(sample_X)

# ------------------------------------------------------------
# 9. Predict
# ------------------------------------------------------------

prediction = rf_model.predict(sample_X_scaled)[0]

failure_probability = rf_model.predict_proba(
    sample_X_scaled
)[0, 1]

# ------------------------------------------------------------
# 10. Results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTIVE MAINTENANCE DEMO")
print("=" * 60)

print(f"Dataset File       : {file_path}")
print(f"Selected Row       : {sample_row}")
print(f"Failure Probability: {failure_probability:.2%}")

print()

if prediction == 1:
    print("Predicted Value    : MACHINE FAILURE")
else:
    print("Predicted Value    : NO FAILURE")

if actual_value == 1:
    print("Actual Value       : MACHINE FAILURE")
else:
    print("Actual Value       : NO FAILURE")

print()

if prediction == actual_value:
    print("✓ Prediction Correct")
else:
    print("✗ Prediction Incorrect")

print("=" * 60)