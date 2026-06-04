# ============================================================
# Generate Test CSV Files for Dashboard Testing
# Run this ONCE to create sample data, then run the dashboard.
# ============================================================

import pandas as pd
import numpy as np

np.random.seed(42)
n = 500  # number of test customers

# --- 1. churn_predictions.csv ---
predictions = pd.DataFrame({
    "Customer ID": [f"CUST-{i:04d}" for i in range(1, n + 1)],
    "Age": np.random.randint(18, 65, n),
    "Monthly Income": np.random.randint(20000, 100000, n),
    "Subscription Plan": np.random.choice(["Basic", "Standard", "Premium"], n),
    "Device Used Most Often": np.random.choice(["Mobile", "Laptop", "Smart TV", "Tablet"], n),
    "Genre Preference": np.random.choice(["Action", "Drama", "Comedy", "Sci-Fi", "Romance"], n),
    "Region": np.random.choice(["North America", "Europe", "Asia", "South America"], n),
    "Payment History": np.random.choice(["On-time", "Delayed"], n, p=[0.7, 0.3]),
    "Subscription Length (Months)": np.random.randint(1, 48, n),
    "Watch Time (Hours/Week)": np.round(np.random.uniform(1, 40, n), 1),
    "Churn Status": np.random.choice(["Churned", "Stayed"], n, p=[0.3, 0.7]),
})

# Generate predictions based on churn status with some noise
actual_churn = (predictions["Churn Status"] == "Churned").astype(int)
lr_proba = np.clip(actual_churn * 0.6 + np.random.uniform(0, 0.4, n), 0, 1).round(4)
rf_proba = np.clip(actual_churn * 0.7 + np.random.uniform(0, 0.3, n), 0, 1).round(4)

predictions["LR_Predicted_Churn"] = (lr_proba > 0.5).astype(int)
predictions["LR_Churn_Probability"] = lr_proba
predictions["RF_Predicted_Churn"] = (rf_proba > 0.5).astype(int)
predictions["RF_Churn_Probability"] = rf_proba
predictions["LR_Prediction_Label"] = predictions["LR_Predicted_Churn"].map({1: "Churn", 0: "No Churn"})
predictions["RF_Prediction_Label"] = predictions["RF_Predicted_Churn"].map({1: "Churn", 0: "No Churn"})
predictions["Risk_Category"] = pd.cut(
    predictions["RF_Churn_Probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

predictions.to_csv("churn_predictions.csv", index=False)
print(f"1. churn_predictions.csv  ({len(predictions)} rows)")

# --- 2. model_comparison.csv ---
comparison = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": 0.82, "Precision": 0.78, "Recall": 0.75, "F1 Score": 0.76, "AUC-ROC": 0.85},
    {"Model": "Random Forest",      "Accuracy": 0.88, "Precision": 0.85, "Recall": 0.82, "F1 Score": 0.83, "AUC-ROC": 0.91}
])
comparison.to_csv("model_comparison.csv", index=False)
print("2. model_comparison.csv")

# --- 3. feature_importance.csv ---
features = ["Subscription Length (Months)", "Watch Time (Hours/Week)", "Monthly Income",
            "Age", "Payment History", "Subscription Plan", "Device Used Most Often",
            "Genre Preference", "Region"]
importance = pd.DataFrame({
    "Feature": features,
    "RF_Importance": [0.22, 0.19, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03],
    "LR_Coefficient": [0.85, -0.72, 0.45, -0.30, 0.65, -0.20, 0.10, -0.08, 0.05]
})
importance.to_csv("feature_importance.csv", index=False)
print("3. feature_importance.csv")

# --- 4. churn_summary.csv ---
churned = (predictions["Churn Status"] == "Churned").sum()
high_risk = (predictions["Risk_Category"] == "High Risk").sum()
summary = pd.DataFrame({
    "Metric": [
        "Total Customers (Test Set)", "Actual Churned", "Predicted Churned (RF)",
        "High Risk Customers", "Churn Rate (%)", "Best Model",
        "Best Model Accuracy", "Best Model F1 Score"
    ],
    "Value": [
        n, churned, (predictions["RF_Predicted_Churn"] == 1).sum(),
        high_risk, f"{churned / n * 100:.1f}", "Random Forest", "88.00%", "83.00%"
    ]
})
summary.to_csv("churn_summary.csv", index=False)
print("4. churn_summary.csv")

print("\nAll test files created! Now run:")
print("  streamlit run churn_dashboard.py")
