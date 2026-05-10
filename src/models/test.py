import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    max_error
)

# =========================
# 1. LOAD MODEL + ENCODER
# =========================
model = joblib.load("E:/FixAI/models/xgb_model.pkl")
encoder = joblib.load("E:/FixAI/models/service_encoder.pkl")

print("✅ Model loaded!")

# =========================
# 2. LOAD TEST DATA
# =========================
df = pd.read_csv("E:/FixAI/data/processed/test.csv")
print("Test data shape:", df.shape)

# =========================
# 3. ENCODE SERVICE
# =========================
df["service"] = encoder.transform(df["service"])
df["distance_per_exp"] = df["distance"] / (df["experience"] + 1)
# =========================
# 4. SPLIT X / Y
# =========================
X_test = df.drop(columns=["completion_time"])
y_test = df["completion_time"]

negative_count = (y_test < 0).sum()
print(f"Negative completion_time count in test set: {negative_count}")

# =========================
# 5. PREDICTION
# =========================
y_pred = model.predict(X_test)

# remove invalid values (if needed)
mask = y_test != 0
y_test_filtered = y_test[mask]
y_pred_filtered = y_pred[mask]

# =========================
# 6. METRICS
# =========================
rmse = root_mean_squared_error(y_test_filtered, y_pred_filtered)
mae  = mean_absolute_error(y_test_filtered, y_pred_filtered)
r2   = r2_score(y_test_filtered, y_pred_filtered)
mape = np.mean(np.abs((y_test_filtered - y_pred_filtered) / y_test_filtered)) * 100

evs  = explained_variance_score(y_test_filtered, y_pred_filtered)
medae = median_absolute_error(y_test_filtered, y_pred_filtered)
mxerr = max_error(y_test_filtered, y_pred_filtered)

print("\n=== MODEL EVALUATION ===")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"R2   : {r2:.4f}")
print(f"EVS  : {evs:.4f}")
print(f"MedAE: {medae:.2f}")
print(f"MaxErr: {mxerr:.2f}")

# =========================
# 7. RESULT DATAFRAME
# =========================
result = df.copy()
result["actual"] = y_test.values
result["predicted"] = y_pred
result["error"] = result["actual"] - result["predicted"]

print("\nSample predictions:")
print(result.head(10))

# =========================
# 8. VISUALIZATION
# =========================

# 8.1 Actual vs Predicted
plt.figure()
plt.scatter(result["actual"], result["predicted"], alpha=0.5)
plt.xlabel("Actual Completion Time")
plt.ylabel("Predicted Completion Time")
plt.title("Actual vs Predicted")
plt.show()

# 8.2 Residual Plot
residuals = result["error"]

plt.figure()
plt.scatter(result["predicted"], residuals, alpha=0.5)
plt.axhline(0, color="red")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 8.3 Error Distribution
plt.figure()
plt.hist(residuals, bins=30)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# =========================
# 9. ERROR ANALYSIS BY SERVICE
# =========================
service_error = result.groupby("service")["error"].mean().sort_values()

print("\n=== Mean Error by Service ===")
print(service_error)

plt.figure()
service_error.plot(kind="bar")
plt.title("Mean Error by Service Type")
plt.ylabel("Mean Error")
plt.xticks(rotation=45)
plt.show()

# =========================
# 10. OVER / UNDER PREDICTION
# =========================
over = (result["predicted"] > result["actual"]).sum()
under = (result["predicted"] < result["actual"]).sum()

print("\n=== Prediction Bias ===")
print(f"Over-predict : {over}")
print(f"Under-predict: {under}")