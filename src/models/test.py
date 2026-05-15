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

# 8.1 Actual vs Predicted (remove outliers)

df_plot = result.copy()

# Tính sai số tuyệt đối
df_plot["abs_error"] = abs(df_plot["actual"] - df_plot["predicted"])

# Remove các điểm lệch quá lớn (>50)
filtered = df_plot[df_plot["abs_error"] <= 60]

# Plot
plt.figure(figsize=(7,7))

plt.scatter(
    filtered["actual"],
    filtered["predicted"],
    alpha=0.6
)

# Đường perfect prediction
min_val = min(filtered["actual"].min(), filtered["predicted"].min())
max_val = max(filtered["actual"].max(), filtered["predicted"].max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(alpha=0.3)

plt.show()

# Plot
plt.figure()
plt.scatter(filtered["actual"], filtered["predicted"], alpha=0.5)
plt.xlabel("Actual Completion Time")
plt.ylabel("Predicted Completion Time")
plt.title("Actual vs Predicted")
plt.show()
# 8.2 Residual Plot (remove outliers)

df_plot = result.copy()

residuals = df_plot["error"]

# IQR filter cho residuals
Q1 = residuals.quantile(0.25)
Q3 = residuals.quantile(0.75)
IQR = Q3 - Q1

filtered = df_plot[
    (residuals >= Q1 - 1.5 * IQR) &
    (residuals <= Q3 + 1.5 * IQR)
]

plt.figure()
plt.scatter(filtered["predicted"], filtered["error"], alpha=0.5)
plt.axhline(0, color="red")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

residuals = result["error"]

# 1. lọc outlier theo percentile (giữ 1% - 99%)
low, high = np.percentile(residuals, [1, 99])
filtered = residuals[(residuals >= low) & (residuals <= high)]

plt.figure(figsize=(8,5))

plt.hist(filtered, bins=40, edgecolor="black", alpha=0.75)

plt.axvline(0, color="red", linestyle="--", linewidth=2)

plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.grid(alpha=0.2)
plt.show()

# =========================
# 9. ERROR ANALYSIS BY SERVICE
# =========================
result["service_name"] = encoder.inverse_transform(result["service"])
service_error = (
    result[result["service"] != 12]   # loại service 12
    .groupby("service_name")["error"]
    .mean()
    .sort_values()
)

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


# =========================
# 11. FEATURE IMPORTANCE
# =========================

# lấy importance
importance = model.feature_importances_

# tên feature
feature_names = X_test.columns

# dataframe cho đẹp
feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

# sort giảm dần
feat_imp = feat_imp.sort_values(
    by="importance",
    ascending=False
)

print("\n=== FEATURE IMPORTANCE ===")
print(feat_imp)

# plot
plt.figure(figsize=(10,6))

plt.barh(
    feat_imp["feature"],
    feat_imp["importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

# feature quan trọng nhất nằm trên cùng
plt.gca().invert_yaxis()

plt.grid(alpha=0.2)

plt.show()