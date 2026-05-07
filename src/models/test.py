import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 1. Load model + encoder
model = joblib.load("E:/FixAI/models/xgb_model.pkl")
encoder = joblib.load("E:/FixAI/models/service_encoder.pkl")

print("✅ Model loaded!")

# 2. Load test data
df = pd.read_csv("E:/FixAI/data/processed/test.csv")

print("Test data shape:", df.shape)

# 3. Encode service giống train
df["service"] = encoder.transform(df["service"])

# 4. Split X / y
X_test = df.drop(columns=["completion_time"])
y_test = df["completion_time"]

negative_count = (y_test < 0).sum()
print(f"Negative completion_time count in test set: {negative_count}")

# 5. Predict
y_pred = model.predict(X_test)

# 6. Remove invalid y
mask = y_test != 0
y_test_filtered = y_test[mask]
y_pred_filtered = y_pred[mask]

# 7. Metrics
rmse = root_mean_squared_error(y_test_filtered, y_pred_filtered)
mae  = mean_absolute_error(y_test_filtered, y_pred_filtered)
r2   = r2_score(y_test_filtered, y_pred_filtered)

mape = np.mean(
    np.abs((y_test_filtered - y_pred_filtered) / y_test_filtered)
) * 100

# 8. Print result
print("=== MODEL EVALUATION ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R2  : {r2:.4f}")

# 9. Sample prediction
result = df.copy()
result["predicted_time"] = y_pred

print("\nSample predictions:")
print(result.head(10))