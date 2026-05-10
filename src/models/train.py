import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("E:/FixAI/data/processed/train.csv")

print("Original shape:", df.shape)

# =========================
# 2. CLEAN DATA
# =========================
df = df[df["completion_time"] > 0]
df = df[df["completion_time"] < df["completion_time"].quantile(0.99)]

print("After cleaning:", df.shape)

# =========================
# 3. FEATURE ENGINEERING
# =========================

le = LabelEncoder()
df["service"] = le.fit_transform(df["service"])

joblib.dump(le, "E:/FixAI/models/service_encoder.pkl")

df["distance_per_exp"] = df["distance"] / (df["experience"] + 1)

# =========================
# 4. SPLIT X / Y
# =========================
X = df.drop(columns=["completion_time"])
y = df["completion_time"]

# =========================
# 5. MODEL
# =========================
model = XGBRegressor(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=10,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42
)

# =========================
# 6. CROSS VALIDATION (K-FOLD)
# =========================
print("\n=== CROSS VALIDATION START ===")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error"
)

rmse_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

r2_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("\n=== CV RESULTS ===")

print("MAE per fold:", -mae_scores)
print("Mean MAE    :", -mae_scores.mean())

print("\nRMSE per fold:", -rmse_scores)
print("Mean RMSE    :", -rmse_scores.mean())

print("\nR2 per fold:", r2_scores)
print("Mean R2    :", r2_scores.mean())

# =========================
# 7. TRAIN FINAL MODEL (FULL DATA)
# =========================
model.fit(X, y)

print("\n✅ Final training completed!")

# =========================
# 8. SAVE MODEL
# =========================
joblib.dump(model, "E:/FixAI/models/xgb_model.pkl")

print("✅ Model saved!")