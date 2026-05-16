import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV


# ======================
# LOAD DATA
# ======================
df = pd.read_csv("E:/FixAI/data/processed/train.csv")

print("Original shape:", df.shape)


# ======================
# CLEAN DATA
# ======================
df = df.dropna()

df = df[df["completion_time"] > 0]
df = df[df["completion_time"] < df["completion_time"].quantile(0.99)]

print("After cleaning:", df.shape)


# ======================
# ENCODE
# ======================
le = LabelEncoder()
df["service"] = le.fit_transform(df["service"])

joblib.dump(le, "E:/FixAI/models/service_encoder.pkl")




# ======================
# SPLIT X / y
# ======================
X = df.drop(columns=["completion_time"])
y = df["completion_time"]


# ======================
# MODEL BASE
# ======================
xgb = XGBRegressor(random_state=42)


# ======================
# HYPERPARAMETER GRID
# ======================
param_grid = {
    "n_estimators": [500, 1000, 2000],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 0.85, 1.0],
    "colsample_bytree": [0.8, 0.85, 1.0],
}


# ======================
# CROSS VALIDATION SETUP
# ======================
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# ======================
# GRID SEARCH (CV + TUNING)
# ======================
print("\n=== GRID SEARCH START ===")

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=kf,
    scoring="r2",
    verbose=2,
    n_jobs=-1
)

grid.fit(X, y)


# ======================
# BEST RESULT
# ======================
print("\n=== BEST RESULT ===")
print("Best Params:", grid.best_params_)
print("Best Score :", grid.best_score_)


# ======================
# FINAL MODEL
# ======================
best_model = grid.best_estimator_

best_model.fit(X, y)

print("\n✅ Final model trained!")


# ======================
# SAVE MODEL
# ======================
joblib.dump(best_model, "E:/FixAI/models/xgb_model.pkl")

print("✅ Model saved!")