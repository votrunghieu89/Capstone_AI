import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score


df = pd.read_csv("E:/FixAI/data/processed/train.csv")

print("Original shape:", df.shape)


df = df.dropna()

df = df[df["completion_time"] > 0]
df = df[df["completion_time"] < df["completion_time"].quantile(0.99)]

print("After cleaning:", df.shape)



le = LabelEncoder()
df["service"] = le.fit_transform(df["service"])

joblib.dump(le, "E:/FixAI/models/service_encoder.pkl")

df["distance_per_exp"] = df["distance"] / (df["experience"] + 1)


X = df.drop(columns=["completion_time"])
y = df["completion_time"]


model = XGBRegressor(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42
)






model.fit(X, y)

print("\n✅ Final training completed!")


joblib.dump(model, "E:/FixAI/models/xgb_model.pkl")

print("✅ Model saved!")