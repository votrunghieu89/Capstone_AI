import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df = pd.read_csv("E:/FixAI/data/processed/train.csv")

print("Data shape:", df.shape)
print(df.head())

# 2. Remove invalid target
df = df[df["completion_time"] != 0]

print("After remove y=0:", df.shape)

# 3. Encode categorical feature (SERVICE)
le = LabelEncoder()
df["service"] = le.fit_transform(df["service"])

# (optional) save encoder để predict dùng lại
joblib.dump(le, "E:/FixAI/models/service_encoder.pkl")

# 4. Split features / label
X = df.drop(columns=["completion_time"])
y = df["completion_time"]

# 5. Model
model = XGBRegressor(
    n_estimators=800,
    max_depth=5,        # giảm overfit
    learning_rate=0.03, # học mịn hơn
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    random_state=42
)

# 6. Train
model.fit(X, y)

print("✅ Training completed!")

# 7. Save model
joblib.dump(model, "E:/FixAI/models/xgb_model.pkl")

print("✅ Model saved!")