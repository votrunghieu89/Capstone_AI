import pandas as pd
import joblib
from pathlib import Path

# 1. Load model + encoder
BASE_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")
encoder = joblib.load(BASE_DIR / "models" / "service_encoder.pkl")

print("✅ Model loaded!")

# 2. Predict function
def predict_time(service, distance, experience, is_peak_hour):

    # Validate
    if not (0 <= distance <= 150):
        raise ValueError("distance phải từ 0-150 km")

    if not (0 <= experience <= 30):
        raise ValueError("experience phải từ 0-30 năm")

    if is_peak_hour not in [0, 1]:
        raise ValueError("is_peak_hour phải là 0 hoặc 1")

    # Encode service
    service_encoded = encoder.transform([service])[0]

    # IMPORTANT:
    # phải đúng tên cột lúc train
    sample = pd.DataFrame([{
        "service": service_encoded,
        "distance": distance,
        "experience": experience,
        "is_peak_hour": is_peak_hour
    }])

    # Debug
    print("MODEL FEATURES:", model.feature_names_in_)
    print("INPUT FEATURES:", sample.columns.tolist())

    # Predict
    prediction = model.predict(sample)[0]

    return round(prediction, 2)


# Test
if __name__ == "__main__":

    print("\n=== TEST ===")

    print("Case 1:",
          predict_time("Sửa chữa và bảo dưỡng ô tô", 10, 20, 0))

    print("Case 2:",
          predict_time("Sửa chữa và bảo dưỡng ô tô", 20, 20, 0))

    print("Case 3:",
          predict_time("Sửa chữa và bảo dưỡng ô tô", 50, 20, 1))