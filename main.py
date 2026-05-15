from fastapi import FastAPI
import joblib
import numpy as np
from pathlib import Path
from pydantic import BaseModel
import pandas as pd

class PredictRequest(BaseModel):
    service: str
    distance: float
    experience: float
    is_peak_hour: int
    
app = FastAPI()

# Load model + encoder
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "xgb_model.pkl")
encoder = joblib.load(BASE_DIR / "models" / "service_encoder.pkl")

@app.get("/")
def home():
    return {"message": "FastFix API is running 🚀"}

@app.post("/predict")
def predict(req: PredictRequest):

    service = req.service
    distance = req.distance
    experience = req.experience
    is_peak_hour = req.is_peak_hour

    # Validate
    if not (0 <= distance <= 150):
        return {"error": "distance phải 0-100"}

    if not (0 <= experience <= 30):
        return {"error": "experience phải 0-30"}

    if is_peak_hour not in [0, 1]:
        return {"error": "is_peak_hour phải 0 hoặc 1"}

    
    service_encoded = encoder.transform([service])[0]
  
    sample = pd.DataFrame([{
        "service": service_encoded,
        "distance": distance,
        "experience": experience,
        "is_peak_hour": is_peak_hour
    }])

    prediction = model.predict(sample)[0]

    predicted_int = int(prediction + 0.5)

    return {"prediction": predicted_int}
        