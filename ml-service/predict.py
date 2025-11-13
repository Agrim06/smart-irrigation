import sys
import os
import pandas as pd


# make project root importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.save_load import load_model
from config import IRRIGATION_RATE_MM_PER_HR

# load models
model_amount = load_model("rf_amount.joblib")
model_time = load_model("rf_time.joblib")

def predict(soil_moisture, temperature_C, lag_moisture=None, lag_temp=None):
    # simple fallback for missing lags
    if lag_moisture is None:
        lag_moisture = soil_moisture
    if lag_temp is None:
        lag_temp = temperature_C

    # build input features
    row = {
        "soil_moisture": soil_moisture,
        "temperature_C": temperature_C,
        "soil_moisture_lag_1": lag_moisture,
        "temp_lag_1": lag_temp,
    }

    df = pd.DataFrame([row])

    # align columns
    model_features = model_amount.feature_names_in_
    for c in model_features:
        if c not in df.columns:
            df[c] = 0.0
    df = df[model_features]

    mm = float(model_amount.predict(df)[0])
    minutes = float(model_time.predict(df)[0])

    return mm, minutes


if __name__ == "__main__":
    print("\n=== Potato Irrigation Predictor ===")

    # ask user for inputs
    try:
        sm = float(input("Enter soil moisture (0–1): "))
        temp = float(input("Enter temperature (°C): "))
    except ValueError:
        print("Invalid input.")
        sys.exit(1)

    mm, minutes = predict(sm, temp)

    print("\n--- Prediction Result ---")
    print(f"Water required: {mm:.2f} mm")
    print(f"Irrigation duration: {minutes:.2f} minutes")
    print("--------------------------\n")
