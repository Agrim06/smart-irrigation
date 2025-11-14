#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime, timezone
import re

# keywords to match model feature column names
_KEYWORDS = {
    "soil": ["soil", "moist"],
    "temp": ["temp", "temperature"],
    "hum":  ["hum", "humid", "humidity"]
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to joblib .pkl produced by trainA.py")
    p.add_argument("--soil", type=float, default=None, help="soil moisture value (numeric)")
    p.add_argument("--temp", type=float, default=None, help="air temperature (numeric)")
    p.add_argument("--hum", type=float, default=None, help="air humidity (numeric)")
    p.add_argument("--csv", required=True, help="CSV file containing crop irrigation requirements")
    p.add_argument("--crop", required=True, help="Crop name + irrigation method matching CSV")
    return p.parse_args()

def prompt_float(name):
    while True:
        try:
            v = input(f"Enter {name}: ").strip()
            return float(v)
        except Exception:
            print("Please enter a numeric value.")

def load_artifact(path: Path):
    obj = joblib.load(str(path))
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("pipeline") or None
        features = obj.get("features") or obj.get("feature_names") or None
        if model is None:
            for v in obj.values():
                if hasattr(v, "predict"):
                    model = v
                    break
        return model, features
    if hasattr(obj, "predict"):
        return obj, None
    raise RuntimeError("Model artifact does not contain a sklearn model or expected dict.")

def map_inputs_to_features(features, provided):
    row = {}
    for feat in features:
        fl = feat.lower()
        assigned = False
        for key, kws in _KEYWORDS.items():
            if any(k in fl for k in kws):
                val = provided.get(key)
                if val is None:
                    val = prompt_float(key)
                row[feat] = float(val)
                assigned = True
                break
        if not assigned:
            for key in ("soil","temp","hum"):
                if key == fl and provided.get(key) is not None:
                    row[feat] = float(provided[key])
                    assigned = True
                    break
        if not assigned:
            row[feat] = prompt_float(feat)
    return row

def parse_range(value):
    """
    Parse 'min-max' strings into floats.
    Handles extra text, single numbers, or 0.
    Returns (min_val, max_val) as floats.
    """
    value = str(value).strip()
    
    # If it's a simple number
    if re.fullmatch(r"[\d.]+", value):
        num = float(value)
        return num, num
    
    # If it's a range with or without extra text
    if "-" in value:
        parts = value.split("-")
        # Extract first number from each part
        min_val = float(re.findall(r"[\d.]+", parts[0])[0])
        max_val = float(re.findall(r"[\d.]+", parts[1])[0])
        return min_val, max_val
    
    # fallback: extract first number in the string
    nums = re.findall(r"[\d.]+", value)
    if len(nums) >= 1:
        num = float(nums[0])
        return num, num
    
    # if nothing numeric found, default to 0
    return 0.0, 0.0

def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print("Model file not found:", model_path)
        return

    model, features = load_artifact(model_path)

    provided = {
        "soil": args.soil if args.soil is not None else prompt_float("soil"),
        "temp": args.temp if args.temp is not None else prompt_float("temperature"),
        "hum": args.hum if args.hum is not None else prompt_float("humidity")
    }

    # --- Prepare features DataFrame ---
    if features:
        features = list(features)
        row = map_inputs_to_features(features, provided)
        X = pd.DataFrame([row], columns=features)
    else:
        X = pd.DataFrame([[provided["soil"], provided["temp"], provided["hum"]]],
                         columns=["soil","temp","hum"])

    # --- Predict pump ---
    yhat = model.predict(X)
    pump_status = int(yhat[0])
    print(f"PUMP: {'ON' if pump_status == 1 else 'OFF'}")

    if pump_status == 0:
        print("No irrigation required.")
        return

    # --- Load CSV for crop irrigation ---
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("CSV file not found:", csv_path)
        return
    df = pd.read_csv(csv_path)
    df[["Irrigation_Min_mm","Irrigation_Max_mm"]] = df["Irrigation Water Requirement, mm"]\
        .apply(lambda x: pd.Series(parse_range(x)))

    crop_row = df[df["Crop and Irrigation Method"] == args.crop]
    if crop_row.empty:
        print(f"Crop '{args.crop}' not found in CSV.")
        return
    crop_row = crop_row.iloc[0]

    min_mm, max_mm = crop_row["Irrigation_Min_mm"], crop_row["Irrigation_Max_mm"]
    irrigation_mm = (min_mm + max_mm)/2  # take average

    # --- Temperature adjustment ---
    temp = provided["temp"]
    if temp > 35:
        increments = int((temp - 35)//10) + 1
        irrigation_mm *= (1 + 0.1*increments)/180

    print(f"Estimated irrigation amount: {irrigation_mm:.1f} mm")
    prediction_time = datetime.now(timezone.utc).isoformat()
    print(f"Prediction time (UTC): {prediction_time}")

if __name__ == "__main__":
    main()
