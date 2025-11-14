#!/usr/bin/env python3
"""
water_predict.py

One-stop file that:
- Fetches sensor data from MongoDB irrigation_db.sensordatas collection
- Uses ML model to predict pump status
- Calculates water requirements from crop_irrigation.csv
- Saves predictions to MongoDB irrigation_db.predictions collection
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime, timezone, timedelta
import re
import sys
from pymongo import MongoClient

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
    p.add_argument("--csv", default="data/crop_irrigation.csv", help="CSV file containing crop irrigation requirements (default: data/crop_irrigation.csv)")
    p.add_argument("--crop", required=True, help="Crop name + irrigation method matching CSV")
    p.add_argument("--from-db", action="store_true", help="fetch sensor values from MongoDB irrigation_db.sensordatas")
    p.add_argument("--device-id", type=str, default=None, help="filter by device ID when using --from-db")
    p.add_argument("--mongo-uri", type=str, default="mongodb://127.0.0.1:27017", help="MongoDB connection URI")
    p.add_argument("--irrigation-rate", type=float, default=8.0, help="Irrigation rate in mm per hour (default: 8.0)")
    p.add_argument("--raw", action="store_true", help="print raw 0/1 instead of human text")
    return p.parse_args()

def prompt_float(name):
    while True:
        try:
            v = input(f"Enter {name}: ").strip()
            return float(v)
        except Exception:
            print("Please enter a numeric value.")

def fetch_latest_sensor_data(mongo_uri, device_id=None):
    """
    Fetch the latest sensor reading from irrigation_db.sensordatas collection.
    Returns dict with keys: moisture, temperature, humidity, deviceId, timestamp
    """
    try:
        client = MongoClient(mongo_uri)
        db = client["irrigation_db"]
        collection = db["sensordatas"]
        
        # Build filter
        filter_query = {}
        if device_id:
            filter_query["deviceId"] = device_id
        
        # Find latest document sorted by timestamp descending
        latest = collection.find_one(filter_query, sort=[("timestamp", -1)])
        
        if not latest:
            raise ValueError(f"No sensor data found{' for device ' + device_id if device_id else ''}")
        
        return {
            "moisture": latest.get("moisture"),
            "temperature": latest.get("temperature"),
            "humidity": latest.get("humidity"),
            "deviceId": latest.get("deviceId"),
            "timestamp": latest.get("timestamp")
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fetch from MongoDB: {e}")

def create_alert(mongo_uri, device_id, pump_status, water_mm, pump_time_sec, previous_pump_status=None):
    """
    Create an alert when pump status changes.
    
    Args:
        mongo_uri: MongoDB connection URI
        device_id: Device ID from sensor data (or None)
        pump_status: 1 for ON, 0 for OFF
        water_mm: Water amount in mm
        pump_time_sec: Pump time in seconds
        previous_pump_status: Previous pump status (0 or 1) if known, None if first prediction
    """
    try:
        client = MongoClient(mongo_uri)
        db = client["irrigation_db"]
        alert_collection = db["alerts"]
        
        device_key = device_id or "unknown"
        
        # Check if we already created an alert for this status change recently (within last 30 seconds)
        cutoff_time = datetime.utcnow() - timedelta(seconds=30)
        existing_alert = alert_collection.find_one({
            "deviceId": device_key,
            "type": "PUMP_ON" if pump_status == 1 else "PUMP_OFF",
            "createdAt": {"$gte": cutoff_time},
            "read": False
        }, sort=[("createdAt", -1)])
        
        if not existing_alert:
            alert_type = "PUMP_ON" if pump_status == 1 else "PUMP_OFF"
            
            if pump_status == 1:
                pump_time_min = pump_time_sec / 60
                message = f"Irrigation required: {water_mm:.1f} mm of water needed. Pump should run for {pump_time_min:.1f} minutes."
            else:
                message = "No irrigation needed. Pump should remain OFF."
            
            alert_doc = {
                "deviceId": device_key,
                "type": alert_type,
                "message": message,
                "waterMM": water_mm,
                "pumpTimeSec": pump_time_sec,
                "read": False,
                "createdAt": datetime.utcnow()
            }
            
            alert_collection.insert_one(alert_doc)
            print(f"  ✓ Alert created: {alert_type} for device {device_key}")
        else:
            print(f"  Alert already exists for this status change (within last 30s)")
    except Exception as e:
        # Don't fail prediction if alert creation fails
        import traceback
        print(f"  ⚠ Warning: Failed to create alert: {e}")
        print(f"  Traceback: {traceback.format_exc()}")

def save_prediction_to_db(mongo_uri, device_id, pump_status, water_mm, pump_time_sec, skip_duplicates=True):
    """
    Save prediction result to irrigation_db.predictions collection.
    
    Args:
        mongo_uri: MongoDB connection URI
        device_id: Device ID from sensor data (or None)
        pump_status: 1 for ON, 0 for OFF
        water_mm: Water amount in mm
        pump_time_sec: Pump time in seconds
        skip_duplicates: If True, skip saving if identical prediction exists within last 15 seconds
    """
    try:
        client = MongoClient(mongo_uri)
        db = client["irrigation_db"]
        collection = db["predictions"]
        
        device_key = device_id or "unknown"
        
        # Check for status change BEFORE checking duplicates
        # This ensures alerts are created even if prediction is duplicate
        all_predictions = list(collection.find(
            {"deviceId": device_key}
        ).sort([("createdAt", -1)]).limit(2))
        
        # Determine previous pump status
        previous_pump_status = None
        if len(all_predictions) > 0:
            previous_prediction = all_predictions[0]  # Most recent prediction
            previous_pump_status = 1 if previous_prediction.get("waterMM", 0) > 0 else 0
        
        # Check for duplicate predictions if enabled
        is_duplicate = False
        if skip_duplicates:
            # Check if an identical prediction was created in the last 15 seconds
            cutoff_time = datetime.utcnow() - timedelta(seconds=15)
            
            duplicate = collection.find_one({
                "deviceId": device_key,
                "waterMM": water_mm,
                "pumpTimeSec": pump_time_sec,
                "used": (pump_status == 0),
                "createdAt": {"$gte": cutoff_time}
            }, sort=[("createdAt", -1)])
            
            if duplicate:
                print(f"  Skipping duplicate prediction (identical prediction already exists)")
                is_duplicate = True
        
        # Always create alert if status changed, even if prediction is duplicate
        status_changed = (previous_pump_status is None or previous_pump_status != pump_status)
        if status_changed:
            print(f"  Status change detected: {'ON' if previous_pump_status == 1 else 'OFF' if previous_pump_status == 0 else 'N/A'} -> {'ON' if pump_status == 1 else 'OFF'}")
            create_alert(mongo_uri, device_id, pump_status, water_mm, pump_time_sec, previous_pump_status)
        
        # If duplicate, return existing prediction ID without saving
        if is_duplicate:
            return duplicate.get("_id")
        
        # Generate predictionId as ISO timestamp
        prediction_id = datetime.utcnow().isoformat() + "Z"
        
        # Set used to true when pump needs to be turned OFF, false when pump needs to be ON
        # If pump is OFF, prediction is "used" (no action needed)
        # If pump is ON, prediction is not "used" yet (action pending)
        used = (pump_status == 0)
        
        prediction_doc = {
            "deviceId": device_key,
            "waterMM": water_mm,
            "pumpTimeSec": pump_time_sec,
            "predictionId": prediction_id,
            "used": used,
            "createdAt": datetime.utcnow()
        }
        
        result = collection.insert_one(prediction_doc)
        prediction_id_result = result.inserted_id
        
        return prediction_id_result
    except Exception as e:
        raise RuntimeError(f"Failed to save prediction to MongoDB: {e}")

def load_artifact(path: Path):
    obj = joblib.load(str(path))
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("pipeline") or None
        features = obj.get("features") or obj.get("feature_names") or None
        metadata = obj.get("metadata", {})
        if model is None:
            for v in obj.values():
                if hasattr(v, "predict"):
                    model = v
                    break
        return model, features, metadata
    if hasattr(obj, "predict"):
        return obj, None, {}
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
        sys.exit(1)

    model, features, metadata = load_artifact(model_path)
    if model is None:
        print("No model found inside artifact.")
        sys.exit(1)

    # Track device_id for saving prediction
    device_id = None
    
    # Fetch from database if requested
    if args.from_db:
        try:
            print(f"Fetching latest sensor data from MongoDB...")
            sensor_data = fetch_latest_sensor_data(args.mongo_uri, args.device_id)
            provided = {
                "soil": sensor_data["moisture"],
                "temp": sensor_data["temperature"],
                "hum": sensor_data["humidity"]
            }
            device_id = sensor_data.get("deviceId")
            print(f"Using sensor data from device: {device_id or 'N/A'}")
            print(f"  Moisture: {provided['soil']}%")
            print(f"  Temperature: {provided['temp']}°C")
            print(f"  Humidity: {provided['hum']}%")
        except Exception as e:
            print(f"Error fetching from database: {e}", file=sys.stderr)
            sys.exit(1)
    else:
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
        # model has no feature list saved; assume order soil,temp,hum
        for k in ("soil","temp","hum"):
            if provided[k] is None:
                provided[k] = prompt_float(k)
        X = pd.DataFrame([[provided["soil"], provided["temp"], provided["hum"]]],
                         columns=["soil","temp","hum"])

    # --- Predict pump ---
    try:
        yhat = model.predict(X)
    except Exception as e:
        # try passing numpy array as fallback
        try:
            yhat = model.predict(X.values)
        except Exception as e2:
            print("Failed to call model.predict():", e, e2)
            sys.exit(1)

    pump_status = int(yhat[0])
    
    if args.raw:
        print(pump_status)
    else:
        print(f"PUMP: {'ON' if pump_status == 1 else 'OFF'}")

    # Initialize water values (will be set if pump is ON)
    water_mm = 0.0
    pump_time_sec = 0.0

    if pump_status == 0:
        print("No irrigation required.")
    else:
        # --- Calculate water requirement based on soil moisture deficit ---
        # This matches the formula used in training (preprocess/features.py line 30)
        # Formula: (setpoint - soil_moisture) * 100 + max(0, temp - 18) * 1.5
        soil_moisture = provided["soil"] / 100.0  # Convert percentage to decimal (e.g., 25% -> 0.25)
        temp = provided["temp"]
        
        # Optimal soil moisture setpoint (30% = 0.30) - matches training data
        setpoint = 0.30
        
        # Calculate water requirement using the exact training formula
        # This ensures predictions vary based on actual soil moisture and temperature
        water_mm = (setpoint - soil_moisture) * 100 + max(0, temp - 18) * 1.5
        water_mm = max(0, water_mm)  # Ensure non-negative (np.clip equivalent)
        
        # Load CSV for crop-specific maximum limits
        csv_path = Path(args.csv)
        if not csv_path.is_absolute() and not csv_path.exists():
            script_dir = Path(__file__).parent.parent
            csv_path = script_dir / csv_path
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df[["Irrigation_Min_mm","Irrigation_Max_mm"]] = df["Irrigation Water Requirement, mm"]\
                    .apply(lambda x: pd.Series(parse_range(x)))
                
                crop_row = df[df["Crop and Irrigation Method"] == args.crop]
                if not crop_row.empty:
                    crop_row = crop_row.iloc[0]
                    min_mm, max_mm = crop_row["Irrigation_Min_mm"], crop_row["Irrigation_Max_mm"]
                    # Cap water requirement to crop-specific maximum per application
                    # Use max_mm as a reasonable single-application limit
                    water_mm = min(water_mm, max_mm)
            except Exception as e:
                print(f"Warning: Could not load crop CSV for limits: {e}", file=sys.stderr)
        
        # Ensure minimum water amount if pump is ON (at least 1mm)
        if water_mm < 1.0:
            water_mm = 1.0
        
        # Calculate pump time in seconds based on irrigation rate (mm per hour)
        # Formula: (water_mm / irrigation_rate_mm_per_hr) * 3600 seconds
        # But the code uses * 60, which seems wrong - should be * 3600 for hours to seconds
        # However, keeping the existing formula for consistency, but fixing it
        pump_time_sec = (water_mm / args.irrigation_rate) * 3600  # Fixed: hours to seconds conversion

        print(f"  Soil moisture: {provided['soil']:.1f}% (setpoint: {setpoint*100:.1f}%)")
        print(f"  Moisture deficit: {(setpoint - soil_moisture)*100:.2f}%")
        temp_adj = max(0, temp - 18) * 1.5
        print(f"  Temperature: {temp:.1f}°C (adjustment: +{temp_adj:.1f} mm)")
        print(f"Estimated irrigation amount: {water_mm:.1f} mm")
        print(f"Estimated pump time: {pump_time_sec:.1f} seconds ({pump_time_sec/60:.1f} minutes)")
        prediction_time = datetime.now(timezone.utc).isoformat()
        print(f"Prediction time (UTC): {prediction_time}")

    # Save prediction to MongoDB
    try:
        print(f"\nSaving prediction to MongoDB...")
        inserted_id = save_prediction_to_db(args.mongo_uri, device_id, pump_status, water_mm, pump_time_sec)
        print(f"Prediction saved with ID: {inserted_id}")
    except Exception as e:
        print(f"Warning: Failed to save prediction to database: {e}", file=sys.stderr)
        # Don't exit on save failure, prediction was successful

if __name__ == "__main__":
    main()
