#!/usr/bin/env python3
"""
predict.py

Loads model artifact saved by trainA.py (a joblib dict containing keys: "model", "features", ...).
Maps CLI inputs --soil, --temp, --hum to the model's feature names and calls model.predict()
using a pandas.DataFrame so feature names match exactly (no sklearn warning).
"""

import argparse
from pathlib import Path
import joblib
import sys
import pandas as pd

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
    p.add_argument("--raw", action="store_true", help="print raw 0/1 instead of human text")
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
        metadata = obj.get("metadata", {})
        # if model not found inside dict, attempt to find any estimator inside
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
    """
    Build a row dict keyed by each feature name expected by the model.
    Matching is done using substring keywords. If no mapping found, prompt user.
    """
    row = {}
    for feat in features:
        fl = feat.lower()
        assigned = False
        for key, kws in _KEYWORDS.items():
            if any(k in fl for k in kws):
                val = provided.get(key)
                if val is None:
                    # prompt
                    val = prompt_float(key)
                row[feat] = float(val)
                assigned = True
                break
        if not assigned:
            # if feature name exactly matches one of soil/temp/hum keys use that
            for key in ("soil","temp","hum"):
                if key == fl and provided.get(key) is not None:
                    row[feat] = float(provided[key])
                    assigned = True
                    break
        if not assigned:
            # last resort: prompt user for this feature
            row[feat] = prompt_float(feat)
    return row

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

    provided = {"soil": args.soil, "temp": args.temp, "hum": args.hum}

    if features:
        # ensure features is a plain list of strings
        features = list(features)
        # build a single-row DataFrame with the same column names and order
        row = map_inputs_to_features(features, provided)
        X = pd.DataFrame([row], columns=features)
    else:
        # model has no feature list saved; assume order soil,temp,hum
        for k in ("soil","temp","hum"):
            if provided[k] is None:
                provided[k] = prompt_float(k)
        X = pd.DataFrame([[provided["soil"], provided["temp"], provided["hum"]]],
                         columns=["soil","temp","hum"])

    # call predict
    try:
        yhat = model.predict(X)
    except Exception as e:
        # try passing numpy array as fallback
        try:
            yhat = model.predict(X.values)
        except Exception as e2:
            print("Failed to call model.predict():", e, e2)
            sys.exit(1)

    out = int(yhat[0])
    if args.raw:
        print(out)
    else:
        print("PUMP: ON" if out == 1 else "PUMP: OFF")

if __name__ == "__main__":
    main()
