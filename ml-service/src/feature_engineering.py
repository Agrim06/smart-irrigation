import pandas as pd

def resample_and_aggregate(df: pd.DataFrame, interval_minutes: int = 15) -> pd.DataFrame:
    rule = f"{interval_minutes}T"
    agg = df[["moisture", "temperature", "humidity"]].resample(rule).mean()
    agg = agg.ffill(limit=2)
    return agg

def create_lag_features(df: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    X = df.copy()
    for lag in range(1, lags + 1):
        X[f"moisture_lag_{lag}"] = X["moisture"].shift(lag)
    X["moisture_diff_1"] = X["moisture"] - X["moisture"].shift(1)
    X["moisture_rolling_mean_3"] = X["moisture"].rolling(window=3, min_periods=1).mean()
    X["hour"] = X.index.hour
    X["dayofyear"] = X.index.dayofyear
    X["dayofweek"] = X.index.dayofweek
    X = X.dropna()
    return X

def build_dataset(agg_df: pd.DataFrame, lags: int = 6, horizon: int = 1):
    df = create_lag_features(agg_df, lags=lags)
    df[f"target_t_plus_{horizon}"] = agg_df["moisture"].shift(-horizon).reindex(df.index)
    df = df.dropna(subset=[f"target_t_plus_{horizon}"])
    feature_cols = [
        c for c in df.columns 
        if c.startswith("moisture_lag_") 
        or c.startswith("moisture_rolling")
        or c in ["moisture_diff_1", "temperature", "humidity", "hour", "dayofyear", "dayofweek"]
    ]
    X = df[feature_cols]
    y = df[f"target_t_plus_{horizon}"]
    return X, y
