# ml-service/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import logging
import traceback
from uuid import uuid4

# --------------------
# Configuration
# --------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/moisture_xgb.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "irrigation_db")
DEFAULT_COLLECTION = os.environ.get("MONGO_COLLECTION", "sensordatas")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------
# Global state
# --------------------
class AppState:
    mongodb_client: Optional[AsyncIOMotorClient] = None
    mongodb: Optional[AsyncIOMotorDatabase] = None
    model = None
    model_metadata: Dict[str, Any] = {}
    training_status: Dict[str, Any] = {"status": "idle"}

app_state = AppState()

# --------------------
# Lifespan management
# --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting up ML service...")
    
    # Connect to MongoDB
    try:
        app_state.mongodb_client = AsyncIOMotorClient(MONGO_URI)
        app_state.mongodb = app_state.mongodb_client[DB_NAME]
        await app_state.mongodb.command("ping")
        logger.info(f"Connected to MongoDB: {DB_NAME}")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
    
    # Load model
    load_model()
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML service...")
    if app_state.mongodb_client:
        app_state.mongodb_client.close()
        logger.info("MongoDB connection closed")

app = FastAPI(
    title="Smart Irrigation ML Service",
    version=MODEL_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Model management
# --------------------
def load_model():
    """Load model from disk"""
    if os.path.exists(MODEL_PATH):
        try:
            app_state.model = joblib.load(MODEL_PATH)
            app_state.model_metadata = {
                "version": MODEL_VERSION,
                "loaded_at": datetime.utcnow().isoformat(),
                "path": MODEL_PATH
            }
            logger.info(f"Loaded model from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            app_state.model = None
    else:
        logger.warning(f"No model file at {MODEL_PATH}")
        app_state.model = None

# --------------------
# Pydantic schemas with validation
# --------------------
class Reading(BaseModel):
    timestamp: datetime
    moisture: float = Field(ge=0, le=1023, description="Moisture sensor reading (0-1023)")
    temp: Optional[float] = Field(None, ge=-40, le=80, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")

    @validator('moisture')
    def validate_moisture(cls, v):
        if v < 0 or v > 1023:
            raise ValueError('Moisture must be between 0 and 1023')
        return v

class PredictRequest(BaseModel):
    deviceId: Optional[str] = None
    cropId: str = Field(..., description="Crop type identifier")
    recentReadings: Optional[List[Reading]] = None
    weather: Optional[Dict[str, Any]] = None

class PredictResponse(BaseModel):
    prediction_id: str
    predicted_moisture: Dict[str, float]
    recommend_irrigate: bool
    recommended_duration_minutes: int
    confidence: float
    explanation: Dict[str, Any]
    model_version: str
    timestamp: datetime

class TrainRequest(BaseModel):
    mongo_uri: Optional[str] = None
    device_id: Optional[str] = None
    collection: Optional[str] = DEFAULT_COLLECTION
    interval: int = Field(15, ge=5, le=60, description="Resample interval (5-60 minutes)")
    lags: int = Field(6, ge=1, le=24, description="Number of lag features (1-24)")
    horizon: int = Field(1, ge=1, le=12, description="Prediction horizon")
    out_path: Optional[str] = Field(MODEL_PATH)
    n_estimators: int = Field(200, ge=50, le=1000)
    learning_rate: float = Field(0.05, ge=0.001, le=1.0)
    max_depth: int = Field(4, ge=2, le=10)

class TrainResponse(BaseModel):
    status: str
    train_job_id: str
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    database_connected: bool
    training_status: str
    timestamp: datetime

# --------------------
# Dependencies
# --------------------
async def get_database() -> AsyncIOMotorDatabase:
    """Dependency for database access"""
    if app_state.mongodb is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return app_state.mongodb

# --------------------
# Data fetching (async)
# --------------------
async def fetch_sensor_data_async(
    db: AsyncIOMotorDatabase,
    collection: str = DEFAULT_COLLECTION,
    device_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 10000
) -> pd.DataFrame:
    """Fetch sensor data from MongoDB asynchronously"""
    try:
        coll = db[collection]
        query = {}
        
        if device_id:
            query["deviceId"] = device_id
        if start or end:
            query["timestamp"] = {}
            if start:
                query["timestamp"]["$gte"] = start
            if end:
                query["timestamp"]["$lte"] = end
        
        cursor = coll.find(query).sort("timestamp", 1).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        if not docs:
            return pd.DataFrame()
        
        rows = []
        for doc in docs:
            ts = doc.get("timestamp")
            if ts is None:
                continue
            
            try:
                ts_parsed = pd.to_datetime(ts)
            except Exception:
                continue
            
            rows.append({
                "timestamp": ts_parsed,
                "deviceId": doc.get("deviceId"),
                "moisture": doc.get("moisture"),
                "temperature": doc.get("temperature"),
                "humidity": doc.get("humidity")
            })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df
    
    except Exception as e:
        logger.error(f"Error fetching sensor data: {e}")
        raise

# --------------------
# Feature engineering
# --------------------
def resample_and_aggregate(df: pd.DataFrame, interval_minutes: int = 15) -> pd.DataFrame:
    """Resample time series data"""
    rule = f"{interval_minutes}T"
    agg = df[["moisture", "temperature", "humidity"]].resample(rule).mean()
    agg = agg.ffill(limit=2)
    return agg

def create_lag_features(df: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    """Create lag features for time series prediction"""
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
    """Build training dataset with features and target"""
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

# Crop thresholds
CROP_THRESHOLDS = {
    "wheat": 600,
    "rice": 800,
    "maize": 650,
    "soybean": 700,
    "cotton": 750
}

# --------------------
# Prediction logic
# --------------------
def predict_from_series(model, agg_df: pd.DataFrame, lags: int):
    """Generate prediction from aggregated data"""
    X_all, _ = build_dataset(agg_df, lags=lags, horizon=1)
    
    if X_all.empty:
        raise ValueError("Not enough data to construct features")
    
    X_latest = X_all.iloc[[-1]]
    pred = float(model.predict(X_latest)[0])
    
    # Simple confidence based on data quality
    confidence = min(0.9, 0.5 + (len(agg_df) / 100) * 0.4)
    
    return pred, confidence

def naive_fallback_predict(agg_df: pd.DataFrame):
    """Fallback prediction when model unavailable"""
    if agg_df.empty or agg_df["moisture"].dropna().empty:
        raise ValueError("No moisture readings available")
    
    last = float(agg_df["moisture"].dropna().iloc[-1])
    # Simple decay model
    return max(0.0, last - 5.0), 0.3

# --------------------
# Prediction logging
# --------------------
async def log_prediction(
    db: AsyncIOMotorDatabase,
    prediction_id: str,
    device_id: Optional[str],
    crop_id: str,
    prediction: float,
    confidence: float,
    recommended: bool
):
    """Log predictions for monitoring"""
    try:
        await db.predictions.insert_one({
            "prediction_id": prediction_id,
            "device_id": device_id,
            "crop_id": crop_id,
            "predicted_moisture": prediction,
            "confidence": confidence,
            "recommend_irrigate": recommended,
            "model_version": MODEL_VERSION,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

# --------------------
# Endpoints
# --------------------
@app.get("/health", response_model=HealthResponse)
async def health(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Health check endpoint"""
    db_connected = False
    try:
        await db.command("ping")
        db_connected = True
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if (app_state.model and db_connected) else "degraded",
        model_loaded=app_state.model is not None,
        model_version=app_state.model_metadata.get("version"),
        database_connected=db_connected,
        training_status=app_state.training_status.get("status", "unknown"),
        timestamp=datetime.utcnow()
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    lags: int = 6,
    interval: int = 15
):
    """Predict future moisture levels"""
    prediction_id = str(uuid4())
    
    try:
        # Create agg_df from recentReadings or MongoDB
        if req.recentReadings:
            rows = []
            for r in req.recentReadings:
                rows.append({
                    "timestamp": pd.to_datetime(r.timestamp),
                    "moisture": r.moisture,
                    "temperature": r.temp,
                    "humidity": r.humidity
                })
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            agg_df = resample_and_aggregate(df, interval_minutes=interval)
        else:
            if not req.deviceId:
                raise HTTPException(
                    status_code=400,
                    detail="deviceId required when recentReadings not provided"
                )
            
            df = await fetch_sensor_data_async(
                db,
                collection=DEFAULT_COLLECTION,
                device_id=req.deviceId
            )
            
            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail="No sensor data available for device"
                )
            
            agg_df = resample_and_aggregate(df, interval_minutes=interval)
        
        # Generate prediction
        try:
            if app_state.model is not None:
                pred, confidence = predict_from_series(app_state.model, agg_df, lags=lags)
                method = "model"
            else:
                pred, confidence = naive_fallback_predict(agg_df)
                method = "heuristic"
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}, using fallback")
            pred, confidence = naive_fallback_predict(agg_df)
            method = "fallback"
        
        # Irrigation recommendation
        thresh = CROP_THRESHOLDS.get(req.cropId, 700)
        recommend = pred < thresh
        duration = max(0, min(60, int((thresh - pred) / 10))) if recommend else 0
        
        explanation = {
            "method": method,
            "predicted_value": round(pred, 2),
            "threshold": thresh,
            "crop_type": req.cropId,
            "data_points_used": len(agg_df)
        }
        
        # Log prediction asynchronously
        await log_prediction(
            db, prediction_id, req.deviceId, req.cropId,
            pred, confidence, recommend
        )
        
        return PredictResponse(
            prediction_id=prediction_id,
            predicted_moisture={"t+1h": round(pred, 2)},
            recommend_irrigate=bool(recommend),
            recommended_duration_minutes=int(duration),
            confidence=round(confidence, 2),
            explanation=explanation,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainResponse)
async def train(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Train model in background"""
    if app_state.training_status.get("status") == "training":
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )
    
    train_job_id = str(uuid4())
    
    background_tasks.add_task(
        train_model_task,
        train_job_id,
        req,
        db
    )
    
    app_state.training_status = {
        "status": "training",
        "job_id": train_job_id,
        "started_at": datetime.utcnow().isoformat()
    }
    
    return TrainResponse(
        status="started",
        train_job_id=train_job_id,
        message=f"Training started. Check /train-status/{train_job_id}"
    )

@app.get("/train-status/{job_id}")
async def train_status(job_id: str):
    """Get training job status"""
    if app_state.training_status.get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return app_state.training_status

# --------------------
# Background training task
# --------------------
async def train_model_task(job_id: str, req: TrainRequest, db: AsyncIOMotorDatabase):
    """Background task for model training"""
    try:
        logger.info(f"Starting training job {job_id}")
        
        # Fetch data
        df = await fetch_sensor_data_async(
            db,
            collection=req.collection or DEFAULT_COLLECTION,
            device_id=req.device_id
        )
        
        if df.empty:
            app_state.training_status = {
                "status": "failed",
                "job_id": job_id,
                "error": "No data found"
            }
            return
        
        # Prepare data
        agg = resample_and_aggregate(df, interval_minutes=req.interval)
        
        if agg["moisture"].isna().all():
            app_state.training_status = {
                "status": "failed",
                "job_id": job_id,
                "error": "All moisture values are NaN"
            }
            return
        
        X, y = build_dataset(agg, lags=req.lags, horizon=req.horizon)
        
        if X.empty:
            app_state.training_status = {
                "status": "failed",
                "job_id": job_id,
                "error": "Insufficient data after feature engineering"
            }
            return
        
        # Train/test split
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Train model
        model = XGBRegressor(
            n_estimators=req.n_estimators,
            learning_rate=req.learning_rate,
            max_depth=req.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate
        preds = model.predict(X_test)
        rmse = float(mean_squared_error(y_test, preds, squared=False))
        mae = float(mean_absolute_error(y_test, preds))
        
        # Save model
        out_path = req.out_path or MODEL_PATH
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump(model, out_path)
        
        # Reload model
        load_model()
        
        app_state.training_status = {
            "status": "completed",
            "job_id": job_id,
            "completed_at": datetime.utcnow().isoformat(),
            "metrics": {
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "train_rows": len(X_train),
                "test_rows": len(X_test)
            },
            "model_path": out_path
        }
        
        logger.info(f"Training job {job_id} completed. RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {traceback.format_exc()}")
        app_state.training_status = {
            "status": "failed",
            "job_id": job_id,
            "error": str(e)
        }

# --------------------
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Production: gunicorn -k uvicorn.workers.UvicornWorker main:app -w 4 -b 0.0.0.0:8000
# --------------------
