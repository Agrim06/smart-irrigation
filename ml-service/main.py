# ml-service/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
import os
import json
import joblib
import numpy as np
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import logging
import traceback
from uuid import uuid4

# Import modular components using package-style imports
from src.feature_engineering import resample_and_aggregate, build_dataset
from src.predict import (
    load_model,
    predict_from_series,
    naive_fallback_predict,
    recommend_irrigation,
    CROP_THRESHOLDS,
    MODEL_VERSION as PRED_MODEL_VERSION
)
from src.preprocessing import fetch_sensor_data_async
from src.train_model import train_model

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
    app_state.model = load_model()
    app_state.training_status = {"status": "idle"}
    
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        model_version=MODEL_VERSION,
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
        # Build agg_df
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

        # Prediction
        if app_state.model is not None:
            pred, confidence = predict_from_series(app_state.model, agg_df, lags=lags)
            method = "model"
        else:
            pred, confidence = naive_fallback_predict(agg_df)
            method = "fallback"

        # Irrigation decision
        thresh, duration, thresh_val = CROP_THRESHOLDS.get(req.cropId, (700, 0, 700))
        recommend, duration = recommend_irrigation(pred, req.cropId)
        explanation = {
            "method": method,
            "predicted_value": round(pred, 2),
            "threshold": thresh_val,
            "crop_type": req.cropId,
            "data_points_used": len(agg_df)
        }

        # Log prediction (optional)
        # You can add an async log to Mongo here if desired

        return PredictResponse(
            prediction_id=prediction_id,
            predicted_moisture={"t+1h": round(pred, 2)},
            recommend_irrigate=recommend,
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
    if app_state.training_status.get("status") == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    train_job_id = str(uuid4())

    async def _train_model_task():
        try:
            df = await fetch_sensor_data_async(
                db,
                collection=req.collection or DEFAULT_COLLECTION,
                device_id=req.device_id
            )
            if df.empty:
                app_state.training_status = {"status": "failed", "job_id": train_job_id, "error": "No data found"}
                return
            agg = resample_and_aggregate(df, interval_minutes=req.interval)
            if agg["moisture"].isna().all():
                app_state.training_status = {"status": "failed", "job_id": train_job_id, "error": "All moisture values are NaN"}
                return
            X, y = build_dataset(agg, lags=req.lags, horizon=req.horizon)
            if X.empty:
                app_state.training_status = {"status": "failed", "job_id": train_job_id, "error": "Insufficient data after feature engineering"}
                return
            model, metrics = train_model(
                X, y,
                n_estimators=req.n_estimators,
                learning_rate=req.learning_rate,
                max_depth=req.max_depth,
                out_path=req.out_path
            )
            app_state.model = model
            app_state.training_status = {
                "status": "completed",
                "job_id": train_job_id,
                "completed_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "model_path": req.out_path
            }
        except Exception as e:
            logger.error(f"Training job {train_job_id} failed: {e}")
            app_state.training_status = {"status": "failed", "job_id": train_job_id, "error": str(e)}

    background_tasks.add_task(_train_model_task)
    app_state.training_status = {"status": "training", "job_id": train_job_id, "started_at": datetime.utcnow().isoformat()}
    return TrainResponse(status="started", train_job_id=train_job_id, message=f"Training started. Check /train-status/{train_job_id}")

@app.get("/train-status/{job_id}")
async def train_status(job_id: str):
    if app_state.training_status.get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return app_state.training_status

# Run with: uvicorn main:app --reload --port 8000
