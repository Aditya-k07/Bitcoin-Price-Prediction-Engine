"""
CoinSight ML Service — FastAPI Application

Serves Bitcoin price predictions via REST API.
Endpoints:
  GET  /predict?model=xgboost|prophet&days=30
  POST /retrain?model=xgboost|prophet
  GET  /health
"""

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.data_loader import load_daily_data
from app.features import engineer_features, get_feature_columns
from app.models.xgboost_model import XGBoostPredictor
from app.models.lstm_xgboost_model import LSTMXGBoostPredictor
from app.schemas import PredictionResponse, RetrainResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instances
xgboost_predictor = XGBoostPredictor()
lstm_predictor = LSTMXGBoostPredictor()

# Global data cache
_daily_data = None
_featured_data = None
_data_lock = threading.Lock()


def _load_data():
    """Load and cache the dataset."""
    global _daily_data, _featured_data
    with _data_lock:
        if _daily_data is None:
            csv_path = os.environ.get("BTC_DATA_PATH", None)
            _daily_data = load_daily_data(csv_path)
            _featured_data = engineer_features(_daily_data)
    return _daily_data, _featured_data


def _get_data():
    """Get cached data, loading if necessary."""
    if _daily_data is None:
        return _load_data()
    return _daily_data, _featured_data


def _try_load_models():
    """Try to load pre-trained models from disk on startup."""
    xgb_loaded = xgboost_predictor.load()
    lstm_loaded = lstm_predictor.load()

    if xgb_loaded:
        logger.info("XGBoost model loaded from saved state.")
    if lstm_loaded:
        logger.info("LSTM+XGBoost Hybrid model loaded from saved state.")

    return xgb_loaded, lstm_loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("🚀 CoinSight ML Service starting up...")

    # Try loading saved models (non-blocking — data loading happens on first request)
    _try_load_models()

    logger.info("✅ ML Service ready.")
    yield
    logger.info("🛑 ML Service shutting down.")


# Create FastAPI app
app = FastAPI(
    title="CoinSight ML Service",
    description="Bitcoin price prediction using XGBoost and Prophet",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status and which models are loaded.
    """
    models_loaded = []
    if xgboost_predictor.is_trained:
        models_loaded.append("xgboost")
    if lstm_predictor.is_trained:
        models_loaded.append("lstm_xgboost")

    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded
    )


@app.get("/predict", response_model=PredictionResponse)
async def predict(
    model: str = Query("xgboost", regex="^(xgboost|lstm_xgboost)$",
                       description="Model to use: 'xgboost' or 'lstm_xgboost'"),
    days: int = Query(30, ge=1, le=365,
                      description="Number of days to predict (1-365)")
):
    """
    Generate price predictions with confidence intervals.

    - **model**: 'xgboost' (Aggressive) or 'lstm_xgboost' (Hybrid Network)
    - **days**: Number of future days to predict (default: 30, max: 365)

    Returns predicted prices with 95% confidence bounds.
    """
    daily_data, featured_data = _get_data()

    if model == "xgboost":
        predictor = xgboost_predictor
        if not predictor.is_trained:
            # Auto-train on first request
            logger.info("XGBoost not trained, training now...")
            feature_cols = get_feature_columns(featured_data)
            predictor.train(featured_data, feature_cols)

        predictions = predictor.predict(featured_data, days=days)
        metrics = predictor.metrics

    elif model == "lstm_xgboost":
        predictor = lstm_predictor
        if not predictor.is_trained:
            logger.info("LSTM+XGBoost not trained, training now...")
            feature_cols = get_feature_columns(featured_data)
            predictor.train(featured_data, feature_cols)

        predictions = predictor.predict(featured_data, days=days)
        metrics = predictor.metrics

    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    logger.info(f"Raw predictions from model: {predictions[:2] if len(predictions) > 0 else 'EMPTY'}")
    
    # Convert predictions to OHLC format
    ohlc_predictions = []
    for pred in predictions:
        ohlc_predictions.append({
            "date": pred["date"],
            "open": pred["price"],
            "high": pred["upper"],
            "low": pred["lower"],
            "close": pred["price"]
        })
    
    logger.info(f"Converted OHLC predictions: {ohlc_predictions[:2] if len(ohlc_predictions) > 0 else 'EMPTY'}")

    return PredictionResponse(
        model=model,
        predictions=ohlc_predictions,
        rmse=metrics.rmse,
        mae=metrics.mae,
        r2_score=metrics.r2_score,
        mape=metrics.mape,
        f1_score=metrics.f1_score,
        accuracy=metrics.accuracy,
        directional_accuracy=metrics.directional_accuracy,
        trained_at=metrics.trained_at.isoformat() + "Z",
        architecture_details=metrics.architecture_details
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(
    model: str = Query("xgboost", regex="^(xgboost|lstm_xgboost)$",
                       description="Model to retrain: 'xgboost' or 'lstm_xgboost'")
):
    """
    Retrain a model and return updated metrics.

    This endpoint reloads the data, re-engineers features, and retrains
    the specified model from scratch. Useful when new data is available
    or when you want to refresh the model.
    """
    global _daily_data, _featured_data

    # Force reload data
    logger.info(f"Retraining {model} model...")
    with _data_lock:
        csv_path = os.environ.get("BTC_DATA_PATH", None)
        _daily_data = load_daily_data(csv_path)
        _featured_data = engineer_features(_daily_data)

    if model == "xgboost":
        feature_cols = get_feature_columns(_featured_data)
        metrics = xgboost_predictor.train(_featured_data, feature_cols)
    elif model == "lstm_xgboost":
        feature_cols = get_feature_columns(_featured_data)
        metrics = lstm_predictor.train(_featured_data, feature_cols)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    return RetrainResponse(
        model=model,
        rmse=metrics.rmse,
        mae=metrics.mae,
        r2_score=metrics.r2_score,
        mape=metrics.mape,
        f1_score=metrics.f1_score,
        accuracy=metrics.accuracy,
        directional_accuracy=metrics.directional_accuracy,
        trained_at=metrics.trained_at.isoformat() + "Z",
        message=f"{model.capitalize()} model retrained successfully",
        architecture_details=metrics.architecture_details
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
