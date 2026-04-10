"""
Pydantic schemas for ML service request/response models.
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class PredictionPoint(BaseModel):
    """A single prediction data point with confidence bounds."""
    date: str
    open: float
    high: float
    low: float
    close: float


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    model: str
    predictions: List[PredictionPoint]
    rmse: float
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    mape: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    directional_accuracy: Optional[float] = None
    trained_at: str
    architecture_details: Optional[dict] = None


class RetrainResponse(BaseModel):
    """Response from the /retrain endpoint."""
    model: str
    rmse: float
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    mape: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    directional_accuracy: Optional[float] = None
    trained_at: str
    message: str
    architecture_details: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    models_loaded: List[str]


class ModelMetrics(BaseModel):
    """Internal model tracking metrics."""
    rmse: float
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    mape: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    directional_accuracy: Optional[float] = None
    trained_at: datetime
    num_training_samples: int
    num_features: Optional[int] = None
    architecture_details: Optional[dict] = None


class CandleData(BaseModel):
    """OHLC candlestick data point from CoinGecko."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float


class HistoricalDataRequest(BaseModel):
    """Request payload with fresh OHLC data for predictions."""
    model: str
    days: int
    data: List[CandleData]
