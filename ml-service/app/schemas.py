"""
Pydantic schemas for ML service request/response models.
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class PredictionPoint(BaseModel):
    """A single prediction data point with confidence bounds."""
    date: str
    price: float
    lower: float
    upper: float


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    model: str
    predictions: List[PredictionPoint]
    rmse: float
    trained_at: str


class RetrainResponse(BaseModel):
    """Response from the /retrain endpoint."""
    model: str
    rmse: float
    trained_at: str
    message: str


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    models_loaded: List[str]


class ModelMetrics(BaseModel):
    """Internal model tracking metrics."""
    rmse: float
    trained_at: datetime
    num_training_samples: int
    num_features: Optional[int] = None
