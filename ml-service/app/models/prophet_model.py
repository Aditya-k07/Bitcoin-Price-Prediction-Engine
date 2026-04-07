"""
Prophet model for Bitcoin price prediction.

Uses Facebook Prophet for time-series forecasting with built-in
uncertainty intervals. This is the "Conservative" model option.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error

from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")


class ProphetPredictor:
    """
    Prophet-based price predictor with built-in uncertainty intervals.

    Prophet handles:
    - Trend detection (linear or logistic growth)
    - Seasonality (daily, weekly — though crypto trades 24/7)
    - Holiday effects (not relevant for crypto, but available)
    - Built-in uncertainty intervals (95% by default)

    This is the "Conservative" model: Prophet tends to produce smoother,
    less volatile predictions compared to XGBoost.
    """

    def __init__(self):
        self.model = None
        self.metrics: ModelMetrics = None
        self._ensure_model_dir()

    def _ensure_model_dir(self):
        """Create saved_models directory if it doesn't exist."""
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> ModelMetrics:
        """
        Train Prophet model on daily close prices.

        Prophet requires a DataFrame with columns:
        - ds: datetime (the date)
        - y: float (the value to predict — close price)

        Uses time-based split for evaluation.

        Args:
            df: Daily OHLCV DataFrame with datetime index.
            test_size: Fraction of data for testing.

        Returns:
            ModelMetrics with RMSE and training metadata.
        """
        # Suppress Prophet's verbose output
        import cmdstanpy
        cmdstanpy_logger = logging.getLogger("cmdstanpy")
        cmdstanpy_logger.setLevel(logging.WARNING)

        from prophet import Prophet

        # Prepare Prophet-format DataFrame
        prophet_df = pd.DataFrame({
            "ds": df.index,
            "y": df["Close"].values
        })

        # Time-based split
        split_idx = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]

        logger.info(f"Training Prophet: {len(train_df)} train, {len(test_df)} test samples")

        # Initialize and train Prophet
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95,  # 95% confidence interval
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
        )

        # Suppress fitting output
        self.model.fit(train_df)

        # Evaluate on test set
        future_test = self.model.make_future_dataframe(periods=len(test_df))
        forecast_test = self.model.predict(future_test)

        # Get predictions for the test period only
        test_predictions = forecast_test.iloc[split_idx:]["yhat"].values
        test_actuals = test_df["y"].values

        # Trim to matching length (Prophet might produce slightly different lengths)
        min_len = min(len(test_predictions), len(test_actuals))
        test_predictions = test_predictions[:min_len]
        test_actuals = test_actuals[:min_len]

        rmse = float(np.sqrt(mean_squared_error(test_actuals, test_predictions)))

        self.metrics = ModelMetrics(
            rmse=rmse,
            trained_at=datetime.utcnow(),
            num_training_samples=len(train_df)
        )

        logger.info(f"Prophet training complete. RMSE: {rmse:.2f}")

        # Save model to disk
        self._save()

        return self.metrics

    def predict(self, df: pd.DataFrame, days: int = 30) -> list:
        """
        Generate multi-day predictions with Prophet's built-in uncertainty.

        Prophet has native support for future forecasting with uncertainty
        intervals — no need for the quantile regression trick used in XGBoost.

        Args:
            df: Daily OHLCV DataFrame (used to get the last known date).
            days: Number of days to predict into the future.

        Returns:
            List of dicts with date, price, lower, upper for each predicted day.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        # Create future DataFrame
        prophet_df = pd.DataFrame({
            "ds": df.index,
            "y": df["Close"].values
        })

        future = self.model.make_future_dataframe(periods=days)
        forecast = self.model.predict(future)

        # Extract only the future predictions (last N days)
        future_forecast = forecast.tail(days)

        predictions = []
        for _, row in future_forecast.iterrows():
            predictions.append({
                "date": row["ds"].strftime("%Y-%m-%d"),
                "price": round(float(row["yhat"]), 2),
                "lower": round(float(row["yhat_lower"]), 2),
                "upper": round(float(row["yhat_upper"]), 2)
            })

        return predictions

    def _save(self):
        """Save Prophet model and metrics to disk."""
        joblib.dump(self.model, os.path.join(MODEL_DIR, "prophet_model.pkl"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "prophet_metrics.pkl"))
        logger.info("Prophet model saved to disk.")

    def load(self) -> bool:
        """
        Load saved Prophet model from disk.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        try:
            self.model = joblib.load(os.path.join(MODEL_DIR, "prophet_model.pkl"))
            self.metrics = joblib.load(os.path.join(MODEL_DIR, "prophet_metrics.pkl"))
            logger.info("Prophet model loaded from disk.")
            return True
        except FileNotFoundError:
            logger.warning("No saved Prophet model found.")
            return False

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained or loaded."""
        return self.model is not None
