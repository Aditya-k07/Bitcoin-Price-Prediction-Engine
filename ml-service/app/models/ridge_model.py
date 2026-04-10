"""
Ridge Regressor model for Bitcoin price prediction.

Lightweight alternative to LSTM for fast training and predictions.
Uses Ridge regression with confidence intervals via quantile regression approximation.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")


class RidgePredictor:
    """
    Ridge Regression-based price predictor with confidence intervals.

    Trains a Ridge regressor and approximates confidence intervals using
    residual standard error and prediction intervals.
    
    Fast, lightweight, and suitable for real-time predictions.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics: ModelMetrics = None
        self.feature_columns: list = None
        self.residual_std = None
        self._ensure_model_dir()

    def _ensure_model_dir(self):
        """Create saved_models directory if it doesn't exist."""
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train(self, df: pd.DataFrame, feature_columns: list,
              target_col: str = "target", test_size: float = 0.2) -> ModelMetrics:
        """
        Train Ridge Regression model.

        Uses time-based split (no shuffle) to avoid data leakage.

        Args:
            df: Feature-engineered DataFrame.
            feature_columns: List of feature column names.
            target_col: Name of the target column.
            test_size: Fraction of data to use for testing.

        Returns:
            ModelMetrics with RMSE and training metadata.
        """
        self.feature_columns = feature_columns

        X = df[feature_columns].values
        y = df[target_col].values

        # Time-based split (no shuffle — critical for time series!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Training Ridge Regressor: {len(X_train)} train, {len(X_test)} test samples")

        # Scale features for Ridge regression
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Ridge model
        self.model = Ridge(alpha=1.0, solver='auto', random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))

        # Calculate residual standard error for confidence intervals
        residuals = y_test - y_pred
        self.residual_std = float(np.sqrt(np.mean(residuals ** 2)))

        # Directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = float(np.mean(actual_direction == pred_direction)) * 100 if len(actual_direction) > 0 else 0.0

        # F1 Score approximation
        tp = np.sum((actual_direction) & (pred_direction))
        tn = np.sum((~actual_direction) & (~pred_direction))
        fp = np.sum((~actual_direction) & (pred_direction))
        fn = np.sum((actual_direction) & (~pred_direction))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_score_val = float(f1) * 100

        # Accuracy
        threshold = np.mean(y_test) * 0.02
        accuracy = float(np.mean(np.abs(y_test - y_pred) <= threshold)) * 100

        self.metrics = ModelMetrics(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape,
            f1_score=f1_score_val,
            accuracy=accuracy,
            directional_accuracy=directional_accuracy,
            trained_at=datetime.utcnow(),
            num_training_samples=len(X_train),
            num_features=len(feature_columns),
            architecture_details={
                "model": "Ridge Regression (Fast & Lightweight)",
                "alpha": 1.0,
                "residual_std": round(self.residual_std, 2),
                "strategy": "95% confidence intervals via residual standard error"
            }
        )

        logger.info(f"Ridge training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, F1: {f1_score_val:.2f}%, Accuracy: {accuracy:.2f}%")

        # Save model to disk
        self._save()

        return self.metrics

    def predict(self, df: pd.DataFrame, days: int = 30) -> list:
        """
        Generate multi-day predictions with confidence intervals.

        Uses recursive forecasting with residual-based confidence intervals.

        Args:
            df: Feature-engineered DataFrame.
            days: Number of days to predict.

        Returns:
            List of dicts with date, price, lower, upper for each predicted day.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        predictions = []
        last_date = df.index[-1]
        current_features = df[self.feature_columns].iloc[-1:].values.copy()

        for i in range(days):
            pred_date = last_date + pd.Timedelta(days=i + 1)

            # Scale and predict
            current_scaled = self.scaler.transform(current_features)
            price = float(self.model.predict(current_scaled)[0])

            # Confidence intervals: ±1.96 * residual_std (95% CI)
            margin = 1.96 * self.residual_std
            lower = price - margin
            upper = price + margin

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "lower": round(lower, 2),
                "upper": round(upper, 2)
            })

            # Update features for next step (shift lags)
            current_features = self._update_features_for_next_step(current_features, price)

        return predictions

    def _update_features_for_next_step(self, features: np.ndarray,
                                       new_price: float) -> np.ndarray:
        """
        Update the feature vector for the next recursive prediction step.

        Shifts lag values: lag_1 ← new_price, lag_3 ← old lag_1, etc.

        Args:
            features: Current feature vector (1 x num_features).
            new_price: Predicted price to incorporate.

        Returns:
            Updated feature vector.
        """
        updated = features.copy()

        if self.feature_columns:
            for i, col in enumerate(self.feature_columns):
                if col == "lag_1":
                    updated[0, i] = new_price
                elif col == "lag_3" and "lag_1" in self.feature_columns:
                    lag1_idx = self.feature_columns.index("lag_1")
                    updated[0, i] = features[0, lag1_idx]
                elif col == "lag_7" and "lag_3" in self.feature_columns:
                    lag3_idx = self.feature_columns.index("lag_3")
                    updated[0, i] = features[0, lag3_idx]

        return updated

    def _save(self):
        """Save model to disk."""
        joblib.dump(self.model, os.path.join(MODEL_DIR, "ridge_model.pkl"))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "ridge_scaler.pkl"))
        joblib.dump(self.feature_columns, os.path.join(MODEL_DIR, "ridge_features.pkl"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "ridge_metrics.pkl"))
        joblib.dump(self.residual_std, os.path.join(MODEL_DIR, "ridge_residual_std.pkl"))
        logger.info("Ridge model saved to disk.")

    def load(self) -> bool:
        """
        Load saved model from disk.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        try:
            self.model = joblib.load(os.path.join(MODEL_DIR, "ridge_model.pkl"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "ridge_scaler.pkl"))
            self.feature_columns = joblib.load(os.path.join(MODEL_DIR, "ridge_features.pkl"))
            self.metrics = joblib.load(os.path.join(MODEL_DIR, "ridge_metrics.pkl"))
            self.residual_std = joblib.load(os.path.join(MODEL_DIR, "ridge_residual_std.pkl"))
            logger.info("Ridge model loaded from disk.")
            return True
        except FileNotFoundError:
            logger.warning("No saved Ridge model found.")
            return False

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained or loaded."""
        return self.model is not None
