"""
Ridge Regressor model for Bitcoin price prediction.

Lightweight alternative to XGBoost for fast training and predictions.
Uses RidgeCV with time-series cross-validation to pick the best alpha,
log-target training for better fit on trending prices, and residual-based
95 % confidence intervals.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.features import rsi_from_close_history
from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models"
)


class RidgePredictor:
    """
    Ridge Regression-based price predictor with confidence intervals.

    Trains a Ridge regressor and approximates 95 % confidence intervals using
    residual standard error on the test set.

    Fast, lightweight, and suitable for real-time predictions.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics: ModelMetrics = None
        self.feature_columns: list = None
        self.residual_std = None
        self.log_target = True
        os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _y_log(self, y: np.ndarray) -> np.ndarray:
        return np.log(np.clip(y.astype(float), 1e-6, None))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        target_col: str = "target",
        test_size: float = 0.2,
    ) -> ModelMetrics:
        """
        Train Ridge Regression model.

        Uses time-based split (no shuffle) to avoid data leakage.

        Args:
            df: Feature-engineered DataFrame.
            feature_columns: List of feature column names.
            target_col: Name of the target column.
            test_size: Fraction of data to use for testing.

        Returns:
            ModelMetrics with evaluation metrics and training metadata.
        """
        self.feature_columns = feature_columns

        X = df[feature_columns].values.astype(float)
        y = df[target_col].values.astype(float)

        # Time-based split — never shuffle for time series!
        split_idx = int(len(X) * (1 - test_size))
        if split_idx < 5 or len(X) - split_idx < 3:
            raise ValueError(
                f"Not enough rows for Ridge train/test split (n={len(X)}). "
                "Need more historical candles (at least 8-10 days)."
            )
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            "Training Ridge: %d train, %d test samples (log-target=%s)",
            len(X_train), len(X_test), self.log_target,
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply log transform to target if requested
        y_train_fit = self._y_log(y_train) if self.log_target else y_train

        # Time-series CV for alpha selection; fallback to plain Ridge for tiny datasets
        n_train = len(X_train)
        if n_train < 50:
            self.model = Ridge(alpha=50.0, solver="auto")
            self.model.fit(X_train_scaled, y_train_fit)
        else:
            n_splits = min(5, max(3, n_train // 120))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            alphas = np.logspace(-1, 5, 28)
            self.model = RidgeCV(
                alphas=alphas,
                cv=tscv,
                scoring="neg_mean_squared_error",
            )
            self.model.fit(X_train_scaled, y_train_fit)

        # Predict on test set (invert log if needed)
        y_pred_raw = self.model.predict(X_test_scaled)
        if self.log_target:
            y_pred = np.exp(np.clip(y_pred_raw, -30.0, 40.0))
        else:
            y_pred = y_pred_raw

        # --- Metrics ---
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))

        # Residual std for confidence intervals
        residuals = y_test - y_pred
        self.residual_std = float(np.sqrt(np.mean(residuals ** 2)))

        # Directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = (
            float(np.mean(actual_direction == pred_direction)) * 100
            if len(actual_direction) > 0
            else 0.0
        )

        # F1 approximation via direction classification
        tp = int(np.sum(actual_direction & pred_direction))
        fp = int(np.sum(~actual_direction & pred_direction))
        fn = int(np.sum(actual_direction & ~pred_direction))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score_val = (
            float(2 * precision * recall / (precision + recall)) * 100
            if (precision + recall) > 0
            else 0.0
        )

        # % of predictions within 2 % of true price
        threshold = np.mean(y_test) * 0.02
        accuracy = float(np.mean(np.abs(y_test - y_pred) <= threshold)) * 100

        chosen_alpha = getattr(self.model, "alpha_", None) or getattr(self.model, "alpha", None)

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
                "model": "Ridge Regression (RidgeCV + log target)",
                "alpha": float(chosen_alpha) if chosen_alpha is not None else None,
                "residual_std": round(self.residual_std, 2),
                "log_target": self.log_target,
                "strategy": "95% CI via residual std on price scale",
            },
        )

        logger.info(
            "Ridge training complete. RMSE: %.2f, MAE: %.2f, R²: %.4f, alpha=%s",
            rmse, mae, r2, chosen_alpha,
        )

        self._save()
        return self.metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame, days: int = 30) -> list:
        """
        Generate multi-day predictions with confidence intervals.

        Uses recursive forecasting with residual-based 95 % confidence intervals.

        Args:
            df: Feature-engineered DataFrame.
            days: Number of days to predict.

        Returns:
            List of dicts with date, price, lower, upper for each predicted day.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError(
                "Model not trained. Call train() first or load a saved model."
            )
        if self.residual_std is None:
            raise RuntimeError("Missing residual_std; retrain the Ridge model.")

        predictions = []
        last_date = df.index[-1]
        close_history = df["Close"].astype(float).tolist()

        margin = 1.96 * self.residual_std

        for i in range(days):
            pred_date = last_date + pd.Timedelta(days=i + 1)

            current_features = self._build_feature_vector(close_history)
            current_scaled = self.scaler.transform(current_features)
            raw = float(self.model.predict(current_scaled)[0])

            price = float(np.exp(np.clip(raw, -30.0, 40.0))) if self.log_target else raw
            
            # Enforce logical constraints: lower <= price <= upper, and all >= 0
            price = max(price, 0.0)
            lower = max(price - margin, 0.0)
            upper = max(price + margin, price)


            predictions.append(
                {
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "price": round(price, 2),
                    "lower": round(lower, 2),
                    "upper": round(upper, 2),
                }
            )

            close_history.append(price)

        return predictions

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _build_feature_vector(self, close_history: list) -> np.ndarray:
        """Build a feature vector from a rolling close-price history list."""
        if not close_history:
            raise RuntimeError("close_history is empty")

        last_close = close_history[-1]
        feats = {col: 0.0 for col in self.feature_columns}

        for col in self.feature_columns:
            if col.startswith("lag_"):
                lag = int(col.split("_")[1])
                idx = len(close_history) - lag
                feats[col] = float(close_history[idx]) if idx >= 0 else float(close_history[0])
            elif col.startswith("sma_"):
                window = int(col.split("_")[1])
                sample = close_history[-window:] if len(close_history) >= window else close_history
                feats[col] = float(np.mean(sample))
            elif col == "daily_return":
                feats[col] = (
                    float((close_history[-1] - close_history[-2]) / close_history[-2])
                    if len(close_history) >= 2 and close_history[-2] != 0
                    else 0.0
                )
            elif col == "volatility":
                if len(close_history) >= 3:
                    ch = np.asarray(close_history, dtype=float)
                    prev = ch[:-1]
                    prev = np.where(np.abs(prev) < 1e-12, np.nan, prev)
                    returns = np.diff(ch) / prev
                    returns = returns[~np.isnan(returns)]
                    sample = returns[-14:] if len(returns) >= 14 else returns
                    feats[col] = float(np.std(sample)) if len(sample) > 0 else 0.0
                else:
                    feats[col] = 0.0
            elif col == "rsi":
                feats[col] = rsi_from_close_history(close_history)
            elif col == "mom_7":
                if len(close_history) >= 8 and close_history[-8] != 0:
                    feats[col] = float(
                        (close_history[-1] - close_history[-8]) / close_history[-8]
                    )
            elif col == "mom_14":
                if len(close_history) >= 15 and close_history[-15] != 0:
                    feats[col] = float(
                        (close_history[-1] - close_history[-15]) / close_history[-15]
                    )
            elif col in ("price_range", "oc_diff"):
                feats[col] = 0.0
            else:
                feats[col] = float(last_close)

        return np.array([[feats[col] for col in self.feature_columns]], dtype=float)

    def _update_features_for_next_step(
        self, features: np.ndarray, new_price: float
    ) -> np.ndarray:
        """Shift lag values in the feature vector to incorporate the latest prediction."""
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        """Save model and metadata to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, os.path.join(MODEL_DIR, "ridge_model.pkl"))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "ridge_scaler.pkl"))
        joblib.dump(self.feature_columns, os.path.join(MODEL_DIR, "ridge_features.pkl"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "ridge_metrics.pkl"))
        joblib.dump(self.residual_std, os.path.join(MODEL_DIR, "ridge_residual_std.pkl"))
        joblib.dump(self.log_target, os.path.join(MODEL_DIR, "ridge_log_target.pkl"))
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

            log_path = os.path.join(MODEL_DIR, "ridge_log_target.pkl")
            self.log_target = joblib.load(log_path) if os.path.isfile(log_path) else False

            logger.info("Ridge model loaded from disk.")
            return True
        except FileNotFoundError:
            logger.warning("No saved Ridge model found.")
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained or loaded."""
        return self.model is not None
