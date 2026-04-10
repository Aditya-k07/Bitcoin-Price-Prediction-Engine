"""
Ridge Regressor model for Bitcoin price prediction.

Lightweight alternative to XGBoost for fast training and predictions.
Uses time-series CV to pick alpha, log-target training for better fit on
trending prices, and residual-based confidence bands.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.features import rsi_from_close_history
from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")


class RidgePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics: ModelMetrics = None
        self.feature_columns: list = None
        self.residual_std = None
        self.log_target = True
        self._ensure_model_dir()

    def _ensure_model_dir(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _y_train_log(self, y: np.ndarray) -> np.ndarray:
        return np.log(np.clip(y.astype(float), 1e-6, None))

    def train(self, df: pd.DataFrame, feature_columns: list,
              target_col: str = "target", test_size: float = 0.2) -> ModelMetrics:
        self.feature_columns = feature_columns

        X = df[feature_columns].values.astype(float)
        y = df[target_col].values.astype(float)

        split_idx = int(len(X) * (1 - test_size))
        if split_idx < 10 or len(X) - split_idx < 5:
            raise ValueError(
                f"Not enough rows for Ridge train/test split (n={len(X)}). "
                "Need more historical candles."
            )

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        y_train_log = self._y_train_log(y_train)

        logger.info(f"Training Ridge: {len(X_train)} train, {len(X_test)} test samples (log-target)")

        # Time-series CV for alpha; fallback to fixed Ridge if too few rows
        n_train = len(X_train)
        if n_train < 50:
            self.model = Ridge(alpha=50.0, solver="auto")
            self.model.fit(X_train_scaled, y_train_log)
        else:
            n_splits = min(5, max(3, n_train // 120))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            alphas = np.logspace(-1, 5, 28)
            self.model = RidgeCV(
                alphas=alphas,
                cv=tscv,
                scoring="neg_mean_squared_error",
            )
            self.model.fit(X_train_scaled, y_train_log)

        y_pred_log = self.model.predict(X_test_scaled)
        y_pred = np.exp(np.clip(y_pred_log, -30.0, 40.0))

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))

        residuals = y_test - y_pred
        self.residual_std = float(np.sqrt(np.mean(residuals ** 2)))

        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = (
            float(np.mean(actual_direction == pred_direction)) * 100
            if len(actual_direction) > 0 else 0.0
        )

        tp = np.sum((actual_direction) & (pred_direction))
        tn = np.sum((~actual_direction) & (~pred_direction))
        fp = np.sum((~actual_direction) & (pred_direction))
        fn = np.sum((actual_direction) & (~pred_direction))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_score_val = float(f1) * 100

        threshold = np.mean(y_test) * 0.02
        accuracy = float(np.mean(np.abs(y_test - y_pred) <= threshold)) * 100

        chosen_alpha = getattr(self.model, "alpha_", None)
        if chosen_alpha is None:
            chosen_alpha = getattr(self.model, "alpha", None)

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
            f"Ridge training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, alpha={chosen_alpha}"
        )

        self._save()
        return self.metrics

    def predict(self, df: pd.DataFrame, days: int = 30) -> list:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")
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
            if self.log_target:
                price = float(np.exp(np.clip(raw, -30.0, 40.0)))
            else:
                price = max(raw, 0.0)

            lower = price - margin
            upper = price + margin
            price = max(price, 0.0)
            lower = max(lower, 0.0)
            upper = max(upper, price)

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "lower": round(lower, 2),
                "upper": round(upper, 2),
            })
            close_history.append(price)

        return predictions

    def _build_feature_vector(self, close_history: list[float]) -> np.ndarray:
        if not close_history:
            raise RuntimeError("close_history is empty")

        last_close = close_history[-1]
        features = {col: 0.0 for col in self.feature_columns}

        for col in self.feature_columns:
            if col.startswith("lag_"):
                lag = int(col.split("_")[1])
                idx = len(close_history) - lag
                features[col] = float(close_history[idx]) if idx >= 0 else float(close_history[0])
            elif col.startswith("sma_"):
                window = int(col.split("_")[1])
                sample = close_history[-window:] if len(close_history) >= window else close_history
                features[col] = float(np.mean(sample))
            elif col == "daily_return":
                if len(close_history) >= 2 and close_history[-2] != 0:
                    features[col] = float((close_history[-1] - close_history[-2]) / close_history[-2])
                else:
                    features[col] = 0.0
            elif col == "volatility":
                if len(close_history) >= 3:
                    ch = np.asarray(close_history, dtype=float)
                    prev = ch[:-1]
                    prev = np.where(np.abs(prev) < 1e-12, np.nan, prev)
                    returns = np.diff(ch) / prev
                    returns = returns[~np.isnan(returns)]
                    sample = returns[-14:] if len(returns) >= 14 else returns
                    features[col] = float(np.std(sample)) if len(sample) > 0 else 0.0
                else:
                    features[col] = 0.0
            elif col == "rsi":
                features[col] = rsi_from_close_history(close_history)
            elif col == "mom_7":
                if len(close_history) >= 8 and close_history[-8] != 0:
                    features[col] = float((close_history[-1] - close_history[-8]) / close_history[-8])
                else:
                    features[col] = 0.0
            elif col == "mom_14":
                if len(close_history) >= 15 and close_history[-15] != 0:
                    features[col] = float((close_history[-1] - close_history[-15]) / close_history[-15])
                else:
                    features[col] = 0.0
            elif col == "price_range":
                features[col] = 0.0
            elif col == "oc_diff":
                features[col] = 0.0
            else:
                features[col] = float(last_close)

        return np.array([[features[col] for col in self.feature_columns]], dtype=float)

    def _save(self):
        joblib.dump(self.model, os.path.join(MODEL_DIR, "ridge_model.pkl"))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "ridge_scaler.pkl"))
        joblib.dump(self.feature_columns, os.path.join(MODEL_DIR, "ridge_features.pkl"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "ridge_metrics.pkl"))
        joblib.dump(self.residual_std, os.path.join(MODEL_DIR, "ridge_residual_std.pkl"))
        joblib.dump(self.log_target, os.path.join(MODEL_DIR, "ridge_log_target.pkl"))
        logger.info("Ridge model saved to disk.")

    def load(self) -> bool:
        try:
            self.model = joblib.load(os.path.join(MODEL_DIR, "ridge_model.pkl"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "ridge_scaler.pkl"))
            self.feature_columns = joblib.load(os.path.join(MODEL_DIR, "ridge_features.pkl"))
            self.metrics = joblib.load(os.path.join(MODEL_DIR, "ridge_metrics.pkl"))
            self.residual_std = joblib.load(os.path.join(MODEL_DIR, "ridge_residual_std.pkl"))
            log_path = os.path.join(MODEL_DIR, "ridge_log_target.pkl")
            if os.path.isfile(log_path):
                self.log_target = joblib.load(log_path)
            else:
                self.log_target = False
            logger.info("Ridge model loaded from disk.")
            return True
        except FileNotFoundError:
            logger.warning("No saved Ridge model found.")
            return False

    @property
    def is_trained(self) -> bool:
        return self.model is not None
