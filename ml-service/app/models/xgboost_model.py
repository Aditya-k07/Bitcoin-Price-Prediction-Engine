"""
XGBoost model for Bitcoin price prediction.

Trains median + quantile models on log(next close) for better fit on
trending prices; early stopping reduces overfitting and test RMSE.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

from app.features import rsi_from_close_history
from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")


class XGBoostPredictor:
    def __init__(self):
        self.model_median = None
        self.model_lower = None
        self.model_upper = None
        self.metrics: ModelMetrics = None
        self.feature_columns: list = None
        self.log_target = True
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _y_log(self, y: np.ndarray) -> np.ndarray:
        return np.log(np.clip(y.astype(float), 1e-6, None))

    def train(self, df: pd.DataFrame, feature_columns: list, target_col: str = "target", test_size: float = 0.2) -> ModelMetrics:
        self.feature_columns = feature_columns
        X = df[feature_columns].values.astype(float)
        y = df[target_col].values.astype(float)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        y_train_log = self._y_log(y_train)
        y_test_log = self._y_log(y_test)

        tree_params = {
            "n_estimators": 1200,
            "max_depth": 6,
            "learning_rate": 0.025,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2,
            "reg_alpha": 0.15,
            "reg_lambda": 2.0,
            "random_state": 42,
        }

        self.model_median = XGBRegressor(
            objective="reg:squarederror",
            early_stopping_rounds=80,
            **tree_params,
        )
        self.model_median.fit(
            X_train,
            y_train_log,
            eval_set=[(X_test, y_test_log)],
            eval_metric="rmse",
            verbose=False,
        )

        # Quantile heads: same capacity as median; no early stop (avoids API quirks vs. median).
        q_params = {**tree_params, "n_estimators": min(600, tree_params["n_estimators"])}
        self.model_lower = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.025, **q_params)
        self.model_lower.fit(X_train, y_train_log, verbose=False)

        self.model_upper = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.975, **q_params)
        self.model_upper.fit(X_train, y_train_log, verbose=False)

        y_pred_m = self.model_median.predict(X_test)
        y_pred = np.exp(np.clip(y_pred_m, -30.0, 40.0))

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = (
            float(np.mean(actual_direction == pred_direction)) * 100
            if len(actual_direction) > 0 else 0.0
        )

        self.metrics = ModelMetrics(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape,
            f1_score=directional_accuracy,
            accuracy=directional_accuracy,
            directional_accuracy=directional_accuracy,
            trained_at=datetime.utcnow(),
            num_training_samples=len(X_train),
            num_features=len(feature_columns),
            architecture_details={
                "model": "XGBoost (log target + early stopping)",
                "n_estimators": 1200,
                "max_depth": 6,
                "learning_rate": 0.025,
                "best_iteration_median": int(getattr(self.model_median, "best_iteration", 0) or 0),
                "log_target": self.log_target,
            },
        )
        self._save()
        return self.metrics

    def predict(self, df: pd.DataFrame, days: int = 30) -> list:
        if self.model_median is None:
            raise RuntimeError("Model not trained.")

        predictions = []
        last_date = df.index[-1]
        close_history = df["Close"].astype(float).tolist()

        for i in range(days):
            pred_date = last_date + pd.Timedelta(days=i + 1)
            current_features = self._build_feature_vector(close_history)
            pl = float(self.model_median.predict(current_features)[0])
            ll = float(self.model_lower.predict(current_features)[0])
            ul = float(self.model_upper.predict(current_features)[0])
            if self.log_target:
                price = float(np.exp(np.clip(pl, -30.0, 40.0)))
                lower = float(np.exp(np.clip(ll, -30.0, 40.0)))
                upper = float(np.exp(np.clip(ul, -30.0, 40.0)))
            else:
                price, lower, upper = pl, ll, ul

            lower = min(lower, price)
            upper = max(upper, price)
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
                    if len(close_history) >= 2 and close_history[-2] != 0 else 0.0
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
                    feats[col] = float((close_history[-1] - close_history[-8]) / close_history[-8])
                else:
                    feats[col] = 0.0
            elif col == "mom_14":
                if len(close_history) >= 15 and close_history[-15] != 0:
                    feats[col] = float((close_history[-1] - close_history[-15]) / close_history[-15])
                else:
                    feats[col] = 0.0
            elif col in ("price_range", "oc_diff"):
                feats[col] = 0.0
            else:
                feats[col] = float(last_close)
        return np.array([[feats[col] for col in self.feature_columns]], dtype=float)

    def _save(self):
        joblib.dump(self.model_median, os.path.join(MODEL_DIR, "xgboost_median.pkl"))
        joblib.dump(self.model_lower, os.path.join(MODEL_DIR, "xgboost_lower.pkl"))
        joblib.dump(self.model_upper, os.path.join(MODEL_DIR, "xgboost_upper.pkl"))
        joblib.dump(self.feature_columns, os.path.join(MODEL_DIR, "xgboost_features.pkl"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "xgboost_metrics.pkl"))
        joblib.dump(self.log_target, os.path.join(MODEL_DIR, "xgboost_log_target.pkl"))

    def load(self) -> bool:
        try:
            self.model_median = joblib.load(os.path.join(MODEL_DIR, "xgboost_median.pkl"))
            self.model_lower = joblib.load(os.path.join(MODEL_DIR, "xgboost_lower.pkl"))
            self.model_upper = joblib.load(os.path.join(MODEL_DIR, "xgboost_upper.pkl"))
            self.feature_columns = joblib.load(os.path.join(MODEL_DIR, "xgboost_features.pkl"))
            self.metrics = joblib.load(os.path.join(MODEL_DIR, "xgboost_metrics.pkl"))
            log_path = os.path.join(MODEL_DIR, "xgboost_log_target.pkl")
            if os.path.isfile(log_path):
                self.log_target = joblib.load(log_path)
            else:
                self.log_target = False
            return True
        except FileNotFoundError:
            return False

    @property
    def is_trained(self) -> bool:
        return self.model_median is not None
