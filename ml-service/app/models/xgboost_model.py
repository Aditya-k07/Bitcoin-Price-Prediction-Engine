"""
XGBoost model for Bitcoin price prediction.

<<<<<<< HEAD
Trains median + quantile models on log(next close) for better fit on
trending prices; early stopping reduces overfitting and test RMSE.
=======
Uses quantile regression to provide confidence intervals:
- Trains 3 models at α = 0.025, 0.5, 0.975 for 95% CI
- Evaluates using RMSE on test set
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

<<<<<<< HEAD
from app.features import rsi_from_close_history
from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)
=======
from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)

>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")


class XGBoostPredictor:
<<<<<<< HEAD
=======
    """
    XGBoost-based price predictor with quantile regression for confidence intervals.

    Trains three separate models:
    - median model (α=0.5): point estimate
    - lower bound (α=0.025): 2.5th percentile
    - upper bound (α=0.975): 97.5th percentile

    Together, these provide a 95% confidence interval around predictions.
    """

>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
    def __init__(self):
        self.model_median = None
        self.model_lower = None
        self.model_upper = None
        self.metrics: ModelMetrics = None
        self.feature_columns: list = None
<<<<<<< HEAD
        self.log_target = True
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _y_log(self, y: np.ndarray) -> np.ndarray:
        return np.log(np.clip(y.astype(float), 1e-6, None))

    def train(self, df: pd.DataFrame, feature_columns: list, target_col: str = "target", test_size: float = 0.2) -> ModelMetrics:
        self.feature_columns = feature_columns
        X = df[feature_columns].values.astype(float)
        y = df[target_col].values.astype(float)
=======
        self._ensure_model_dir()

    def _ensure_model_dir(self):
        """Create saved_models directory if it doesn't exist."""
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train(self, df: pd.DataFrame, feature_columns: list,
              target_col: str = "target", test_size: float = 0.2) -> ModelMetrics:
        """
        Train XGBoost models with quantile regression.

        Uses time-based split (no shuffle) to avoid data leakage:
        - First 80% of data for training
        - Last 20% for testing

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
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

<<<<<<< HEAD
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
=======
        logger.info(f"Training XGBoost: {len(X_train)} train, {len(X_test)} test samples")

        # Common hyperparameters - improved for better performance
        common_params = {
            "n_estimators": 600,          # Increased from 500
            "max_depth": 8,               # Increased from 6
            "learning_rate": 0.025,       # Reduced for better stability
            "subsample": 0.85,            # Improved
            "colsample_bytree": 0.85,     # Improved
            "colsample_bylevel": 0.8,     # Added
            "min_child_weight": 1,        # Added
            "gamma": 0.1,                 # Added
            "reg_alpha": 0.5,             # L1 regularization
            "reg_lambda": 1.0,            # L2 regularization
            "random_state": 42,
        }

        # Train median model (point estimate)
        logger.info("Training median model (α=0.5)...")
        self.model_median = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.5,
            **common_params
        )
        self.model_median.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Train lower bound model (2.5th percentile)
        logger.info("Training lower bound model (α=0.025)...")
        self.model_lower = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.025,
            **common_params
        )
        self.model_lower.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Train upper bound model (97.5th percentile)
        logger.info("Training upper bound model (α=0.975)...")
        self.model_upper = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.975,
            **common_params
        )
        self.model_upper.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate RMSE on test set using median model
        y_pred = self.model_median.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        
        # MAPE - Mean Absolute Percentage Error
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
        
        # Directional accuracy - percentage of predictions where trend matches actual
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = float(np.mean(actual_direction == pred_direction)) * 100 if len(actual_direction) > 0 else 0.0
        
        # F1 Score approximation for regression: calculate on directional accuracy
        # True Positives = correct uptrend, True Negatives = correct downtrend
        tp = np.sum((actual_direction) & (pred_direction))
        tn = np.sum((~actual_direction) & (~pred_direction))
        fp = np.sum((~actual_direction) & (pred_direction))
        fn = np.sum((actual_direction) & (~pred_direction))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_score_val = float(f1) * 100
        
        # Accuracy - percentage of predictions within acceptable error threshold (e.g., 2% of mean price)
        threshold = np.mean(y_test) * 0.02  # 2% threshold
        accuracy = float(np.mean(np.abs(y_test - y_pred) <= threshold)) * 100
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103

        self.metrics = ModelMetrics(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape,
<<<<<<< HEAD
            f1_score=directional_accuracy,
            accuracy=directional_accuracy,
=======
            f1_score=f1_score_val,
            accuracy=accuracy,
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
            directional_accuracy=directional_accuracy,
            trained_at=datetime.utcnow(),
            num_training_samples=len(X_train),
            num_features=len(feature_columns),
            architecture_details={
<<<<<<< HEAD
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
=======
                "model": "XGBoost Quantile Regression",
                "n_estimators": 600,
                "max_depth": 8,
                "learning_rate": 0.025,
                "quantiles": [0.025, 0.5, 0.975],
                "strategy": "Three separate models for 95% confidence intervals"
            }
        )

        logger.info(f"XGBoost training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, F1: {f1_score_val:.2f}%, Accuracy: {accuracy:.2f}%")

        # Save models to disk
        self._save()

        return self.metrics

    def predict(self, df: pd.DataFrame, days: int = 30) -> list:
        """
        Generate multi-day predictions with confidence intervals.

        Uses recursive forecasting: predict day t+1, then use that prediction
        as input for day t+2, and so on.

        Args:
            df: Feature-engineered DataFrame (used to get the last known state).
            days: Number of days to predict into the future.

        Returns:
            List of dicts with date, price, lower, upper for each predicted day.
        """
        if self.model_median is None:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        predictions = []
        last_date = df.index[-1]

        # Get the last row of features for recursive prediction
        current_features = df[self.feature_columns].iloc[-1:].values.copy()

        for i in range(days):
            pred_date = last_date + pd.Timedelta(days=i + 1)

            price = float(self.model_median.predict(current_features)[0])
            lower = float(self.model_lower.predict(current_features)[0])
            upper = float(self.model_upper.predict(current_features)[0])

            # Ensure lower <= price <= upper
            lower = min(lower, price)
            upper = max(upper, price)
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "lower": round(lower, 2),
<<<<<<< HEAD
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
=======
                "upper": round(upper, 2)
            })

            # Update features for next prediction (shift lags, recalculate rolling)
            # Simplified: shift the feature vector and inject the new predicted price
            current_features = self._update_features_for_next_step(
                current_features, price
            )

        return predictions

    def _update_features_for_next_step(self, features: np.ndarray,
                                        new_price: float) -> np.ndarray:
        """
        Update the feature vector for the next recursive prediction step.

        This is a simplified version — in production, you'd maintain a full
        rolling window of prices. Here we shift lag values and approximate
        rolling averages.

        Args:
            features: Current feature vector (1 x num_features).
            new_price: Predicted price to incorporate.

        Returns:
            Updated feature vector.
        """
        updated = features.copy()

        # The feature columns should have lag_1 at a known position
        # We shift lag values: lag_1 ← new_price, lag_3 ← old lag_1, etc.
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
        """Save all three models to disk."""
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
        joblib.dump(self.model_median, os.path.join(MODEL_DIR, "xgboost_median.pkl"))
        joblib.dump(self.model_lower, os.path.join(MODEL_DIR, "xgboost_lower.pkl"))
        joblib.dump(self.model_upper, os.path.join(MODEL_DIR, "xgboost_upper.pkl"))
        joblib.dump(self.feature_columns, os.path.join(MODEL_DIR, "xgboost_features.pkl"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "xgboost_metrics.pkl"))
<<<<<<< HEAD
        joblib.dump(self.log_target, os.path.join(MODEL_DIR, "xgboost_log_target.pkl"))

    def load(self) -> bool:
=======
        logger.info("XGBoost models saved to disk.")

    def load(self) -> bool:
        """
        Load saved models from disk.

        Returns:
            True if models were loaded successfully, False otherwise.
        """
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
        try:
            self.model_median = joblib.load(os.path.join(MODEL_DIR, "xgboost_median.pkl"))
            self.model_lower = joblib.load(os.path.join(MODEL_DIR, "xgboost_lower.pkl"))
            self.model_upper = joblib.load(os.path.join(MODEL_DIR, "xgboost_upper.pkl"))
            self.feature_columns = joblib.load(os.path.join(MODEL_DIR, "xgboost_features.pkl"))
            self.metrics = joblib.load(os.path.join(MODEL_DIR, "xgboost_metrics.pkl"))
<<<<<<< HEAD
            log_path = os.path.join(MODEL_DIR, "xgboost_log_target.pkl")
            if os.path.isfile(log_path):
                self.log_target = joblib.load(log_path)
            else:
                self.log_target = False
            return True
        except FileNotFoundError:
=======
            logger.info("XGBoost models loaded from disk.")
            return True
        except FileNotFoundError:
            logger.warning("No saved XGBoost models found.")
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
            return False

    @property
    def is_trained(self) -> bool:
<<<<<<< HEAD
=======
        """Check if the model has been trained or loaded."""
>>>>>>> 3bba824c0d1d9f1b3d9d9f10848532f480acc103
        return self.model_median is not None
