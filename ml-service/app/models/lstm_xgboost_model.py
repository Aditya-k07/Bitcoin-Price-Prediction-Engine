"""
Hybrid PyTorch LSTM + XGBoost model for Bitcoin price prediction.

- PyTorch CPU trains a lightweight LSTM on sequential features.
- LSTM's internal states and predictions are fed into the XGBoost Regressor.
- XGBoost Regressor applies quantile regression (alpha=0.025, 0.5, 0.975) for bounds.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim

from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")

class PriceLSTM(nn.Module):
    """Lightweight PyTorch LSTM Network."""
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x is (batch, seq_len, features)
        out, _ = self.lstm(x)
        # return the last sequence step
        out = self.fc(out[:, -1, :])
        return out


class LSTMXGBoostPredictor:
    """Hybrid LSTM + XGBoost predictor."""

    def __init__(self):
        self.lstm_model = None
        self.scaler = None
        self.model_median = None
        self.model_lower = None
        self.model_upper = None
        self.metrics: ModelMetrics = None
        self.feature_columns: list = None
        self.device = torch.device("cpu")
        self._ensure_model_dir()

    def _ensure_model_dir(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train(self, df: pd.DataFrame, feature_columns: list, target_col: str = "target", test_size: float = 0.2) -> ModelMetrics:
        self.feature_columns = feature_columns

        # Extract features and targets
        X = df[feature_columns].values
        y = df[target_col].values

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 1. Scale data for Neural Network
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Training Hybrid Model: {len(X_train)} train, {len(X_test)} test samples")

        # 2. Train PyTorch LSTM Model
        self.lstm_model = PriceLSTM(input_size=len(feature_columns), hidden_size=64, num_layers=2).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.01)

        # Reshape to (batch, seq_len, features) with seq_len=1 since features already have lags
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        epochs = 80
        lstm_losses = []
        self.lstm_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            if epoch == epochs - 1:
                lstm_losses.append(loss.item())
                logger.info(f"LSTM Final Epoch Loss: {loss.item():.4f}")

        self.lstm_model.eval()
        
        # Extract LSTM predictions to feed into XGBoost
        with torch.no_grad():
            lstm_train_preds = self.lstm_model(X_train_tensor).numpy()
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
            lstm_test_preds = self.lstm_model(X_test_tensor).numpy()

        # Combine LSTM feature with original features
        X_train_hybrid = np.hstack((X_train, lstm_train_preds))
        X_test_hybrid = np.hstack((X_test, lstm_test_preds))

        # 3. Train XGBoost
        common_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

        self.model_median = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **common_params)
        self.model_median.fit(X_train_hybrid, y_train, verbose=False)

        self.model_lower = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.025, **common_params)
        self.model_lower.fit(X_train_hybrid, y_train, verbose=False)

        self.model_upper = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.975, **common_params)
        self.model_upper.fit(X_train_hybrid, y_train, verbose=False)

        # 4. Evaluate
        y_pred = self.model_median.predict(X_test_hybrid)
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
                "base_model": "PyTorch LSTM (2 layers, 64 units, seq_len=1)",
                "ensemble_model": "XGBoost Regressor (300 estimators, max_depth=5)",
                "lstm_final_loss": round(lstm_losses[0], 2) if lstm_losses else 0.0,
                "strategy": "Stacked LSTM predictions into XGBoost Features for quantile mapping"
            }
        )

        logger.info(f"Hybrid training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, F1: {f1_score_val:.2f}%, Accuracy: {accuracy:.2f}%")
        self._save()
        return self.metrics

    def predict(self, df: pd.DataFrame, days: int = 30) -> list:
        if self.model_median is None:
            raise RuntimeError("Model not trained.")

        predictions = []
        last_date = df.index[-1]
        current_features = df[self.feature_columns].iloc[-1:].values.copy()
        
        self.lstm_model.eval()

        for i in range(days):
            pred_date = last_date + pd.Timedelta(days=i + 1)
            
            # 1. Scale and pass to LSTM
            curr_f_scaled = self.scaler.transform(current_features)
            with torch.no_grad():
                tensor_f = torch.tensor(curr_f_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
                lstm_pred = self.lstm_model(tensor_f).numpy()

            # 2. Hybrid Combine
            hybrid_features = np.hstack((current_features, lstm_pred))

            # 3. XGBoost Quantiles
            price = float(self.model_median.predict(hybrid_features)[0])
            lower = float(self.model_lower.predict(hybrid_features)[0])
            upper = float(self.model_upper.predict(hybrid_features)[0])

            lower = min(lower, price)
            upper = max(upper, price)

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "lower": round(lower, 2),
                "upper": round(upper, 2)
            })

            current_features = self._update_features_for_next_step(current_features, price)

        return predictions

    def _update_features_for_next_step(self, features: np.ndarray, new_price: float) -> np.ndarray:
        updated = features.copy()
        if self.feature_columns:
            for i, col in enumerate(self.feature_columns):
                if col == "lag_1":
                    updated[0, i] = new_price
                elif col == "lag_3" and "lag_1" in self.feature_columns:
                    updated[0, i] = features[0, self.feature_columns.index("lag_1")]
                elif col == "lag_7" and "lag_3" in self.feature_columns:
                    updated[0, i] = features[0, self.feature_columns.index("lag_3")]
        return updated

    def _save(self):
        joblib.dump(self.model_median, os.path.join(MODEL_DIR, "hybrid_median.pkl"))
        joblib.dump(self.model_lower, os.path.join(MODEL_DIR, "hybrid_lower.pkl"))
        joblib.dump(self.model_upper, os.path.join(MODEL_DIR, "hybrid_upper.pkl"))
        joblib.dump(self.feature_columns, os.path.join(MODEL_DIR, "hybrid_features.pkl"))
        joblib.dump(self.metrics, os.path.join(MODEL_DIR, "hybrid_metrics.pkl"))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "hybrid_scaler.pkl"))
        torch.save(self.lstm_model.state_dict(), os.path.join(MODEL_DIR, "hybrid_lstm.pth"))
        logger.info("Hybrid LSTM+XGB models saved to disk.")

    def load(self) -> bool:
        try:
            self.model_median = joblib.load(os.path.join(MODEL_DIR, "hybrid_median.pkl"))
            self.model_lower = joblib.load(os.path.join(MODEL_DIR, "hybrid_lower.pkl"))
            self.model_upper = joblib.load(os.path.join(MODEL_DIR, "hybrid_upper.pkl"))
            self.feature_columns = joblib.load(os.path.join(MODEL_DIR, "hybrid_features.pkl"))
            self.metrics = joblib.load(os.path.join(MODEL_DIR, "hybrid_metrics.pkl"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "hybrid_scaler.pkl"))
            self.lstm_model = PriceLSTM(input_size=len(self.feature_columns), hidden_size=64, num_layers=2).to(self.device)
            self.lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "hybrid_lstm.pth")))
            self.lstm_model.eval()
            logger.info("Hybrid LSTM+XGB models loaded from disk.")
            return True
        except Exception:
            logger.warning("No saved Hybrid model found.")
            return False

    @property
    def is_trained(self) -> bool:
        return self.model_median is not None
