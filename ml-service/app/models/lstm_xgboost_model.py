"""
Hybrid PyTorch LSTM + XGBoost model for Bitcoin price prediction.

- PyTorch CPU trains an improved LSTM with dropout and better architecture.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from app.schemas import ModelMetrics

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")

class ImprovedPriceLSTM(nn.Module):
    """Improved PyTorch LSTM Network with Dropout and better architecture."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ImprovedPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        # Multi-layer dense encoder
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        # Use the last sequence step
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class LSTMXGBoostPredictor:
    """Hybrid LSTM + XGBoost predictor with improved training."""

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

    @property
    def is_trained(self) -> bool:
        return (
            self.model_median is not None and
            self.lstm_model is not None and
            self.scaler is not None
        )

    def train(self, df: pd.DataFrame, feature_columns: list, target_col: str = "target", test_size: float = 0.2) -> ModelMetrics:
        self.feature_columns = feature_columns

        # Extract features and targets
        X = df[feature_columns].values
        y = df[target_col].values

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 1. Scale data using MinMaxScaler for better LSTM training
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Training Improved LSTM+XGBoost: {len(X_train)} train, {len(X_test)} test samples")

        # 2. Train improved PyTorch LSTM Model
        self.lstm_model = ImprovedPriceLSTM(
            input_size=len(feature_columns), 
            hidden_size=128,  # Increased from 64
            num_layers=3,     # Increased from 2
            dropout=0.2
        ).to(self.device)
        
        criterion = nn.SmoothL1Loss()  # More robust to outliers than MSE
        optimizer = optim.AdamW(self.lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

        # Reshape to (batch, seq_len, features)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)

        epochs = 150  # Increased from 80
        batch_size = 32
        lstm_losses = []
        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        self.lstm_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            lstm_losses.append(avg_epoch_loss)
            
            # Evaluate on validation set
            self.lstm_model.eval()
            with torch.no_grad():
                val_outputs = self.lstm_model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 30 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            self.lstm_model.train()

        logger.info(f"LSTM Final Loss: {lstm_losses[-1]:.4f}")

        self.lstm_model.eval()
        
        # Extract LSTM predictions to feed into XGBoost
        with torch.no_grad():
            lstm_train_preds = self.lstm_model(X_train_tensor).cpu().numpy()
            lstm_test_preds = self.lstm_model(X_test_tensor).cpu().numpy()

        # Combine LSTM feature with original features
        X_train_hybrid = np.hstack((X_train, lstm_train_preds))
        X_test_hybrid = np.hstack((X_test, lstm_test_preds))

        # 3. Train XGBoost with improved hyperparameters
        common_params = {
            "n_estimators": 500,          # Increased from 300
            "max_depth": 7,               # Slightly increased
            "learning_rate": 0.03,        # Reduced for better stability
            "subsample": 0.85,            # Improved
            "colsample_bytree": 0.85,     # Improved
            "colsample_bylevel": 0.8,
            "min_child_weight": 1,
            "gamma": 0.1,
            "random_state": 42,
        }

        logger.info("Training XGBoost quantile models...")
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
        mape = float(mean_absolute_percentage_error(y_test, y_pred))
        
        # Directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = float(np.mean(actual_direction == pred_direction)) * 100 if len(actual_direction) > 0 else 0.0
        
        # F1 Score
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
                "base_model": "PyTorch LSTM (3 layers, 128 units, dropout=0.2, SmoothL1Loss)",
                "ensemble_model": "XGBoost Regressor (500 estimators, max_depth=7)",
                "lstm_final_loss": round(lstm_losses[-1], 4) if lstm_losses else 0.0,
                "strategy": "Improved LSTM predictions stacked with original features into XGBoost Quantile Regression"
            }
        )

        logger.info(f"Training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, F1: {f1_score_val:.2f}%, Accuracy: {accuracy:.2f}%")
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
                lstm_output = self.lstm_model(tensor_f)
                lstm_pred = lstm_output.cpu().numpy()

            # 2. Hybrid Combine - flatten lstm_pred if needed
            if lstm_pred.ndim > 1:
                lstm_pred = lstm_pred.flatten()
            hybrid_features = np.hstack((current_features.flatten(), lstm_pred)).reshape(1, -1)

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
            # Use improved LSTM architecture
            self.lstm_model = ImprovedPriceLSTM(input_size=len(self.feature_columns), hidden_size=128, num_layers=3, dropout=0.2).to(self.device)
            self.lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "hybrid_lstm.pth"), map_location=self.device))
            self.lstm_model.eval()
            logger.info("Hybrid LSTM+XGB models loaded from disk.")
            return True
        except Exception as e:
            logger.warning(f"No saved Hybrid model found: {str(e)}")
            return False

    @property
    def is_trained(self) -> bool:
        return self.model_median is not None
