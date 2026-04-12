"""
Simple Python Backend — Serves historical data and proxies ML predictions
"""

import logging
import requests
import json
import io
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import os
import sys

# Add ml-service to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml-service'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8000")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8080"))
# Use the same env var as the ml-service so both containers find the data
BTC_DATA_PATH = os.environ.get("BTC_DATA_PATH", None)

# Try to import ML service components
try:
    from app.data_loader import load_daily_data
    from app.features import engineer_features, get_feature_columns
    from app.models.xgboost_model import XGBoostPredictor
    from app.models.ridge_model import RidgePredictor
    ML_SERVICE_AVAILABLE = True
    logger.info("✓ ML Service components loaded (local mode)")
except Exception as e:
    ML_SERVICE_AVAILABLE = False
    logger.warning(f"⚠ ML Service components not available: {e}")
    logger.info("  Falling back to remote ML service proxy mode")

# Global model instances (local mode)
xgboost_predictor = None
ridge_predictor = None
featured_data = None

if ML_SERVICE_AVAILABLE:
    xgboost_predictor = XGBoostPredictor()
    ridge_predictor = RidgePredictor()
    try:
        # Use BTC_DATA_PATH env var — same one configured in docker-compose
        daily_data = load_daily_data(BTC_DATA_PATH)
        featured_data = engineer_features(daily_data)
        logger.info(f"✓ Data loaded and engineered (local mode) — {len(featured_data)} samples")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Will fall back to remote ML service proxy for predictions")


def _build_flat_prediction_response(model, predictions, metrics):
    """Build a flat response matching the frontend's expected schema."""
    ohlc_predictions = []
    for pred in predictions:
        ohlc_predictions.append({
            'date': pred['date'],
            'open': float(pred['price']),
            'high': float(pred['upper']),
            'low': float(pred['lower']),
            'close': float(pred['price'])
        })

    return {
        'model': model,
        'predictions': ohlc_predictions,
        # Flat metric fields — consumed directly by App.jsx
        'rmse': float(metrics.rmse),
        'mae': float(metrics.mae) if metrics.mae is not None else None,
        'r2_score': float(metrics.r2_score) if metrics.r2_score is not None else None,
        'mape': float(metrics.mape) if metrics.mape is not None else None,
        'f1_score': float(metrics.f1_score) if metrics.f1_score is not None else None,
        'accuracy': float(metrics.accuracy) if metrics.accuracy is not None else None,
        'directional_accuracy': float(metrics.directional_accuracy) if metrics.directional_accuracy is not None else None,
        'trained_at': metrics.trained_at.isoformat() + 'Z' if hasattr(metrics.trained_at, 'isoformat') else str(metrics.trained_at),
        'architecture_details': metrics.architecture_details,
        'cached': False,
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'python-backend',
        'ml_service_available': ML_SERVICE_AVAILABLE,
        'ml_service_url': ML_SERVICE_URL,
        'data_loaded': featured_data is not None,
    }), 200


@app.route('/api/health', methods=['GET'])
def api_health():
    """API health check (same as /health but at /api/health path)"""
    ml_status = 'unavailable'
    models_loaded = []
    try:
        resp = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ml_status = data.get('status', 'unknown')
            models_loaded = data.get('models_loaded', [])
    except Exception:
        pass

    return jsonify({
        'status': 'healthy',
        'services': {
            'ml_service': {
                'status': ml_status,
                'models_loaded': models_loaded,
            },
            'redis': {'status': 'unavailable'},
        }
    }), 200


@app.route('/api/historical', methods=['GET'])
def get_historical():
    """Get historical OHLC data"""
    try:
        limit = request.args.get('days', request.args.get('limit', 30), type=int)
        currency = request.args.get('currency', 'usd', type=str)

        if not ML_SERVICE_AVAILABLE or featured_data is None:
            return jsonify({'error': 'Historical data not available'}), 503

        # Return last 'limit' rows of featured data
        df = featured_data.tail(limit)

        # Convert to OHLC format
        data = []
        for idx, row in df.iterrows():
            ts = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
            data.append({
                'date': ts,
                'open': float(row.get('Open', 0)),
                'high': float(row.get('High', 0)),
                'low': float(row.get('Low', 0)),
                'close': float(row.get('Close', 0)),
                'volume': float(row.get('Volume', 0))
            })

        return jsonify({'data': data, 'currency': currency, 'source': 'local', 'cached': False}), 200
    except Exception as e:
        logger.error(f"Error in /api/historical: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['GET'])
def predict():
    """Get price predictions"""
    try:
        model = request.args.get('model', 'xgboost', type=str)
        days = request.args.get('days', 30, type=int)
        currency = request.args.get('currency', 'usd', type=str)

        if model not in ('xgboost', 'ridge'):
            return jsonify({'error': f'Unknown model: {model}. Use xgboost or ridge'}), 400

        # --- Local mode (data + model available) ---
        if ML_SERVICE_AVAILABLE and featured_data is not None:
            predictor = xgboost_predictor if model == 'xgboost' else ridge_predictor

            if not predictor.is_trained:
                logger.info(f"Training {model} model...")
                feature_cols = get_feature_columns(featured_data)
                predictor.train(featured_data, feature_cols)

            predictions = predictor.predict(featured_data, days=days)
            metrics = predictor.metrics

            return jsonify(_build_flat_prediction_response(model, predictions, metrics)), 200

        # --- Proxy mode (forward to ml-service) ---
        logger.info(f"Proxying to remote ML service: {ML_SERVICE_URL}/predict")
        response = requests.get(
            f"{ML_SERVICE_URL}/predict",
            params={'model': model, 'days': days},
            timeout=120
        )
        return jsonify(response.json()), response.status_code

    except Exception as e:
        logger.error(f"Error in /api/predict: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain model and return flat metrics matching frontend schema"""
    try:
        model = request.args.get('model', 'xgboost', type=str)

        if model not in ('xgboost', 'ridge'):
            return jsonify({'error': f'Unknown model: {model}. Use xgboost or ridge'}), 400

        if ML_SERVICE_AVAILABLE and featured_data is not None:
            predictor = xgboost_predictor if model == 'xgboost' else ridge_predictor

            logger.info(f"Retraining {model}...")
            feature_cols = get_feature_columns(featured_data)
            metrics = predictor.train(featured_data, feature_cols)

            return jsonify({
                'model': model,
                'status': 'retrained',
                'message': f'{model.capitalize()} model retrained successfully',
                # Flat fields — App.jsx reads result.rmse directly
                'rmse': float(metrics.rmse),
                'mae': float(metrics.mae) if metrics.mae is not None else None,
                'r2_score': float(metrics.r2_score) if metrics.r2_score is not None else None,
                'mape': float(metrics.mape) if metrics.mape is not None else None,
                'f1_score': float(metrics.f1_score) if metrics.f1_score is not None else None,
                'accuracy': float(metrics.accuracy) if metrics.accuracy is not None else None,
                'directional_accuracy': float(metrics.directional_accuracy) if metrics.directional_accuracy is not None else None,
                'trained_at': metrics.trained_at.isoformat() + 'Z' if hasattr(metrics.trained_at, 'isoformat') else str(metrics.trained_at),
                'architecture_details': metrics.architecture_details,
            }), 200

        # Proxy mode
        response = requests.post(
            f"{ML_SERVICE_URL}/retrain",
            params={'model': model},
            timeout=600
        )
        return jsonify(response.json()), response.status_code

    except Exception as e:
        logger.error(f"Error in /api/retrain: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/export', methods=['GET'])
def export_predictions():
    """Export predictions to Excel"""
    try:
        model = request.args.get('model', 'xgboost', type=str)
        days = request.args.get('days', 30, type=int)

        if model not in ('xgboost', 'ridge'):
            return jsonify({'error': f'Unknown model: {model}'}), 400

        # Get predictions from local predictor or proxy
        if ML_SERVICE_AVAILABLE and featured_data is not None:
            predictor = xgboost_predictor if model == 'xgboost' else ridge_predictor
            if not predictor.is_trained:
                feature_cols = get_feature_columns(featured_data)
                predictor.train(featured_data, feature_cols)
            raw_preds = predictor.predict(featured_data, days=days)
            predictions = [
                {'date': p['date'], 'open': p['price'], 'high': p['upper'],
                 'low': p['lower'], 'close': p['price']}
                for p in raw_preds
            ]
        else:
            # Proxy through remote ML service export endpoint
            response = requests.get(
                f"{ML_SERVICE_URL}/predict/export",
                params={'model': model, 'days': days},
                timeout=120,
                stream=True
            )
            if response.status_code != 200:
                return jsonify({'error': 'ML service export failed'}), 502
            from flask import Response
            return Response(
                response.content,
                status=200,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': response.headers.get(
                    'Content-Disposition',
                    f'attachment; filename=predictions_{model}_{days}days.xlsx'
                )}
            )

        # Build Excel locally
        df = pd.DataFrame([{
            'Date': p['date'],
            'Open': p['open'],
            'High': p['high'],
            'Low': p['low'],
            'Close': p['close']
        } for p in predictions])

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')

        output.seek(0)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'bitcoin_predictions_{model}_{days}days_{timestamp}.xlsx'
        )

    except Exception as e:
        logger.error(f"Error exporting predictions: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info(f"🚀 Backend starting on port {BACKEND_PORT}")
    logger.info(f"   ML Service URL: {ML_SERVICE_URL}")
    logger.info(f"   BTC Data Path:  {BTC_DATA_PATH or '(using default)'}")
    logger.info(f"   Mode: {'Local' if ML_SERVICE_AVAILABLE and featured_data is not None else 'Proxy'}")
    app.run(host='0.0.0.0', port=BACKEND_PORT, debug=False)
