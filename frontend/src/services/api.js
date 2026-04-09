/**
 * API service — Axios client for REST endpoints.
 * Handles communication with ML service directly (go-backend removed).
 */

import axios from 'axios';

// Create axios instance for ML service
const mlService = axios.create({
  timeout: 120000, // 120s timeout (ML predictions can take time to train)
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Fetch historical OHLC candlestick data from CoinGecko.
 * NOTE: CoinGecko free tier limits OHLC data to approximately 24 days (daily candles).
 * Requesting more days will still return only the most recent ~24 days of data.
 * @param {number} days - Number of historical days (capped at 90, but CoinGecko free tier returns max ~24)
 * @param {string} currency - Currency (usd, eur, gbp, etc.)
 * @returns {Promise<{data: Array, source: string}>}
 */
export const fetchHistorical = async (days = 30, currency = 'usd') => {
  try {
    const response = await axios.get(
      `https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=${currency}&days=${Math.min(days, 90)}`,
      { timeout: 10000 }
    );
    // CoinGecko returns [[timestamp, o, h, l, c], ...]
    const data = response.data.map(([timestamp, open, high, low, close]) => ({
      timestamp: new Date(timestamp),
      open,
      high,
      low,
      close,
    }));
    return { data, source: 'coingecko', cached: false };
  } catch (error) {
    console.error('Error fetching from CoinGecko:', error);
    throw new Error(`Failed to fetch historical data: ${error.message}`);
  }
};

/**
 * Fetch price predictions from the ML service via nginx proxy.
 * @param {string} model - 'xgboost' or 'lstm_xgboost'
 * @param {number} days - Prediction horizon (default 30)
 * @param {string} currency - Currency (currently USD only in ML service)
 * @returns {Promise<{model: string, predictions: Array, rmse: number, metrics...}>}
 */
export const fetchPredictions = async (model = 'xgboost', days = 30, currency = 'usd') => {
  try {
    const response = await axios.get(
      `/api/ml/predict?model=${model}&days=${days}`,
      { timeout: 120000 }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw new Error(`Failed to fetch predictions: ${error.message}`);
  }
};

/**
 * Trigger model retraining via ML service (through nginx proxy).
 * @param {string} model - 'xgboost' or 'lstm_xgboost'
 * @returns {Promise<{model: string, rmse: number, metrics...}>}
 */
export const retrainModel = async (model = 'xgboost') => {
  try {
    const response = await axios.post(
      `/api/ml/retrain?model=${model}`,
      {},
      { timeout: 600000 } // 10 minute timeout for retraining
    );
    return response.data;
  } catch (error) {
    console.error('Error retraining model:', error);
    throw new Error(`Retrain failed: ${error.message}`);
  }
};

/**
 * Check ML service health status via nginx proxy.
 * @returns {Promise<Object>}
 */
export const checkHealth = async () => {
  try {
    const response = await axios.get('/api/ml/health', { timeout: 10000 });
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

