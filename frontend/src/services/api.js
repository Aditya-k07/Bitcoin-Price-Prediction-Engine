/**
 * API service — Axios client for REST endpoints.
 * Handles communication with the Go backend.
 */

import axios from 'axios';

// Base URL — in dev, Vite proxy handles /api routes
const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60s timeout (ML predictions can take a while)
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Fetch historical OHLC candlestick data from CoinGecko via Go backend.
 * @param {number} days - Number of historical days (1-365)
 * @returns {Promise<{data: Array, source: string, cached: boolean}>}
 */
export const fetchHistorical = async (days = 90) => {
  const response = await api.get(`/historical?days=${days}`);
  return response.data;
};

/**
 * Fetch price predictions from the ML service via Go backend.
 * @param {string} model - 'xgboost' or 'prophet'
 * @param {number} days - Prediction horizon (1-365)
 * @returns {Promise<{model: string, predictions: Array, rmse: number, trained_at: string, cached: boolean}>}
 */
export const fetchPredictions = async (model = 'xgboost', days = 30) => {
  const response = await api.get(`/predict?model=${model}&days=${days}`);
  return response.data;
};

/**
 * Trigger model retraining via Go backend → ML service.
 * @param {string} model - 'xgboost' or 'prophet'
 * @returns {Promise<{model: string, rmse: number, trained_at: string, message: string}>}
 */
export const retrainModel = async (model = 'xgboost') => {
  const response = await api.post(`/retrain?model=${model}`);
  return response.data;
};

/**
 * Check backend health status.
 * @returns {Promise<Object>}
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
