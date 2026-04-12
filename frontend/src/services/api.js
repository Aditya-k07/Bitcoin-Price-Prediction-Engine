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
 * Fetch historical OHLC candlestick data via Go backend.
 * @param {number} days - Number of historical days (1-365)
 * @param {string} currency - Currency (usd, eur, gbp, etc.)
 * @returns {Promise<{data: Array, source: string}>}
 */
export const fetchHistorical = async (days = 30, currency = 'usd') => {
  try {
    const response = await axios.get(
      `/api/historical?days=${days}&currency=${currency}`,
      { timeout: 20000 }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching historical data:', error);
    throw new Error(`Failed to fetch historical data: ${error.message}`);
  }
};

/**
 * Fetch price predictions via Go backend.
 * @param {string} model - 'xgboost' or 'ridge'
 * @param {number} days - Prediction horizon (default 30)
 * @param {string} currency - Currency (currently USD only in ML service)
 * @returns {Promise<{model: string, predictions: Array, rmse: number, metrics...}>}
 */
export const fetchPredictions = async (model = 'xgboost', days = 30, currency = 'usd') => {
  try {
    const response = await axios.get(
      `/api/predict?model=${model}&days=${days}&currency=${currency}`,
      { timeout: 120000 }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw new Error(`Failed to fetch predictions: ${error.message}`);
  }
};

/**
 * Trigger model retraining via Go backend.
 * @param {string} model - 'xgboost' or 'ridge'
 * @returns {Promise<{model: string, rmse: number, metrics...}>}
 */
export const retrainModel = async (model = 'xgboost') => {
  try {
    const response = await axios.post(
      `/api/retrain?model=${model}`,
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
 * Download predictions as Excel file.
 * @param {string} model - 'xgboost' or 'ridge'
 * @param {number} days - Prediction horizon (default 30)
 * @returns {Promise<void>}
 */
export const downloadPredictionsExcel = async (model = 'xgboost', days = 30) => {
  try {
    const response = await axios.get(
      `/api/predict/export?model=${model}&days=${days}`,
      { responseType: 'blob', timeout: 120000 }
    );
    
    // Get filename from Content-Disposition header
    const contentDisposition = response.headers['content-disposition'];
    let filename = `predictions_${model}_${days}days.xlsx`;
    if (contentDisposition) {
      const matches = contentDisposition.match(/filename="([^"]+)"/);
      if (matches) filename = matches[1];
    }
    
    // Create blob and download
    const blob = new Blob([response.data], { 
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
    });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error downloading predictions:', error);
    throw new Error(`Failed to download predictions: ${error.message}`);
  }
};

/**
 * Check backend health status (includes ML service status).
 * @returns {Promise<Object>}
 */
export const checkHealth = async () => {
  try {
    const response = await axios.get('/api/health', { timeout: 10000 });
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

export default mlService;

