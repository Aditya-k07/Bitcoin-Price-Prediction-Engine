/**
 * App.jsx — Main CoinSight application.
 *
 * Orchestrates:
 * - Historical data fetching from Go backend (CoinGecko)
 * - Prediction data fetching from Go backend (ML service proxy)
 * - Model switching (XGBoost / Prophet)
 * - Retrain triggering
 * - Live ticker via WebSocket
 * - Combined chart rendering
 */

import { useState, useEffect, useCallback } from 'react';
import PriceChart from './components/PriceChart';
import LiveTicker from './components/LiveTicker';
import ModelSelector from './components/ModelSelector';
import RetrainButton from './components/RetrainButton';
import { fetchHistorical, fetchPredictions, retrainModel } from './services/api';

export default function App() {
  // State
  const [historicalData, setHistoricalData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [predictionMeta, setPredictionMeta] = useState(null);
  const [activeModel, setActiveModel] = useState('xgboost');
  const [historicalDays, setHistoricalDays] = useState(90);
  const [predictionDays, setPredictionDays] = useState(30);
  const [isLoadingChart, setIsLoadingChart] = useState(true);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [isRetraining, setIsRetraining] = useState(false);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);

  // Show toast notification
  const showToast = useCallback((message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  }, []);

  // Fetch historical data
  const loadHistorical = useCallback(async (days) => {
    try {
      setError(null);
      const data = await fetchHistorical(days);
      setHistoricalData(data.data || []);
    } catch (err) {
      console.error('Failed to load historical data:', err);
      setError('Failed to load historical data. Is the Go backend running?');
    }
  }, []);

  // Fetch predictions
  const loadPredictions = useCallback(async (model, days) => {
    try {
      setIsLoadingPredictions(true);
      setError(null);
      const data = await fetchPredictions(model, days);
      setPredictions(data.predictions || []);
      setPredictionMeta({
        model: data.model,
        rmse: data.rmse,
        trainedAt: data.trained_at,
        cached: data.cached,
      });
    } catch (err) {
      console.error('Failed to load predictions:', err);
      setError('Failed to load predictions. The ML service may be training or unavailable.');
      setPredictions([]);
      setPredictionMeta(null);
    } finally {
      setIsLoadingPredictions(false);
    }
  }, []);

  // Initial data load
  useEffect(() => {
    const loadAll = async () => {
      setIsLoadingChart(true);
      await loadHistorical(historicalDays);
      await loadPredictions(activeModel, predictionDays);
      setIsLoadingChart(false);
    };
    loadAll();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle model change
  const handleModelChange = useCallback(
    async (model) => {
      setActiveModel(model);
      await loadPredictions(model, predictionDays);
    },
    [loadPredictions, predictionDays]
  );

  // Handle historical days change
  const handleHistoricalDaysChange = useCallback(
    async (days) => {
      setHistoricalDays(days);
      setIsLoadingChart(true);
      await loadHistorical(days);
      setIsLoadingChart(false);
    },
    [loadHistorical]
  );

  // Handle prediction days change
  const handlePredictionDaysChange = useCallback(
    async (days) => {
      setPredictionDays(days);
      await loadPredictions(activeModel, days);
    },
    [loadPredictions, activeModel]
  );

  // Handle retrain
  const handleRetrain = useCallback(
    async (model) => {
      try {
        setIsRetraining(true);
        setError(null);
        const result = await retrainModel(model);
        showToast(`${model} model retrained! New RMSE: $${result.rmse.toFixed(2)}`);

        // Refresh predictions with new model
        await loadPredictions(model, predictionDays);
      } catch (err) {
        console.error('Retrain failed:', err);
        showToast('Retrain failed. Check the ML service.', 'error');
      } finally {
        setIsRetraining(false);
      }
    },
    [loadPredictions, predictionDays, showToast]
  );

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header__logo">
          <span className="header__icon">🪙</span>
          <div>
            <h1 className="header__title">CoinSight</h1>
            <p className="header__subtitle">Bitcoin Price Prediction Engine</p>
          </div>
        </div>
        <div className="header__actions">
          <LiveTicker />
        </div>
      </header>

      {/* Error display */}
      {error && <div className="error-message">{error}</div>}

      {/* Controls */}
      <div className="controls">
        <ModelSelector
          activeModel={activeModel}
          onModelChange={handleModelChange}
          disabled={isLoadingPredictions || isRetraining}
        />

        <RetrainButton
          model={activeModel}
          onRetrain={handleRetrain}
          isLoading={isRetraining}
        />

        <div className="days-selector">
          <label className="days-selector__label">History:</label>
          <select
            className="days-selector__select"
            value={historicalDays}
            onChange={(e) => handleHistoricalDaysChange(Number(e.target.value))}
          >
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={180}>180 days</option>
            <option value={365}>1 year</option>
          </select>
        </div>

        <div className="days-selector">
          <label className="days-selector__label">Predict:</label>
          <select
            className="days-selector__select"
            value={predictionDays}
            onChange={(e) => handlePredictionDaysChange(Number(e.target.value))}
          >
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
            <option value={60}>60 days</option>
            <option value={90}>90 days</option>
          </select>
        </div>
      </div>

      {/* Metrics */}
      {predictionMeta && (
        <div className="metrics">
          <div className="metric-card">
            <span className="metric-card__label">Active Model</span>
            <span className="metric-card__value">
              {predictionMeta.model === 'xgboost' ? '⚡ XGBoost' : '🔮 Prophet'}
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-card__label">RMSE (Test Set)</span>
            <span className="metric-card__value metric-card__value--accent">
              ${predictionMeta.rmse.toFixed(2)}
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-card__label">Last Trained</span>
            <span className="metric-card__value">
              {new Date(predictionMeta.trainedAt).toLocaleString()}
            </span>
          </div>
          <div className="metric-card">
            <span className="metric-card__label">Cache Status</span>
            <span className="metric-card__value">
              {predictionMeta.cached ? '📦 Cached' : '🔄 Fresh'}
            </span>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="glass-card">
        <div className="glass-card__header">
          <span className="glass-card__title">
            Price Chart — Historical + Predictions
          </span>
          {isLoadingPredictions && (
            <span className="status-badge status-badge--connected">
              Loading predictions...
            </span>
          )}
        </div>
        <PriceChart
          historicalData={historicalData}
          predictions={predictions}
          isLoading={isLoadingChart}
        />
      </div>

      {/* Footer */}
      <footer className="footer">
        CoinSight — Built with Go, Python (XGBoost + Prophet), React & ApexCharts
      </footer>

      {/* Toast Notification */}
      {toast && (
        <div className={`toast toast--${toast.type}`}>{toast.message}</div>
      )}
    </div>
  );
}
