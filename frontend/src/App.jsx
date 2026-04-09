/**
 * App.jsx — Main CoinSight application.
 *
 * Orchestrates:
 * - Historical data fetching from CoinGecko API (free tier limited to ~24 days of OHLC)
 * - Prediction data fetching from ML service (via nginx proxy)
 * - Model switching (XGBoost / LSTM+XGBoost)
 * - Retrain triggering
 * - Live ticker via WebSocket
 * - Combined chart rendering
 */

import { useState, useEffect, useCallback } from 'react';
import PriceChart from './components/PriceChart';
import LiveTicker from './components/LiveTicker';
import ModelSelector from './components/ModelSelector';
import RetrainButton from './components/RetrainButton';
import CurrencySelector from './components/CurrencySelector';
import ModelMetricsTab from './components/ModelMetricsTab';
import { fetchHistorical, fetchPredictions, retrainModel } from './services/api';

export default function App() {
  // State
  const [historicalData, setHistoricalData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [predictionMeta, setPredictionMeta] = useState(null);
  const [activeModel, setActiveModel] = useState('xgboost');
  const [historicalDays, setHistoricalDays] = useState(30); // CoinGecko free tier: ~24 days max OHLC
  const [predictionDays, setPredictionDays] = useState(30);
  const [currency, setCurrency] = useState('usd');
  const [activeTab, setActiveTab] = useState('chart'); // 'chart' or 'metrics'
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
  const loadHistorical = useCallback(async (days, curr) => {
    try {
      setError(null);
      const data = await fetchHistorical(days, curr);
      setHistoricalData(data.data || []);
    } catch (err) {
      console.error('Failed to load historical data:', err);
      setError('Failed to load historical data from CoinGecko. Please check your internet connection.');
    }
  }, []);

  // Fetch predictions
  const loadPredictions = useCallback(async (model, days, curr) => {
    try {
      setIsLoadingPredictions(true);
      setError(null);
      const data = await fetchPredictions(model, days, curr);
      setPredictions(data.predictions || []);
      setPredictionMeta({
        model: data.model,
        rmse: data.rmse,
        mae: data.mae,
        r2_score: data.r2_score,
        mape: data.mape,
        f1_score: data.f1_score,
        accuracy: data.accuracy,
        directional_accuracy: data.directional_accuracy,
        trainedAt: data.trained_at,
        architecture_details: data.architecture_details,
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

  // Initial data load on mount
  useEffect(() => {
    const loadAll = async () => {
      setIsLoadingChart(true);
      try {
        await loadHistorical(historicalDays, currency);
        await loadPredictions(activeModel, predictionDays, currency);
      } catch (err) {
        console.error('Initial data load failed:', err);
      } finally {
        setIsLoadingChart(false);
      }
    };
    loadAll();
  }, [currency, activeModel]); // Re-load when currency or model changes

  // Handle model change
  const handleModelChange = useCallback(
    async (model) => {
      setActiveModel(model);
      await loadPredictions(model, predictionDays, currency);
    },
    [loadPredictions, predictionDays, currency]
  );

  // Handle historical days change
  const handleHistoricalDaysChange = useCallback(
    async (days) => {
      setHistoricalDays(days);
      setIsLoadingChart(true);
      await loadHistorical(days, currency);
      setIsLoadingChart(false);
    },
    [loadHistorical, currency]
  );

  // Handle prediction days change
  const handlePredictionDaysChange = useCallback(
    async (days) => {
      setPredictionDays(days);
      await loadPredictions(activeModel, days, currency);
    },
    [loadPredictions, activeModel, currency]
  );

  // Handle retrain
  const handleRetrain = useCallback(
    async (model) => {
      try {
        setIsRetraining(true);
        setError(null);
        const result = await retrainModel(model);
        const currPrefix = currency === 'usd' ? '$' : '₹';
        showToast(`${model} model retrained! New RMSE: ${currPrefix}${result.rmse.toFixed(2)}`);

        // Refresh predictions with new model
        await loadPredictions(model, predictionDays, currency);
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
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#C19A5B" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" className="header__icon-svg">
            <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"></polygon>
            <line x1="12" y1="22" x2="12" y2="15.5"></line>
            <polyline points="22 8.5 12 15.5 2 8.5"></polyline>
            <polyline points="2 15.5 12 8.5 22 15.5"></polyline>
            <line x1="12" y1="2" x2="12" y2="8.5"></line>
          </svg>
          <div>
            <h1 className="header__title">CoinSight</h1>
            <p className="header__subtitle">Asset Prediction Engine</p>
          </div>
        </div>
        <div className="header__actions">
          <CurrencySelector currency={currency} setCurrency={setCurrency} />
          <LiveTicker currency={currency} />
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

      {/* Tabs Control */}
      <div className="tabs-container">
        <button 
          className={`tab-btn ${activeTab === 'chart' ? 'active' : ''}`}
          onClick={() => setActiveTab('chart')}
        >
          Market Overview
        </button>
        <button 
          className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          Model Metrics
        </button>
      </div>

      {/* Tab Content */}
      <div className="glass-card">
        {activeTab === 'chart' ? (
          <>
            <div className="glass-card__header">
              <span className="glass-card__title">
                Price Chart — Historical + Predictions
              </span>
              {isLoadingPredictions && (
                <span className="status-badge status-badge--connected">
                  Computing predictions...
                </span>
              )}
            </div>
            <PriceChart
              historicalData={historicalData}
              predictions={predictions}
              isLoading={isLoadingChart}
              currency={currency}
            />
          </>
        ) : (
          <ModelMetricsTab 
            predictionMeta={predictionMeta} 
            currency={currency}
            isLoading={isLoadingPredictions} 
          />
        )}
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
