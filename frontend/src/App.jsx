/**
 * App.jsx — Main CoinSight application.
 *
 * Orchestrates:
 * - Historical data fetching from CoinGecko API
 * - Prediction data fetching from ML service (via Go backend)
 * - Model switching (XGBoost / Ridge)
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
import { fetchHistorical, fetchPredictions, retrainModel, downloadPredictionsExcel } from './services/api';

export default function App() {
  // State
  const [historicalData, setHistoricalData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [predictionMeta, setPredictionMeta] = useState(null);
  const [activeModel, setActiveModel] = useState('xgboost');
  const [historicalDays, setHistoricalDays] = useState(90); // Default to 90 days for better context

  const [predictionDays, setPredictionDays] = useState(30);
  const [currency, setCurrency] = useState('usd');
  const [activeTab, setActiveTab] = useState('chart'); // 'chart' or 'metrics'
  const [isLoadingChart, setIsLoadingChart] = useState(true);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [isRetraining, setIsRetraining] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
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
      console.log(`[App] Loading historical data: days=${days}, currency=${curr}`);
      const data = await fetchHistorical(days, curr);
      console.log('[App] Historical data response:', data);
      console.log(`[App] Setting historical data: ${data.data?.length || 0} items`);
      setHistoricalData(data.data || []);
    } catch (err) {
      console.error('Failed to load historical data:', err);
      console.error('Error details:', err.response?.data || err.message);
      setError(`Failed to load historical data: ${err.message}`);
    }
  }, []);

  // Fetch predictions
  const loadPredictions = useCallback(async (model, days, curr) => {
    try {
      setIsLoadingPredictions(true);
      setError(null);
      console.log(`[App] Loading predictions: model=${model}, days=${days}`);
      const data = await fetchPredictions(model, days, curr);
      console.log('[App] Predictions response:', data);
      console.log(`[App] Setting predictions: ${data.predictions?.length || 0} items`);
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
      });
      console.log('[App] Prediction metadata set');
    } catch (err) {
      console.error('Failed to load predictions:', err);
      console.error('Error details:', err.response?.data || err.message);
      setError(`Failed to load predictions: ${err.message}`);
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

        // Immediate state update for metrics UI
        setPredictionMeta({
          model: result.model,
          rmse: result.rmse,
          mae: result.mae,
          r2_score: result.r2_score,
          mape: result.mape,
          f1_score: result.f1_score,
          accuracy: result.accuracy,
          directional_accuracy: result.directional_accuracy,
          trainedAt: result.trained_at,
          architecture_details: result.architecture_details,
        });

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

  // Handle download predictions as Excel
  const handleDownloadExcel = useCallback(
    async () => {
      try {
        setIsDownloading(true);
        setError(null);
        await downloadPredictionsExcel(activeModel, predictionDays);
        showToast('Predictions downloaded successfully!', 'success');
      } catch (err) {
        console.error('Download failed:', err);
        showToast('Download failed. Check the service.', 'error');
      } finally {
        setIsDownloading(false);
      }
    },
    [activeModel, predictionDays, showToast]
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

        <button 
          className="download-btn"
          onClick={handleDownloadExcel}
          disabled={isDownloading || predictions.length === 0}
          title={predictions.length === 0 ? 'No predictions available' : 'Download predictions as Excel'}
        >
          {isDownloading ? (
            <>
              <span className="download-btn__spinner"></span>
              Downloading...
            </>
          ) : (
            <>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Export to Excel
            </>
          )}
        </button>

        <div className="days-selector">
          <label className="days-selector__label">History:</label>
          <select
            className="days-selector__select"
            value={historicalDays}
            onChange={(e) => handleHistoricalDaysChange(e.target.value === 'max' ? 'max' : Number(e.target.value))}
          >
            <option value={1}>24 hours</option>
            <option value={7}>7 days</option>
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
            <option value={90}>3 months</option>
            <option value={180}>6 months</option>
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
