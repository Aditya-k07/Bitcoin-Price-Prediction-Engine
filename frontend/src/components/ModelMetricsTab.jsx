import React from 'react';

export default function ModelMetricsTab({ predictionMeta, currency, isLoading }) {
  if (isLoading) {
    return <div className="metrics-tab-loading">Loading metrics details...</div>;
  }

  if (!predictionMeta) {
    return <div className="metrics-tab-empty">No metrics available. Please wait or trigger a retrain.</div>;
  }

  const { 
    architecture_details, 
    rmse, 
    mae, 
    r2_score, 
    mape, 
    f1_score, 
    accuracy, 
    directional_accuracy,
    trainedAt, 
    model 
  } = predictionMeta;
  const currencySymbol = '$';

  // Format metric values
  const formatMetric = (value, isPercentage = false) => {
    if (value === null || value === undefined) return 'N/A';
    if (isPercentage) return `${parseFloat(value).toFixed(2)}%`;
    return `${parseFloat(value).toFixed(4)}`;
  };

  return (
    <div className="model-metrics-container">
      <div className="metrics-header">
        <h2 className="metrics-title">
          {model === 'xgboost' ? '⚡ XGBoost Regressor' : '🧠 Hybrid LSTM + XGBoost'}
        </h2>
        <span className="metrics-timestamp">Last trained: {new Date(trainedAt).toLocaleString()}</span>
      </div>

      <div className="metrics-grid">
        {/* Primary Metrics - Top Row */}
        <div className="metric-box box-primary">
          <span className="metric-box-label">RMSE (Root Mean Squared Error)</span>
          <span className="metric-box-value">
            {currencySymbol}{rmse?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="metric-box">
          <span className="metric-box-label">MAE (Mean Absolute Error)</span>
          <span className="metric-box-value">
            {currencySymbol}{mae?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="metric-box">
          <span className="metric-box-label">R² Score</span>
          <span className="metric-box-text">{formatMetric(r2_score)}</span>
        </div>

        {/* Secondary Metrics - Second Row */}
        <div className="metric-box">
          <span className="metric-box-label">MAPE (Mean Absolute %)</span>
          <span className="metric-box-text">{formatMetric(mape, true)}</span>
        </div>

        <div className="metric-box">
          <span className="metric-box-label">F1 Score</span>
          <span className="metric-box-text">{formatMetric(f1_score, true)}</span>
        </div>

        <div className="metric-box">
          <span className="metric-box-label">Accuracy</span>
          <span className="metric-box-text">{formatMetric(accuracy, true)}</span>
        </div>

        <div className="metric-box">
          <span className="metric-box-label">Directional Accuracy</span>
          <span className="metric-box-text">{formatMetric(directional_accuracy, true)}</span>
        </div>

        {/* Architecture Details */}
        {architecture_details && (
          <>
            <div className="metric-box full-width">
              <span className="metric-box-label">Base Model</span>
              <span className="metric-box-text">{architecture_details.base_model || 'XGBoost Trees'}</span>
            </div>
            
            {architecture_details.ensemble_model && (
              <div className="metric-box full-width">
                <span className="metric-box-label">Ensemble Model</span>
                <span className="metric-box-text">{architecture_details.ensemble_model}</span>
              </div>
            )}
            
            {architecture_details.lstm_final_loss && (
              <div className="metric-box full-width">
                <span className="metric-box-label">LSTM Final Loss (MSE)</span>
                <span className="metric-box-text">{architecture_details.lstm_final_loss}</span>
              </div>
            )}
            
            <div className="metric-box full-width">
              <span className="metric-box-label">Strategy</span>
              <span className="metric-box-text">{architecture_details.strategy || 'Feature Engineering + Quantile Regression'}</span>
            </div>
          </>
        )}
      </div>

      <div className="metrics-footer">
        <p>The models automatically fetch dynamic arrays of OHLC historical prices from CoinGecko, compute technical indicators (Simple Moving Averages, Volatility, momentum), and apply an optimized loss function to derive accurate predictions with comprehensive performance metrics.</p>
      </div>
    </div>
  );
}
