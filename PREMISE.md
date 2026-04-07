# PREMISE.md — Assumptions & Technical Decisions

This document captures the technical assumptions and design choices made during development, as required by the evaluation criteria.

---

## 1. Dataset Assumptions

| Decision | Rationale |
|----------|-----------|
| **Resample minute data to daily OHLCV** | The Kaggle dataset has ~7M minute-level rows. Daily resampling reduces noise, aligns with CoinGecko's daily candle granularity, and matches the prediction horizon (days, not minutes). |
| **Forward-fill missing days** | Crypto trades 24/7, so gaps are rare but possible (exchange downtime). Forward-fill preserves the last known price rather than introducing NaN artifacts. |
| **Use Close price as primary target** | Close price is the standard target for daily price prediction. It represents the final agreed-upon price for that trading period. |

## 2. ML Model Decisions

| Decision | Rationale |
|----------|-----------|
| **Time-based train/test split (80/20, no shuffle)** | Random splitting leaks future information into training data, inflating accuracy. Time-based split respects the temporal order of financial data. |
| **XGBoost quantile regression for CI** | XGBoost doesn't natively produce confidence intervals. Training 3 models at α=0.025, 0.5, 0.975 simulates a 95% CI via quantile regression — a well-established technique. |
| **Prophet for conservative model** | Prophet handles trend + seasonality decomposition natively and has built-in uncertainty. It produces smoother, less volatile forecasts — suitable as a "Conservative" option. |
| **Recursive multi-step prediction** | For XGBoost, we predict day t+1, then use that prediction as input for t+2. This is simpler than training separate models for each horizon but introduces compounding error — documented as a known limitation. |
| **Auto-train on first request** | Models train automatically when the first `/predict` request arrives. This avoids requiring a manual training step before the service is useful. |
| **Persist models to disk** | Trained models are saved with `joblib` so they survive container restarts. Retraining is expensive and should be explicit. |

## 3. Feature Engineering Choices

| Feature | Purpose |
|---------|---------|
| **Lag values (1, 3, 7, 14, 30 days)** | Capture short-term momentum and medium-term trends. These horizons are standard in financial ML. |
| **SMA (7, 14, 30 days)** | Smooth out volatility. The crossover of short vs long SMA is a classic trading signal. |
| **RSI (14-day)** | Momentum oscillator. Values > 70 = overbought, < 30 = oversold. Helps detect potential reversals. |
| **Rolling volatility (14-day)** | Measures recent price dispersion. High volatility periods often precede trend changes. |
| **Daily return %** | Normalizes price movement regardless of absolute price level. |
| **Price range, OC diff** | Intraday range and direction. Large ranges suggest high uncertainty. |

## 4. Backend Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Go + Gin** | Excellent concurrency model for handling multiple simultaneous API/WebSocket connections. Compiled binary = minimal Docker image. |
| **Circuit breaker on ML client** | After 3 consecutive ML failures, the backend stops forwarding requests for 30s. This prevents cascading failures and gives the ML service time to recover. |
| **Logic isolation for /api/historical** | Historical data endpoint has zero dependency on the ML service. If Python is down or retraining, users still see chart data. |
| **Redis caching with 5-min TTL** | Predictions don't change frequently. Caching reduces ML service load and improves response times. Cache is invalidated on retrain. |
| **WebSocket hub pattern** | One goroutine manages all connections. CoinGecko is polled only when clients are connected, avoiding unnecessary API calls. |
| **CoinGecko free tier** | Free API with reasonable rate limits (10-30 req/min). No API key required for basic endpoints. |

## 5. Frontend Decisions

| Decision | Rationale |
|----------|-----------|
| **ApexCharts** | One of the few charting libraries that supports candlestick, line, and rangeArea in a single combined chart. Good React integration. |
| **Seamless stitching** | The prediction line starts from the last historical candle's close price, creating a smooth visual transition. |
| **Dark theme** | Financial dashboards are traditionally dark-themed to reduce eye strain during extended monitoring sessions. |
| **Vite proxy for dev** | In development, Vite proxies `/api` and `/ws` to the Go backend, avoiding CORS issues without any additional config. |
| **Nginx for production** | In production (Docker), nginx serves the built React bundle and proxies API/WebSocket to Go backend via Docker networking. |

## 6. Known Limitations

| Limitation | Mitigation |
|------------|------------|
| **Recursive prediction compounding error** | Confidence intervals widen for longer horizons, visually communicating increased uncertainty. |
| **CoinGecko rate limits** | WebSocket ticker polls every 10s (6 req/min), well within free tier limits. |
| **No authentication** | This is a demonstration project. Production deployment would need auth, rate limiting, and input validation hardening. |
| **Prophet training speed** | Prophet can take 30-60s to train on large datasets. The retrain endpoint blocks during this period. A production system would use background task queues. |
| **No live data retraining** | Models retrain on the static Kaggle dataset. Live data from CoinGecko could be appended to the dataset for incremental retraining. |
