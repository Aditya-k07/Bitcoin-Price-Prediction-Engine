# 🪙 CoinSight — Bitcoin Price Prediction Engine

A full-stack analytical tool that bridges raw financial data and actionable machine learning insights. CoinSight visualizes where Bitcoin has been and uses historical patterns to forecast where it might go.

![Architecture: Go + Python + React](https://img.shields.io/badge/Architecture-Microservices-blue)
![ML: XGBoost + LSTM+XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20LSTM-green)
![Frontend: React + ApexCharts](https://img.shields.io/badge/Frontend-React%20%2B%20ApexCharts-purple)

---
# For live demonstration check the link - https://www.loom.com/share/71a4a1c7f0f347f796a7b89f716208a2

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Tech Stack](#tech-stack)
4. [Key Features](#key-features)
5. [Project Structure](#project-structure)
6. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
7. [API Contracts](#api-contracts)
8. [Getting Started](#getting-started)
9. [Evaluation Criteria](#evaluation-criteria)

---

## Overview

CoinSight uses a **microservices-lite** architecture with three core services:

| Service | Language | Role |
|---------|----------|------|
| **ML Service** | Python (FastAPI) | The Brain — trains models, generates predictions with comprehensive metrics |
| **Go Backend** | Go (Gin) | The Orchestrator — serves historical data, proxies ML requests, manages caching |
| **Frontend** | React (Vite) | The Interface — dual candlestick charts (historical + predicted), metrics dashboard, live ticker |

**Dataset**: [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data) — minute-level BTC price data resampled to daily OHLCV.

**Currency**: USD only for consistency and clarity.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        React Frontend                            │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Dual Candlestick View│  │  Model Metrics Dashboard        │   │
│  │ • Historical Candles │  │  • RMSE (primary metric)        │   │
│  │ • Predicted Candles  │  │  • MAE, R² Score, MAPE         │   │
│  │ • Seamless Stitching │  │  • F1 Score, Accuracy          │   │
│  │                      │  │  • Directional Accuracy        │   │
│  └──────────────────────┘  └─────────────────────────────────┘   │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐   │
│  │  Live Ticker (USD)   │  │  Model & Retrain Controls       │   │
│  │  WebSocket Updates   │  │  • XGBoost / LSTM+XGBoost       │   │
│  │  Real-time Prices    │  │  • Retrain Button               │   │
│  └──────────────────────┘  └─────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP + WebSocket
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Go Backend (Gin)                           │
│                     USD-only API Endpoints                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │  CoinGecko   │  │  ML Service  │  │  WebSocket Hub     │     │
│  │  API Client  │  │  HTTP Client │  │  (Live Ticker)     │     │
│  └──────┬───────┘  └──────┬───────┘  └────────────────────┘     │
│         │                 │                                      │
│         │          ┌──────┴───────┐                              │
│         │          │  Redis Cache │                              │
│         │          │  (5-min TTL) │                              │
│         │          └──────────────┘                              │
└─────────┼─────────────────┼──────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌──────────────┐   ┌──────────────────────────────────────────────┐
│  CoinGecko   │   │           ML Service (FastAPI)               │
│  Public API  │   │                                              │
│              │   │  ┌─────────────────┐  ┌─────────────────┐    │
└──────────────┘   │  │  XGBoost        │  │  LSTM+XGBoost   │    │
                   │  │  Quantile Reg.  │  │  Hybrid Network │    │
                   │  └─────────────────┘  └─────────────────┘    │
                   │                                              │
                   │  ┌────────────────────────────────────────┐  │
                   │  │  Evaluation Metrics Pipeline           │  │
                   │  │  • RMSE, MAE, R² Score                │  │
                   │  │  • MAPE, F1 Score, Accuracy           │  │
                   │  │  • Directional Accuracy               │  │
                   │  └────────────────────────────────────────┘  │
                   │                                              │
                   │  ┌────────────────────────────────────────┐  │
                   │  │  Feature Engineering Pipeline          │  │
                   │  │  • Lag Values (1, 3, 7, 14, 30 days)  │  │
                   │  │  • SMA (7, 14, 30 days)               │  │
                   │  │  • RSI (14-day momentum)              │  │
                   │  │  • Volatility & Daily Returns         │  │
                   │  └────────────────────────────────────────┘  │
                   │                                              │
                   │  Dataset: Kaggle BTC Historical (CSV)        │
                   │  OHLC Output Format for Predictions         │
                   └──────────────────────────────────────────────┘
```

---

## Key Features

### Visualization
- **Dual Candlestick Charts**: Historical and predicted BTC prices side-by-side
- **Seamless Stitching**: Predictions seamlessly connect from last historical candle
- **OHLC Format**: Both historical and predicted prices display as proper candlesticks

### Model Comparison
- **XGBoost Regressor**: Aggressive, fast predictions using quantile regression
- **Hybrid LSTM+XGBoost**: Deep learning + ensemble for refined accuracy

### Comprehensive Metrics Dashboard
| Metric | Purpose | Type |
|--------|---------|------|
| **RMSE** | Root Mean Squared Error (primary metric) | Error (USD) |
| **MAE** | Mean Absolute Error | Error (USD) |
| **R² Score** | Coefficient of Determination (0-1) | Fit Quality |
| **MAPE** | Mean Absolute Percentage Error | % Error |
| **F1 Score** | Directional precision on trend predictions | % |
| **Accuracy** | Predictions within 2% threshold | % |
| **Directional Accuracy** | Trend direction match rate | % |

### Live Updates
- **WebSocket Ticker**: Real-time USD/BTC prices from CoinGecko
- **Auto-Retrain**: Manual retrain button to update model with latest data
- **Redis Caching**: 5-minute prediction caching for performance

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **ML Service** | Python 3.11, FastAPI, XGBoost, PyTorch (LSTM), scikit-learn, pandas | Industry standard for data science; FastAPI for async; PyTorch for LSTM hybrid |
| **Backend** | Go 1.22, Gin, go-redis, gorilla/websocket | Superior concurrency, fast API routing, built-in timeout handling |
| **Frontend** | React 18, Vite, ApexCharts, Axios | High-perf data viz, candlestick + line support, modern tooling |
| **Cache** | Redis 7 | In-memory store for 5-min prediction caching |
| **Orchestration** | Docker Compose | One-command spin-up of all services |
| **Metrics** | scikit-learn | RMSE, MAE, R², MAPE, F1, Accuracy, Directional Accuracy |

---

## Project Structure

```
backend_task_golang/
├── docker-compose.yml          # Full-stack orchestration
├── README.md                   # This file
├── PREMISE.md                  # Assumptions & technical decisions
│
├── ml-service/                 # ── Python ML Service ──
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py             # FastAPI entry (endpoints)
│   │   ├── data_loader.py      # Kaggle CSV → daily OHLCV
│   │   ├── features.py         # Lag values, rolling avg, RSI
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── models/
│   │       ├── xgboost_model.py  # XGBoost + metrics computation
│   │       └── lstm_xgboost_model.py  # Hybrid LSTM+XGBoost
│   └── data/
│       └── btc_historical.csv  # Kaggle dataset (not committed)
│
├── go-backend/                 # ── Go Backend ──
│   ├── Dockerfile
│   ├── go.mod
│   ├── cmd/server/main.go      # Entry point, graceful shutdown
│   ├── internal/
│   │   ├── api/
│   │   │   ├── router.go       # Route definitions
│   │   │   └── handlers.go     # HTTP handlers (USD-only)
│   │   ├── coingecko/
│   │   │   └── client.go       # CoinGecko OHLC client
│   │   ├── mlclient/
│   │   │   └── client.go       # ML service client + circuit breaker
│   │   ├── cache/
│   │   │   └── redis.go        # Redis get/set (5-min TTL)
│   │   ├── websocket/
│   │   │   └── ticker.go       # WebSocket hub for live prices (USD)
│   │   └── models/
│   │       └── types.go        # Shared data types (metrics support)
│   └── config/
│       └── config.go           # Env-based configuration
│
└── frontend/                   # ── React Frontend ──
    ├── Dockerfile
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        ├── components/
        │   ├── PriceChart.jsx              # Dual candlestick view
        │   ├── LiveTicker.jsx              # WebSocket ticker (USD)
        │   ├── ModelMetricsTab.jsx         # Metrics dashboard
        │   ├── ModelSelector.jsx           # Model selection
        │   ├── RetrainButton.jsx           # Retrain trigger
        │   └── CurrencySelector.jsx        # USD-only selector
        └── services/
            ├── api.js                      # Axios REST client
            └── websocket.js                # WS connection manager
```

---

## Phase-by-Phase Breakdown

We build this project incrementally. Each phase is self-contained and testable.

### Phase 1: ML Service (Python) — *The Brain*

**Goal**: A standalone Python service that ingests historical BTC data, engineers features, trains models, and serves predictions with comprehensive metrics via REST.

#### Requirements
| # | Requirement | Details |
|---|-------------|---------|
| 1.1 | **Data Ingestion** | Load Kaggle CSV (~7M rows, minute-level). Resample to daily OHLCV (Open, High, Low, Close, Volume). Handle missing values. |
| 1.2 | **Feature Engineering** | **Lag Values**: Close price at t-1, t-3, t-7, t-14, t-30 days. **Rolling Averages**: 7-day, 14-day, 30-day SMA. **Extras**: RSI (14-day), daily return %, rolling volatility (14-day std dev). |
| 1.3 | **XGBoost Model** | Train/test split (80/20, time-based, no shuffle). Target = next-day close. **Confidence Intervals** via quantile regression (train 3 models: α=0.025, 0.5, 0.975 for 95% CI). |
| 1.4 | **LSTM+XGBoost Hybrid** | PyTorch LSTM (2 layers, 64 units) stacked with XGBoost Regressor. LSTM extracts temporal patterns, fed into XGBoost for quantile mapping. |
| 1.5 | **Evaluation Metrics** | Compute RMSE, MAE, R² Score, MAPE, F1 Score (directional), Accuracy (2% threshold), Directional Accuracy on test set. Report all during training. |
| 1.6 | **OHLC Prediction Format** | Return predictions as OHLC candles: open=price, high=upper_bound, low=lower_bound, close=price. Enables candlestick visualization. |
| 1.7 | **API Endpoints** | `GET /predict` (with metrics), `POST /retrain` (with metrics), `GET /health` (see API Contracts below). |
| 1.8 | **Dockerization** | `python:3.11-slim`, install deps, expose port 8000. |

#### Key Concepts
- **Lag Values**: "What was the price N days ago?" — lets the model learn from recent history.
- **Rolling Averages**: Smooth out daily volatility to reveal trends.
- **Confidence Intervals**: "We predict $65,000 (±$3,000 95% CI)" — communicates uncertainty honestly.
- **Comprehensive Metrics**: RMSE (error magnitude), MAE (robustness), R² (fit quality), MAPE (% error), F1/Accuracy (directional prediction), Directional Accuracy (trend matching).
- **Hybrid Model**: LSTM captures long-term dependencies, XGBoost provides quantile-based uncertainty bounds.

---

### Phase 2: Go Backend — *The Orchestrator*

**Goal**: A Go API server that serves as the single source of truth, fetching live data, proxying ML requests, and managing resilience.

#### Requirements
| # | Requirement | Details |
|---|-------------|---------|
| 2.1 | **CoinGecko Integration** | Fetch OHLC data via `/coins/bitcoin/ohlc?days=N`. Parse into `CandleData` structs. Handle rate limiting (10-30 req/min free tier). |
| 2.2 | **Historical Data Endpoint** | `GET /api/historical?days=90` — serve data independently of ML service. This must work even if ML service is down. |
| 2.3 | **Prediction Proxy** | `GET /api/predict?model=xgboost&days=30` — forward to ML service, cache response in Redis for 5 minutes. Return cached if available. |
| 2.4 | **Retrain Proxy** | `POST /api/retrain?model=xgboost` — forward to ML service. Invalidate Redis cache on success. |
| 2.5 | **Resiliency** | HTTP client timeouts (10s). Graceful error responses for ML service downtime. Malformed JSON handling. Circuit breaker pattern (optional). |
| 2.6 | **Redis Caching** | Predictions cached with `prediction:{model}:{days}` key, 5-minute TTL. |
| 2.7 | **WebSocket Ticker** | `WS /ws/ticker` — poll CoinGecko simple price every 10s, broadcast to all connected clients. Hub pattern for connection management. |
| 2.8 | **Dockerization** | Multi-stage build: `golang:1.22-alpine` → `alpine:3.19`. |

#### Key Concepts
- **Logic Isolation**: Historical data endpoint has zero dependency on ML service.
- **Circuit Breaker**: After N consecutive ML failures, stop forwarding for a cooldown period to avoid cascading failures.

---

### Phase 3: React Frontend — *The Interface*

**Goal**: A polished, data-rich dashboard that stitches historical candles to predicted prices with visual uncertainty.

#### Requirements
| # | Requirement | Details |
|---|-------------|---------|
| 3.1 | **Dual Candlestick Chart** | Historical OHLC candles + Predicted OHLC candles on single chart. Both rendered as candlesticks (not line overlay). Seamless stitching at last historical close. |
| 3.2 | **Candlestick Format** | Predictions returned as OHLC: open=close price, high=upper bound, low=lower bound, close=predicted price. Enables realistic candlestick rendering. |
| 3.3 | **Combined View** | Historical + predicted candlesticks synchronized on shared time axis. Dashed lines for predicted series to distinguish from historical. |
| 3.4 | **Metrics Dashboard** | Tab showing: RMSE (primary), MAE, R² Score, MAPE, F1 Score, Accuracy, Directional Accuracy. Updated with each prediction. |
| 3.5 | **Model Selector** | Toggle: "XGBoost Regressor" vs "Hybrid LSTM+XGBoost". Fetches new predictions and metrics on change. |
| 3.6 | **Retrain Button** | Triggers `POST /api/retrain` → loading spinner → updates metrics and chart on success. |
| 3.7 | **Live Ticker** | WebSocket connection to `ws://backend/ws/ticker`. USD-only, displays current BTC price with real-time updates. |
| 3.8 | **Currency Selector** | USD-only option (INR removed). Consistent pricing in dollars. |
| 3.9 | **UI/UX Polish** | Dark theme, metrics cards, smooth animations, responsive layout. Legends show series names and colors. |

---

### Phase 4: Integration & Deployment

**Goal**: One-command spin-up and end-to-end verification.

#### Requirements
| # | Requirement | Details |
|---|-------------|---------|
| 4.1 | **Docker Compose** | Services: `ml-service` (8000), `go-backend` (8080), `frontend` (3000), `redis` (6379). Shared network. Health checks. |
| 4.2 | **README** | Final documentation with setup instructions, architecture diagram, API reference. |
| 4.3 | **PREMISE.md** | Document all assumptions and technical decisions with rationale. |
| 4.4 | **End-to-End Test** | Full flow: load page → chart renders → click retrain → chart updates → toggle model → ticker updates. |

---

## API Contracts

### ML Service (Port 8000)

#### `GET /predict`
```
Query: ?model=xgboost&days=30
Response:
{
  "model": "xgboost",
  "predictions": [
    {"date": "2024-01-15", "open": 65000.0, "high": 68000.0, "low": 62000.0, "close": 65000.0},
    ...
  ],
  "rmse": 1250.5,
  "mae": 850.2,
  "r2_score": 0.9145,
  "mape": 1.32,
  "f1_score": 87.5,
  "accuracy": 94.2,
  "directional_accuracy": 82.1,
  "trained_at": "2024-01-14T12:00:00Z",
  "architecture_details": {
    "base_model": "XGBoost (500 Trees)",
    "strategy": "Quantile Regression for 95% CI"
  }
}
```

#### `POST /retrain`
```
Query: ?model=xgboost
Response:
{
  "model": "xgboost",
  "rmse": 1180.3,
  "mae": 812.5,
  "r2_score": 0.9287,
  "mape": 1.19,
  "f1_score": 89.3,
  "accuracy": 95.1,
  "directional_accuracy": 84.6,
  "trained_at": "2024-01-15T14:30:00Z",
  "message": "Model retrained successfully",
  "architecture_details": {...}
}
```

#### `GET /health`
```
Response:
{
  "status": "healthy",
  "models_loaded": ["xgboost", "prophet"]
}
```

### Go Backend (Port 8080)

#### `GET /api/historical?days=90`
```
Response:
{
  "data": [
    {"timestamp": 1705276800000, "open": 64500, "high": 65200, "low": 64100, "close": 65000},
    ...
  ],
  "source": "coingecko",
  "cached": false
}
```

#### `GET /api/predict?model=xgboost&days=30`
```
Response: (same as ML /predict, with added "cached" field)
```

#### `POST /api/retrain?model=xgboost`
```
Response: (same as ML /retrain)
```

#### `WS /ws/ticker`
```
Message (every ~10s):
{
  "price": 65123.45,
  "currency": "usd",
  "timestamp": "2024-01-15T14:30:00Z",
  "change_24h": 2.5
}
```

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- (Optional) Go 1.22+, Python 3.11+, Node.js 20+ for local development

### Quick Start
```bash
# Clone the repo
git clone <repo-url>
cd backend_task_golang

# Download the Kaggle dataset and place it at:
# ml-service/data/btc_historical.csv

# Spin up everything
docker-compose up --build

# Access the app
# Frontend:    http://localhost:3000
# Go Backend:  http://localhost:8080
# ML Service:  http://localhost:8000
```

### Local Development (without Docker)
```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 2. Start ML Service
cd ml-service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start Go Backend
cd go-backend
go run cmd/server/main.go

# 4. Start Frontend
cd frontend
npm install
npm run dev
```

---

## Bonus Features

| Feature | Status | Description |
|---------|--------|-------------|
| 🔌 WebSocket Live Ticker | Planned | Real-time BTC/USD price updates without page refresh |
| 📦 Redis Caching | Planned | 5-minute TTL cache for predictions to reduce ML service load |
| 🔄 Model Comparison | Planned | Toggle between Conservative (Prophet) and Aggressive (XGBoost) |
| 🔁 Live Data Retrain | Planned | Retrain models using incoming live data from CoinGecko |

---

## Evaluation Criteria

| Criterion | How We Address It |
|-----------|-------------------|
| **Clean Code** | Modular structure, Go packages, Python modules, typed schemas |
| **ML Rigor** | Time-based train/test split, RMSE evaluation, 95% confidence intervals, proper resampling |
| **Documentation** | This README + PREMISE.md + inline code comments |
| **Resiliency** | Timeouts, circuit breaker, isolated historical endpoint, Redis cache |
| **Autonomy** | All technical decisions documented in PREMISE.md |
