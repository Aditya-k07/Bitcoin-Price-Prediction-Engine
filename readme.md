# 🪙 CoinSight — Bitcoin Price Prediction Engine

A full-stack analytical tool that bridges raw financial data and actionable machine learning insights. CoinSight visualizes where Bitcoin has been and uses historical patterns to forecast where it might go.

![Architecture: Go + Python + React](https://img.shields.io/badge/Architecture-Microservices-blue)
![ML: XGBoost + Prophet](https://img.shields.io/badge/ML-XGBoost%20%2B%20Prophet-green)
![Frontend: React + ApexCharts](https://img.shields.io/badge/Frontend-React%20%2B%20ApexCharts-purple)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
6. [API Contracts](#api-contracts)
7. [Getting Started](#getting-started)
8. [Bonus Features](#bonus-features)
9. [Evaluation Criteria](#evaluation-criteria)

---

## Overview

CoinSight uses a **microservices-lite** architecture with three core services:

| Service | Language | Role |
|---------|----------|------|
| **ML Service** | Python (FastAPI) | The Brain — trains models, generates predictions with confidence intervals |
| **Go Backend** | Go (Gin) | The Orchestrator — serves historical data, proxies ML requests, manages caching |
| **Frontend** | React (Vite) | The Interface — candlestick charts, prediction overlays, live ticker |

**Dataset**: [Bitcoin Historical Data (Kaggle)](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data) — minute-level BTC price data resampled to daily OHLCV.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        React Frontend                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Candlestick  │  │  Prediction  │  │  Confidence Bands    │    │
│  │   Chart      │  │   Overlay    │  │  (Range Area Chart)  │    │
│  └─────────────┘  └──────────────┘  └──────────────────────┘    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Live Ticker  │  │   Retrain    │  │  Model Selector      │    │
│  │ (WebSocket)  │  │   Button     │  │  (XGBoost/Prophet)   │    │
│  └─────────────┘  └──────────────┘  └──────────────────────┘    │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP + WebSocket
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Go Backend (Gin)                           │
│                                                                  │
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
│              │   │  ┌────────────┐  ┌────────────────────┐      │
└──────────────┘   │  │  XGBoost   │  │     Prophet        │      │
                   │  │  (Aggr.)   │  │     (Conserv.)     │      │
                   │  └────────────┘  └────────────────────┘      │
                   │                                              │
                   │  ┌────────────────────────────────────┐      │
                   │  │  Feature Engineering Pipeline      │      │
                   │  │  Lag Values · Rolling Avg · RSI    │      │
                   │  └────────────────────────────────────┘      │
                   │                                              │
                   │  Dataset: Kaggle BTC Historical (CSV)        │
                   └──────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **ML Service** | Python 3.11, FastAPI, XGBoost, Prophet, pandas, scikit-learn | Industry standard for data science; FastAPI for async endpoints |
| **Backend** | Go 1.22, Gin, go-redis, gorilla/websocket | Superior concurrency, fast API routing, built-in timeout handling |
| **Frontend** | React 18, Vite, ApexCharts, Axios | High-perf data viz, modern tooling, smooth candlestick rendering |
| **Cache** | Redis 7 | In-memory store for 5-min prediction caching |
| **Orchestration** | Docker Compose | One-command spin-up of all services |

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
│   │       ├── xgboost_model.py  # XGBoost train/predict/evaluate
│   │       └── prophet_model.py  # Prophet train/predict
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
│   │   │   └── handlers.go     # HTTP handlers
│   │   ├── coingecko/
│   │   │   └── client.go       # CoinGecko OHLC client
│   │   ├── mlclient/
│   │   │   └── client.go       # ML service client + circuit breaker
│   │   ├── cache/
│   │   │   └── redis.go        # Redis get/set (5-min TTL)
│   │   ├── websocket/
│   │   │   └── ticker.go       # WebSocket hub for live prices
│   │   └── models/
│   │       └── types.go        # Shared data types
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
        │   ├── CandlestickChart.jsx   # Historical OHLC candles
        │   ├── PredictionOverlay.jsx   # Predicted price line
        │   ├── ConfidenceBands.jsx     # 95% CI range-area
        │   ├── LiveTicker.jsx          # WebSocket price ticker
        │   ├── RetrainButton.jsx       # Trigger retrain flow
        │   └── ModelSelector.jsx       # XGBoost vs Prophet toggle
        └── services/
            ├── api.js                  # Axios REST client
            └── websocket.js            # WS connection manager
```

---

## Phase-by-Phase Breakdown

We build this project incrementally. Each phase is self-contained and testable.

### Phase 1: ML Service (Python) — *The Brain*

**Goal**: A standalone Python service that ingests historical BTC data, engineers features, trains models, and serves predictions via REST.

#### Requirements
| # | Requirement | Details |
|---|-------------|---------|
| 1.1 | **Data Ingestion** | Load Kaggle CSV (~7M rows, minute-level). Resample to daily OHLCV (Open, High, Low, Close, Volume). Handle missing values. |
| 1.2 | **Feature Engineering** | **Lag Values**: Close price at t-1, t-3, t-7, t-14, t-30 days. **Rolling Averages**: 7-day, 14-day, 30-day SMA. **Extras**: RSI (14-day), daily return %, rolling volatility (14-day std dev). |
| 1.3 | **XGBoost Model** | Train/test split (80/20, time-based, no shuffle). Target = next-day close. **Confidence Intervals** via quantile regression (train 3 models: α=0.025, 0.5, 0.975 for 95% CI). |
| 1.4 | **Prophet Model** | Facebook Prophet with daily seasonality. Built-in uncertainty intervals (95%). Renamed columns to `ds` (date) and `y` (close). |
| 1.5 | **Evaluation** | Report RMSE on test set for both models. Log during training. |
| 1.6 | **API Endpoints** | `GET /predict`, `POST /retrain`, `GET /health` (see API Contracts below). |
| 1.7 | **Dockerization** | `python:3.11-slim`, install deps, expose port 8000. |

#### Key Concepts
- **Lag Values**: "What was the price N days ago?" — lets the model learn from recent history.
- **Rolling Averages**: Smooth out daily volatility to reveal trends.
- **Confidence Intervals**: "We predict $65,000 ± $3,000 (95% CI)" — communicates uncertainty honestly.

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
| 3.1 | **Candlestick Chart** | ApexCharts `candlestick` type for historical OHLC. Responsive, zoomable, dark theme. |
| 3.2 | **Prediction Overlay** | `line` series appended after the last candle. Different color (e.g., cyan dashed) to distinguish from historical. |
| 3.3 | **Confidence Bands** | `rangeArea` series showing upper/lower bounds as a shaded region around the prediction line. |
| 3.4 | **Combined View** | Historical candles + prediction line + confidence bands on a single synchronized chart. Seamless time axis transition. |
| 3.5 | **Model Selector** | Toggle/dropdown: "Conservative (Prophet)" vs "Aggressive (XGBoost)". Fetches new predictions on change. |
| 3.6 | **Retrain Button** | Triggers `POST /api/retrain` → shows loading spinner → on success, refetches predictions and updates chart. |
| 3.7 | **Live Ticker** | WebSocket connection to `ws://backend/ws/ticker`. Displays current BTC/USD price with green/red flash on change. |
| 3.8 | **UI/UX Polish** | Dark theme, glassmorphism cards, smooth animations, responsive layout. |

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
Query: ?model=xgboost|prophet&days=30
Response:
{
  "model": "xgboost",
  "predictions": [
    {"date": "2024-01-15", "price": 65000.0, "lower": 62000.0, "upper": 68000.0},
    ...
  ],
  "rmse": 1250.5,
  "trained_at": "2024-01-14T12:00:00Z"
}
```

#### `POST /retrain`
```
Query: ?model=xgboost|prophet
Response:
{
  "model": "xgboost",
  "rmse": 1180.3,
  "trained_at": "2024-01-15T14:30:00Z",
  "message": "Model retrained successfully"
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
