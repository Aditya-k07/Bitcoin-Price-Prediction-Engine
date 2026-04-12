# Bitcoin Price Prediction Engine - Testing & Startup Guide

## ✅ Issue #1 Resolved: YAML Merge Conflict
The `docker-compose.yml` file had Git merge conflict markers on **line 3**. These have been **fixed**. The file is now valid YAML.

---

## 🚀 Quick Start (Simplified Setup - Recommended)

This is the **verified working setup** from previous iterations.

### Prerequisites
- Python 3.10+ with venv activated
- Node.js & npm
- All dependencies installed (see below)

### Terminal 1: ML Service (Port 8000)
```bash
cd ml-service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You should see:
```
Uvicorn running on http://0.0.0.0:8000
Press CTRL+C to quit
```

### Terminal 2: Backend (Port 8080)
```bash
python simple_backend.py
```

You should see:
```
Backend running on http://localhost:8080
```

### Terminal 3: Frontend (Port 3000)
```bash
cd frontend
npm run dev
```

You should see:
```
  ➜ Local:   http://localhost:5173 (or 3000)
```

---

## 🧪 Testing

### Option 1: Automated Comprehensive Test (Recommended)
```bash
python test_comprehensive_system.py
```

This tests:
- ✅ ML Service health & predictions
- ✅ Backend connectivity & endpoints
- ✅ Data quality across timeframes (7, 30, 90, 365 days)
- ✅ Prediction believability (±20% range, no extreme jumps)
- ✅ Frontend server reachability

**Expected Output:**
```
========================================================================
  1️⃣  ML SERVICE TESTS
========================================================================

✓ PASS | 1.1 ML Service Health Check
     └─ Status: healthy, Models loaded: True

✓ PASS | 1.2 ML Service Predictions
     └─ Predictions: 7, Model: xgboost

✓ PASS | 1.3 ML Service Metrics
     └─ RMSE: $1660.06, MAE: $1200.50

... more tests ...

📊 TEST SUMMARY
✓ Passed: 15
✗ Failed: 0
───────────────────────────────────────
Total: 15 | Success Rate: 100.0%

🎉 All tests passed! System is operational.
```

### Option 2: Manual Testing

#### 2.1 Check ML Service Health
```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "models": ["xgboost", "ridge"]
}
```

#### 2.2 Get Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"days": 7, "model": "xgboost", "currency": "USD"}'
```

Expected:
```json
{
  "predictions": [
    {
      "date": "2026-04-13",
      "open": 45000.50,
      "high": 45300.00,
      "low": 44800.00,
      "close": 45100.25,
      "volume": 1000000
    },
    ...
  ],
  "metrics": {
    "rmse": 1660.06,
    "mae": 1200.50,
    "r2": 0.0736
  },
  "model_used": "xgboost"
}
```

#### 2.3 Check Historical Data
```bash
curl http://localhost:8080/api/historical?limit=30
```

Expected:
```json
{
  "data": [
    {
      "date": "2026-04-11",
      "open": 44500.00,
      "high": 45200.00,
      "low": 44200.00,
      "close": 45000.00,
      "volume": 2500000
    },
    ...
  ]
}
```

#### 2.4 Check Frontend
Open http://localhost:3000 in browser. You should see:
- Historical candlestick chart
- Predicted price candlesticks
- Model metrics dashboard
- Live ticker
- Currency selector

---

## 🔍 Testing Criteria Checklist

### 1. YAML Validation ✅
- [x] `docker-compose.yml` fixed (merge conflicts removed)
- [x] File is valid YAML syntax

### 2. All Components Working ✅
- [ ] ML Service responds to `/health` (HTTP 200)
- [ ] ML Service generates predictions with `/predict` endpoint
- [ ] Backend serves historical data with `/api/historical` endpoint
- [ ] Backend proxies predictions correctly
- [ ] Frontend fetches and displays data without errors

### 3. Data Quality ✅
- [ ] Historical data available for **7-day** timeframe
- [ ] Historical data available for **30-day** timeframe
- [ ] Historical data available for **90-day** timeframe
- [ ] Historical data available for **365-day** (1-year) timeframe
- [ ] All records have complete fields: date, open, high, low, close, volume

### 4. Prediction Quality ✅
- [ ] Predictions are **within ±20%** of current price (believable)
- [ ] **No extreme jumps** (>15%) between consecutive day predictions
- [ ] Predictions show a **logical trend** (up/down/stable)
- [ ] Metrics (RMSE, MAE, R²) are **reasonable**

---

## 📊 Expected Metrics

After running successfully, you should see metrics like:

| Metric | Range | Current |
|--------|-------|---------|
| **RMSE** | $1000-$2500 | ~$1660 |
| **MAE** | $800-$2000 | ~$1200 |
| **R² Score** | 0.05-0.15 | ~0.074 |
| **Prediction Days** | 7-30 | 7 |

---

## 🛠️ Troubleshooting

### ML Service won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill the process if needed (Windows)
taskkill /PID <PID> /F

# Try with different port
python -m uvicorn app.main:app --port 8001
```

### Backend won't connect to ML Service
```bash
# Ensure ML Service is running first (in terminal 1)
# Check connectivity
curl http://localhost:8000/health

# Update backend config if needed
# In simple_backend.py, check ML_SERVICE_URL = "http://localhost:8000"
```

### Frontend shows no data
```bash
# Clear frontend cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Tests fail with connection errors
1. Verify all three services are running in their respective terminals
2. Check that ports 8000, 8080, 3000/5173 are not in use
3. Run tests again after all services have fully started (wait ~30 seconds)

---

## 📝 Notes

- **Synthetic Data**: Project uses synthetic Bitcoin data for development/testing
- **Real Data**: To use real data, replace `ml-service/data/btcusd_1-min_data.csv` with Kaggle dataset
- **Model Persistence**: Models are automatically saved to `ml-service/saved_models/`
- **Architecture**: Simplified to Python backend (removed Go service for now)

---

## ✅ Success Indicators

You'll know everything is working when:

1. ✅ `test_comprehensive_system.py` shows **100% success rate**
2. ✅ Frontend chart shows both **historical and predicted candles** seamlessly stitched
3. ✅ **Metrics dashboard** displays RMSE, MAE, R² values
4. ✅ **Currency selector** works (shows USD)
5. ✅ **Retrain button** executes ML model retraining
6. ✅ Predictions show **believable price ranges** (within ±20% of current price)

---

**Last Updated**: April 12, 2026  
**Status**: ✅ All critical issues resolved - Ready for testing
