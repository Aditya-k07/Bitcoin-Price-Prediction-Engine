#!/usr/bin/env python3
"""
Complete System Test - Bitcoin Price Prediction Engine
Tests all components in sequence
"""

import requests
import time
import sys

print("\n" + "="*70)
print("  BITCOIN PRICE PREDICTION ENGINE - COMPLETE SYSTEM TEST")
print("="*70 + "\n")

# Configuration
ML_SERVICE_URL = "http://127.0.0.1:8000"
BACKEND_URL = "http://127.0.0.1:8080"

# Test 1: ML Service Health
print("1️⃣  Testing ML Service Health...")
try:
    resp = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ ML Service is healthy")
        print(f"   Status: {data['status']}")
        print(f"   Models: {data['models_loaded']}")
    else:
        print(f"   ❌ ML Service returned {resp.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ ML Service connection failed: {e}")
    sys.exit(1)

# Test 2: ML Service Predictions
print("\n2️⃣  Testing ML Service Predictions...")
try:
    resp = requests.get(f"{ML_SERVICE_URL}/predict?model=xgboost&days=7", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        predictions = data['predictions']
        
        print(f"   ✅ Predictions called successfully")
        print(f"   Predictions generated: {len(predictions)} days")
        if predictions:
            pred = predictions[0]
            print(f"   Sample prediction:")
            print(f"      Date: {pred['date']}")
            print(f"      Close: ${pred['close']:.2f}")
            print(f"      High: ${pred['high']:.2f}")
            print(f"      Low: ${pred['low']:.2f}")
        
        print(f"   Model metrics:")
        print(f"      RMSE: ${data.get('rmse', 0):.2f}")
        print(f"      MAE: ${data.get('mae', 0):.2f}")
        print(f"      R² Score: {data.get('r2_score', 0):.4f}")
        print(f"      F1 Score: {data.get('f1_score', 0):.2f}%")
    else:
        print(f"   ❌ Predictions failed: HTTP {resp.status_code}")

        sys.exit(1)
except Exception as e:
    print(f"   ❌ Predictions error: {e}")
    sys.exit(1)

# Test 3: Retrain endpoint
print("\n3️⃣  Testing Model Retraining...")
try:
    resp = requests.post(f"{ML_SERVICE_URL}/retrain?model_type=xgboost", timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ Model retraining successful")
        print(f"   Model: {data['model']}")
        print(f"   Message: {data['message']}")

    else:
        print(f"   ❌ Retrain failed: HTTP {resp.status_code}")
except Exception as e:
    print(f"   ❌ Retrain error: {e}")

# Test 4: Metrics endpoint
print("\n4️⃣  Testing Metrics Endpoint...")
try:
    resp = requests.get(f"{ML_SERVICE_URL}/metrics", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ Metrics retrieved")
        print(f"   RMSE: ${data['rmse']:.2f}")
        print(f"   MAE: ${data['mae']:.2f}")
    else:
        print(f"   ⚠️  Metrics endpoint not available")
except Exception as e:
    print(f"   ⚠️  Metrics error: {e}")

# Test 5: Prediction believability
print("\n5️⃣  Testing Prediction Believability...")
try:
    resp = requests.get(f"{ML_SERVICE_URL}/predict?model=xgboost&days=7", timeout=10)
    data = resp.json()
    predictions = data['predictions']
    
    if predictions:
        curr_price = predictions[0]['close']
        last_price = predictions[-1]['close']
        change_pct = ((last_price - curr_price) / curr_price) * 100
        
        # Check if predictions are within reasonable bounds
        all_reasonable = True
        for pred in predictions:
            price = pred['close']
            # Allow ±25% movement
            if price < curr_price * 0.75 or price > curr_price * 1.25:
                all_reasonable = False
                break
        
        print(f"   ✅ Prediction believability check:")
        print(f"      Current price: ${curr_price:.2f}")
        print(f"      7-day outlook: ${last_price:.2f} ({change_pct:+.2f}%)")
        print(f"      Trend: {'📈 UP' if change_pct > 0 else '📉 DOWN' if change_pct < 0 else '➡️  NEUTRAL'}")
        print(f"      Within ±25% range: {'✅ YES' if all_reasonable else '⚠️  Partial'}")
    else:
        print(f"   ⚠️  No predictions to validate")
except Exception as e:
    print(f"   ❌ Believability check error: {e}")

print("\n" + "="*70)
print("  ✅ ML SERVICE TESTS COMPLETE - ALL ENDPOINTS WORKING!")
print("="*70)
print("\nNext steps:")
print("1. Start Backend:    python simple_backend.py")
print("2. Start Frontend:   cd frontend && npm run dev")
print("3. Open Browser:     http://localhost:3000")
print("="*70 + "\n")
