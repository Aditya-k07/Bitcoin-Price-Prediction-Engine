#!/usr/bin/env python3
"""
Comprehensive System Test for Bitcoin Price Prediction Engine
Tests all components: ML Service, Backend, Frontend connectivity, and prediction quality
"""

import requests
import json
import time
import sys
from typing import Dict, Any
from datetime import datetime

# Configuration
ML_SERVICE_URL = "http://localhost:8000"
BACKEND_URL = "http://localhost:8080"
FRONTEND_URL = "http://localhost:3000"

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class SystemTester:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def print_header(self, title: str):
        print(f"\n{BLUE}{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}{RESET}\n")
    
    def print_test(self, name: str, status: bool, details: str = ""):
        status_str = f"{GREEN}✓ PASS{RESET}" if status else f"{RED}✗ FAIL{RESET}"
        print(f"{status_str} | {name}")
        if details:
            print(f"     └─ {details}")
        
        if status:
            self.passed += 1
        else:
            self.failed += 1
        
        self.results.append((name, status, details))
    
    def test_ml_service(self):
        """Test ML Service endpoints"""
        self.print_header("1️⃣  ML SERVICE TESTS")
        
        # Test 1.1: Health Check
        try:
            resp = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status") == "healthy"
                self.print_test(
                    "1.1 ML Service Health Check",
                    status,
                    f"Status: {data.get('status')}, Models loaded: {data.get('models_loaded', 'N/A')}"
                )
            else:
                self.print_test("1.1 ML Service Health Check", False, f"HTTP {resp.status_code}")
        except RequestException as e:
            self.print_test("1.1 ML Service Health Check", False, f"Connection error: {str(e)}")
        
        # Test 1.2: Prediction Endpoint
        try:
            payload = {
                "days": 7,
                "model": "xgboost",
                "currency": "USD"
            }
            resp = requests.post(f"{ML_SERVICE_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                has_predictions = len(data.get("predictions", [])) > 0
                has_metrics = all(k in data for k in ["metrics", "model_used"])
                self.print_test(
                    "1.2 ML Service Predictions",
                    has_predictions and has_metrics,
                    f"Predictions: {len(data.get('predictions', []))}, Model: {data.get('model_used')}"
                )
            else:
                self.print_test("1.2 ML Service Predictions", False, f"HTTP {resp.status_code}")
        except RequestException as e:
            self.print_test("1.2 ML Service Predictions", False, str(e))
        
        # Test 1.3: Model Metrics
        try:
            resp = requests.get(f"{ML_SERVICE_URL}/metrics", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                has_rmse = "rmse" in data
                has_mae = "mae" in data
                self.print_test(
                    "1.3 ML Service Metrics",
                    has_rmse and has_mae,
                    f"RMSE: ${data.get('rmse', 'N/A'):.2f}, MAE: ${data.get('mae', 'N/A'):.2f}"
                )
            else:
                self.print_test("1.3 ML Service Metrics", False, f"HTTP {resp.status_code}")
        except RequestException as e:
            self.print_test("1.3 ML Service Metrics", False, str(e))
    
    def test_backend_service(self):
        """Test Backend endpoints"""
        self.print_header("2️⃣  BACKEND SERVICE TESTS")
        
        # Test 2.1: Backend Health
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self.print_test(
                    "2.1 Backend Health Check",
                    True,
                    f"Status: {data.get('status', 'unknown')}"
                )
            else:
                self.print_test("2.1 Backend Health Check", False, f"HTTP {resp.status_code}")
        except RequestException as e:
            self.print_test("2.1 Backend Health Check", False, f"Connection error: {str(e)}")
        
        # Test 2.2: Historical Data
        try:
            resp = requests.get(f"{BACKEND_URL}/api/historical?limit=30", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                has_data = len(data.get("data", [])) > 0
                self.print_test(
                    "2.2 Historical Data Endpoint",
                    has_data,
                    f"Records returned: {len(data.get('data', []))}"
                )
            else:
                self.print_test("2.2 Historical Data Endpoint", False, f"HTTP {resp.status_code}")
        except RequestException as e:
            self.print_test("2.2 Historical Data Endpoint", False, str(e))
        
        # Test 2.3: Predictions via Backend
        try:
            payload = {"days": 7, "model": "xgboost"}
            resp = requests.post(f"{BACKEND_URL}/api/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                has_predictions = len(data.get("predictions", [])) > 0
                self.print_test(
                    "2.3 Backend Predictions",
                    has_predictions,
                    f"Prediction points: {len(data.get('predictions', []))}"
                )
            else:
                self.print_test("2.3 Backend Predictions", False, f"HTTP {resp.status_code}")
        except RequestException as e:
            self.print_test("2.3 Backend Predictions", False, str(e))
    
    def test_data_quality(self):
        """Test data quality and timeframes"""
        self.print_header("3️⃣  DATA QUALITY TESTS")
        
        try:
            # Get historical data for different timeframes
            timeframes = {
                "7_days": 7,
                "30_days": 30,
                "90_days": 90,
                "1_year": 365
            }
            
            for tf_name, days in timeframes.items():
                try:
                    resp = requests.get(f"{BACKEND_URL}/api/historical?limit={days}", timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        records = data.get("data", [])
                        if records:
                            # Check data completeness
                            required_fields = ["date", "open", "high", "low", "close", "volume"]
                            all_fields_present = all(
                                all(field in record for field in required_fields)
                                for record in records[:5]
                            )
                            self.print_test(
                                f"3.1 Data Quality - {tf_name.replace('_', ' ').title()}",
                                all_fields_present,
                                f"Records: {len(records)}, Fields: {required_fields}"
                            )
                        else:
                            self.print_test(f"3.1 Data Quality - {tf_name}", False, "No data returned")
                    else:
                        self.print_test(f"3.1 Data Quality - {tf_name}", False, f"HTTP {resp.status_code}")
                except RequestException as e:
                    self.print_test(f"3.1 Data Quality - {tf_name}", False, str(e)[:50])
        
        except Exception as e:
            self.print_test("3.1 Data Quality", False, str(e)[:50])
    
    def test_prediction_believability(self):
        """Test if predictions are within believable ranges"""
        self.print_header("4️⃣  PREDICTION BELIEVABILITY TESTS")
        
        try:
            # Get current price and predictions
            resp_hist = requests.get(f"{BACKEND_URL}/api/historical?limit=1", timeout=10)
            if resp_hist.status_code != 200:
                self.print_test("4.1 Prediction Believability", False, "Cannot fetch current price")
                return
            
            current_data = resp_hist.json().get("data", [])
            if not current_data:
                self.print_test("4.1 Prediction Believability", False, "No historical data")
                return
            
            current_price = float(current_data[0].get("close", 0))
            
            # Get predictions
            resp_pred = requests.post(
                f"{BACKEND_URL}/api/predict",
                json={"days": 7, "model": "xgboost"},
                timeout=10
            )
            
            if resp_pred.status_code != 200:
                self.print_test("4.1 Prediction Believability", False, "Prediction endpoint failed")
                return
            
            predictions = resp_pred.json().get("predictions", [])
            
            if not predictions:
                self.print_test("4.1 Prediction Believability", False, "No predictions returned")
                return
            
            pred_prices = [float(p.get("close", 0)) for p in predictions]
            
            # Check believability criteria
            # 1. Predictions within ±20% of current price (reasonable market movement)
            # 2. No extreme jumps between consecutive predictions
            
            within_range = all(
                0.8 * current_price <= price <= 1.2 * current_price
                for price in pred_prices
            )
            
            no_extreme_jumps = True
            for i in range(1, len(pred_prices)):
                change = abs(pred_prices[i] - pred_prices[i-1]) / pred_prices[i-1] if pred_prices[i-1] != 0 else 0
                if change > 0.15:  # More than 15% jump between days
                    no_extreme_jumps = False
                    break
            
            believable = within_range and no_extreme_jumps
            details = (
                f"Current: ${current_price:.2f}, "
                f"Pred Range: ${min(pred_prices):.2f}-${max(pred_prices):.2f}, "
                f"Within ±20%: {within_range}, No extreme jumps: {no_extreme_jumps}"
            )
            
            self.print_test("4.1 Prediction Believability", believable, details)
            
            # Show prediction trend
            if predictions:
                trend = "UP" if pred_prices[-1] > current_price else "DOWN"
                change_pct = ((pred_prices[-1] - current_price) / current_price) * 100
                print(f"     └─ 7-day trend: {trend} ({change_pct:+.2f}%)")
        
        except Exception as e:
            self.print_test("4.1 Prediction Believability", False, str(e)[:50])
    
    def test_frontend_connectivity(self):
        """Test frontend connectivity"""
        self.print_header("5️⃣  FRONTEND CONNECTIVITY TESTS")
        
        try:
            resp = requests.get(FRONTEND_URL, timeout=5)
            frontend_ok = resp.status_code == 200
            self.print_test(
                "5.1 Frontend Server",
                frontend_ok,
                f"HTTP {resp.status_code}"
            )
        except RequestException as e:
            self.print_test("5.1 Frontend Server", False, f"Connection error: {str(e)}")
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("📊 TEST SUMMARY")
        
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        
        print(f"{GREEN}✓ Passed: {self.passed}{RESET}")
        print(f"{RED}✗ Failed: {self.failed}{RESET}")
        print(f"{'─'*40}")
        print(f"Total: {total} | Success Rate: {percentage:.1f}%\n")
        
        if self.failed == 0:
            print(f"{GREEN}🎉 All tests passed! System is operational.{RESET}\n")
        else:
            print(f"{YELLOW}⚠️  {self.failed} test(s) failed. Check the errors above.{RESET}\n")
        
        return self.failed == 0
    
    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{BLUE}{'='*70}")
        print(f"  Bitcoin Price Prediction Engine - System Test")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}{RESET}\n")
        
        self.test_ml_service()
        self.test_backend_service()
        self.test_data_quality()
        self.test_prediction_believability()
        self.test_frontend_connectivity()
        
        success = self.print_summary()
        return success


class RequestException(Exception):
    pass


if __name__ == "__main__":
    print(f"{YELLOW}Starting comprehensive system tests...{RESET}")
    print(f"{YELLOW}Make sure all services are running:{RESET}")
    print(f"  1. ML Service: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print(f"  2. Backend: python simple_backend.py")
    print(f"  3. Frontend: npm run dev (in frontend/)\n")
    
    tester = SystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
