# ML Service Data Directory

Place the Bitcoin historical dataset here:

**Download from**: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data

**Expected file**: `btc_historical.csv`

The CSV should contain minute-level Bitcoin price data with columns like:
- Timestamp
- Open, High, Low, Close
- Volume_(BTC), Volume_(Currency)
- Weighted_Price

The data loader will automatically resample this to daily OHLCV.
