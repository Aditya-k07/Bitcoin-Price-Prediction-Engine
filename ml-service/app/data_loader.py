"""
Data loader for Bitcoin historical data.
Handles loading the Kaggle CSV dataset and resampling from minute-level to daily OHLCV.
"""

import os
import logging
import pandas as pd
import numpy as np

import requests
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Path to the data directory within the ml-service package
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Cache for historical data to avoid hitting rate limits
CACHE_PATH = os.path.join(DATA_DIR, "api_historical_cache.csv")


def fetch_binance_ohlcv(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 1000, start_time: int = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance Public API.
    
    Binance returns:
    [
      [
        1499040000000,      // Open time
        "0.01634790",       // Open
        "0.80000000",       // High
        "0.01575800",       // Low
        "0.01577100",       // Close
        "148976.11427815",  // Volume
        1499644799999,      // Close time
        "2434.19055334",    // Quote asset volume
        308,                // Number of trades
        "1756.87402397",    // Taker buy base asset volume
        "28.46694368",      // Taker buy quote asset volume
        "17928899.62484339" // Ignore
      ]
    ]
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["startTime"] = start_time
    
    logger.info(f"Fetching Binance {interval} data for {symbol} (start_time={start_time})...")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
            
        columns = [
            "timestamp", "Open", "High", "Low", "Close", "Volume", 
            "close_time", "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(data, columns=columns)
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
            
        df.set_index("timestamp", inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]
        
    except Exception as e:
        logger.error(f"Failed to fetch from Binance: {e}")
        return pd.DataFrame()

def fetch_full_history_binance(symbol: str = "BTCUSDT", years: int = 4) -> pd.DataFrame:
    """
    Fetch long-term history from Binance using pagination.
    """
    limit = 1000
    all_dfs = []
    
    # Approx start time (e.g. 4 years ago)
    current_ts = int(time.time() * 1000)
    start_ts = current_ts - (years * 365 * 24 * 60 * 60 * 1000)
    
    while True:
        df = fetch_binance_ohlcv(symbol=symbol, start_time=start_ts, limit=limit)
        if df.empty:
            break
            
        all_dfs.append(df)
        
        # Next start time is the last timestamp + 1 interval (1 day)
        last_ts = int(df.index[-1].timestamp() * 1000)
        if last_ts >= current_ts - (24 * 60 * 60 * 1000): # Stop if we reached today
            break
            
        if start_ts == last_ts + (24 * 60 * 60 * 1000): # Prevent infinite loop
            break
            
        start_ts = last_ts + (24 * 60 * 60 * 1000)
        time.sleep(0.1) # Respect rate limits
        
    if not all_dfs:
        return pd.DataFrame()
        
    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined

def load_daily_data(csv_path: str = None, force_sync: bool = False) -> pd.DataFrame:
    """
    Load historical data using Binance API with local caching and backfilling.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    target_years = 5
    
    df_cached = pd.DataFrame()
    if os.path.exists(CACHE_PATH):
        try:
            df_cached = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df_cached)} rows from cache ({CACHE_PATH})")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")

    # 1. Backfill missing history
    desired_start = datetime.utcnow() - pd.Timedelta(days=365 * target_years)
    if df_cached.empty or df_cached.index.min() > desired_start:
        logger.info(f"Backfilling {target_years} years of history...")
        df_full = fetch_full_history_binance(years=target_years)
        if not df_full.empty:
            df_cached = pd.concat([df_cached, df_full])
            df_cached = df_cached[~df_cached.index.duplicated(keep='last')].sort_index()

    # 2. Delta update for newest data
    if not df_cached.empty:
        last_ts = int(df_cached.index[-1].timestamp() * 1000)
        # Fetch delta if forced OR if last point is > 5 minutes old
        # When force_sync=True, we bypass the safety check and pull live data
        if force_sync or last_ts < int((time.time() - 300) * 1000):
            logger.info(f"Syncing newest data (force_sync={force_sync})...")
            # Pull 1h data for the last few hours to ensure the daily candle is updated with live price
            df_fresh = fetch_binance_ohlcv(start_time=last_ts)
            if not df_fresh.empty:
                df_cached = pd.concat([df_cached, df_fresh])
                # Ensure we drop duplicates and keep the newest (live) data
                df_cached = df_cached[~df_cached.index.duplicated(keep='last')].sort_index()

    # Save to cache
    if not df_cached.empty:
        df_cached.to_csv(CACHE_PATH)
        logger.info(f"Cache synchronized. Total records: {len(df_cached)}")
        
    return df_cached




def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-level data to daily OHLCV.

    Uses standard OHLCV resampling rules:
    - Open: first value of the day
    - High: max value of the day
    - Low: min value of the day
    - Close: last value of the day
    - Volume: sum of the day

    Args:
        df: DataFrame with datetime index and OHLCV columns.

    Returns:
        Daily OHLCV DataFrame with NaN rows dropped.
    """
    logger.info("Resampling to daily OHLCV...")

    ohlcv_rules = {}
    if "Open" in df.columns:
        ohlcv_rules["Open"] = "first"
    if "High" in df.columns:
        ohlcv_rules["High"] = "max"
    if "Low" in df.columns:
        ohlcv_rules["Low"] = "min"
    if "Close" in df.columns:
        ohlcv_rules["Close"] = "last"

    # Handle volume column (might be named differently)
    vol_col = None
    for candidate in ["Volume_BTC", "Volume", "Volume_Currency"]:
        if candidate in df.columns:
            vol_col = candidate
            break

    if vol_col:
        ohlcv_rules[vol_col] = "sum"

    daily = df.resample("D").agg(ohlcv_rules)

    # Rename volume column to standard name
    if vol_col and vol_col != "Volume":
        daily.rename(columns={vol_col: "Volume"}, inplace=True)

    # Drop rows where all OHLC values are NaN (no trading that day)
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in daily.columns]
    daily.dropna(subset=price_cols, how="all", inplace=True)

    # Forward-fill small gaps (weekends/holidays shouldn't exist for crypto, but just in case)
    daily.ffill(inplace=True)

    # Drop any remaining NaN rows
    daily.dropna(inplace=True)

    logger.info(f"Resampled to {len(daily):,} daily records, "
                f"date range: {daily.index.min().date()} to {daily.index.max().date()}")
    return daily

def load_raw_data(csv_path: str = None) -> pd.DataFrame:
    """Legacy: redirected to API-based loading."""
    return load_daily_data(csv_path)




def load_from_coingecko_ohlc(candles: list) -> pd.DataFrame:
    """
    Load and process fresh OHLC data from CoinGecko.

    Expects a list of dicts with keys: timestamp (Unix ms), open, high, low, close.
    Converts to daily OHLCV DataFrame compatible with feature engineering.

    Args:
        candles: List of dicts with OHLC data from CoinGecko.

    Returns:
        Daily OHLCV DataFrame ready for feature engineering.

    Raises:
        ValueError: If data is empty or malformed.
    """
    if not candles:
        raise ValueError("No OHLC data provided")

    try:
        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Convert timestamp from milliseconds to seconds if needed
        if "timestamp" in df.columns:
            # Check if timestamp is in milliseconds (13 digits) or seconds (10 digits)
            sample_ts = df["timestamp"].iloc[0]
            if sample_ts > 1e11:  # milliseconds
                df["timestamp"] = df["timestamp"] / 1000

            df["Timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.drop(columns=["timestamp"], inplace=True)
        else:
            raise ValueError("Missing 'timestamp' column in OHLC data")

        # Standardize column names to match expected format
        col_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        }
        df.rename(columns=col_mapping, inplace=True)

        # Set datetime index
        df.set_index("Timestamp", inplace=True)

        logger.info(f"Loaded {len(df):,} candles from CoinGecko, date range: {df.index.min().date()} to {df.index.max().date()}")

        # For daily data from CoinGecko, resample to ensure consistency
        # (in case some data points are missing)
        daily = resample_to_daily(df)

        return daily

    except Exception as e:
        logger.error(f"Failed to load CoinGecko OHLC data: {e}")
        raise ValueError(f"Invalid OHLC data format: {e}")
