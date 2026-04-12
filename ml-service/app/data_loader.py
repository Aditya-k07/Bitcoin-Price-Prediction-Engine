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

# Path to the dataset relative to the ml-service directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "btc_historical.csv")


def fetch_coingecko_ohlc(days: int = 30, currency: str = "usd") -> pd.DataFrame:
    """
    Fetch the latest OHLC data from CoinGecko public API.
    
    CoinGecko free tier /coins/bitcoin/ohlc endpoint returns:
    [
      [timestamp, open, high, low, close],
      ...
    ]
    Intervals: 1/7/14/30 days = 30 min intervals; 90+ days = 4 hour intervals.
    """
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {
        "vs_currency": currency,
        "days": days
    }
    
    logger.info(f"Fetching latest {days} days of BTC/{currency} from CoinGecko...")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError("Empty data returned from CoinGecko")
            
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Resample to daily to match our model's expectation
        daily = resample_to_daily(df)
        logger.info(f"Successfully fetched and processed {len(daily)} days of fresh data from CoinGecko.")
        return daily
        
    except Exception as e:
        logger.error(f"Failed to fetch data from CoinGecko: {e}")
        return pd.DataFrame()


def load_raw_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load the raw Bitcoin historical CSV from Kaggle.

    The Kaggle dataset (mczielinski/bitcoin-historical-data) has columns:
    - Timestamp (Unix timestamp)
    - Open, High, Low, Close (prices in USD)
    - Volume_(BTC), Volume_(Currency)
    - Weighted_Price

    Args:
        csv_path: Path to the CSV file. Defaults to data/btc_historical.csv.

    Returns:
        DataFrame with parsed datetime index and OHLCV columns.
    """
    path = csv_path or DEFAULT_CSV_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please download it from: "
            "https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data "
            "and place it in the ml-service/data/ directory."
        )

    logger.info(f"Loading raw data from {path}...")

    df = pd.read_csv(path)

    # The Kaggle dataset uses 'Timestamp' as Unix timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
        df.set_index("Timestamp", inplace=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
    else:
        # Try to parse the first column as datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_")
        if "open" in lower:
            col_mapping[col] = "Open"
        elif "high" in lower:
            col_mapping[col] = "High"
        elif "low" in lower:
            col_mapping[col] = "Low"
        elif "close" in lower:
            col_mapping[col] = "Close"
        elif "volume_(btc)" in lower or "volume_btc" in lower:
            col_mapping[col] = "Volume_BTC"
        elif "volume_(currency)" in lower or "volume_currency" in lower:
            col_mapping[col] = "Volume_Currency"
        elif "volume" in lower:
            col_mapping[col] = "Volume"
        elif "weighted" in lower:
            col_mapping[col] = "Weighted_Price"

    df.rename(columns=col_mapping, inplace=True)

    logger.info(f"Loaded {len(df):,} rows, date range: {df.index.min()} to {df.index.max()}")
    return df


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


def load_daily_data(csv_path: str = None) -> pd.DataFrame:
    """
    Convenience function: load raw data and resample to daily.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Daily OHLCV DataFrame ready for feature engineering.
    """
    raw = load_raw_data(csv_path)
    daily = resample_to_daily(raw)
    return daily


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
