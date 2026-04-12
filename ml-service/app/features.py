"""
Feature engineering pipeline for Bitcoin price prediction.

Generates features from daily OHLCV data:
- Lag values (past prices at various horizons)
- Rolling averages (SMA at various windows)
- Technical indicators (RSI, volatility, daily returns)
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Configuration
LAG_DAYS = [1, 3, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]
RSI_PERIOD = 14
VOLATILITY_WINDOW = 14


def add_lag_features(df: pd.DataFrame, target_col: str = "Close") -> pd.DataFrame:
    """
    Add lag features: what was the price N days ago?

    Lag values help the model learn from recent price history.
    For example, lag_1 = yesterday's close, lag_7 = last week's close.

    Args:
        df: Daily OHLCV DataFrame.
        target_col: Column to create lags from.

    Returns:
        DataFrame with lag columns added.
    """
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
        logger.debug(f"Added lag_{lag} feature")
    return df


def add_rolling_averages(df: pd.DataFrame, target_col: str = "Close") -> pd.DataFrame:
    """
    Add rolling average (Simple Moving Average) features.

    Rolling averages smooth out daily volatility to reveal underlying trends.
    - SMA_7: short-term trend
    - SMA_14: medium-term trend
    - SMA_30: long-term trend

    Args:
        df: Daily OHLCV DataFrame.
        target_col: Column to compute rolling averages from.

    Returns:
        DataFrame with SMA columns added.
    """
    for window in ROLLING_WINDOWS:
        df[f"sma_{window}"] = df[target_col].rolling(window=window).mean()
        logger.debug(f"Added sma_{window} feature")
    return df


def add_rsi(df: pd.DataFrame, target_col: str = "Close", period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI).

    RSI measures momentum: values above 70 indicate overbought conditions,
    below 30 indicate oversold. This helps the model detect reversal signals.

    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = average gain / average loss over the period.

    Args:
        df: Daily OHLCV DataFrame.
        target_col: Column to compute RSI from.
        period: Lookback period (default: 14 days).

    Returns:
        DataFrame with RSI column added.
    """
    delta = df[target_col].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Fill NaN RSI with 50 (neutral)
    df["rsi"].fillna(50, inplace=True)

    logger.debug(f"Added RSI feature (period={period})")
    return df


def add_volatility(df: pd.DataFrame, target_col: str = "Close",
                   window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    """
    Add rolling volatility (standard deviation of daily returns).

    Higher volatility means more uncertainty in price movements.
    This feature helps the model adjust predictions during turbulent periods.

    Args:
        df: Daily OHLCV DataFrame.
        target_col: Column to compute volatility from.
        window: Rolling window size.

    Returns:
        DataFrame with volatility column added.
    """
    daily_returns = df[target_col].pct_change()
    df["daily_return"] = daily_returns
    df["volatility"] = daily_returns.rolling(window=window).std()

    logger.debug(f"Added volatility feature (window={window})")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived price features from OHLCV data.

    - price_range: High - Low (daily trading range)
    - oc_diff: Close - Open (daily price movement direction)

    Args:
        df: Daily OHLCV DataFrame.

    Returns:
        DataFrame with price feature columns added.
    """
    if "High" in df.columns and "Low" in df.columns:
        df["price_range"] = df["High"] - df["Low"]

    if "Open" in df.columns and "Close" in df.columns:
        df["oc_diff"] = df["Close"] - df["Open"]

    logger.debug("Added price range and OC diff features")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Applies all feature transformations and drops rows with NaN values
    that result from lag/rolling computations.

    Args:
        df: Daily OHLCV DataFrame from data_loader.

    Returns:
        Feature-engineered DataFrame ready for model training.
        NaN rows from rolling/lag computations are dropped.
    """
    logger.info("Starting feature engineering pipeline...")

    df = df.copy()

    # Add the target column: next-day close price
    df["target"] = df["Close"].shift(-1)

    # Add all features
    df = add_lag_features(df)
    df = add_rolling_averages(df)
    df = add_rsi(df)
    df = add_volatility(df)
    df = add_price_features(df)

    # Momentum (% change over N days) — compact signal for linear + tree models
    df["mom_7"] = df["Close"].pct_change(7)
    df["mom_14"] = df["Close"].pct_change(14)

    # Drop rows with NaN (from lag/rolling window computations + last row target)
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)

    logger.info(f"Feature engineering complete. "
                f"Features: {len(get_feature_columns(df))}, "
                f"Samples: {len(df)} (dropped {dropped} NaN rows)")
    return df


def rsi_from_close_history(close_history: list, period: int = RSI_PERIOD) -> float:
    """
    RSI from a list of closes (matches training-time RSI logic for inference).
    """
    if len(close_history) <= period:
        return 50.0
    changes = np.diff(np.asarray(close_history[-(period + 1) :], dtype=float))
    gains = changes[changes > 0]
    losses = -changes[changes < 0]
    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get the list of feature column names (excludes target and raw OHLCV).

    Returns:
        List of feature column names used for model training.
    """
    exclude = {"Open", "High", "Low", "Close", "Volume", "Volume_BTC",
               "Volume_Currency", "Weighted_Price", "target"}
    return [col for col in df.columns if col not in exclude]
