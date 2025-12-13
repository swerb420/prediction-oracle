"""
Technical indicator feature engineering for ML predictions.
Computes features from OHLCV candle data for price direction prediction.
"""

import numpy as np
from typing import Any
from pydantic import BaseModel

from .crypto_data import CandleData


class FeatureSet(BaseModel):
    """Computed features for ML model."""
    symbol: str
    timestamp: str  # ISO format of latest candle
    
    # Price features
    price_change_1: float  # 1-candle return
    price_change_3: float  # 3-candle return
    price_change_6: float  # 6-candle return (1.5 hours for 15m)
    price_change_12: float  # 12-candle return (3 hours)
    
    # Volatility
    volatility_6: float  # Std dev of returns over 6 candles
    volatility_12: float
    atr_14: float  # Average True Range (normalized)
    
    # Momentum indicators
    rsi_14: float  # Relative Strength Index
    rsi_6: float   # Faster RSI
    macd_signal: float  # MACD - Signal line
    macd_histogram: float
    
    # Volume features
    volume_ratio_6: float  # Current volume / 6-candle avg
    volume_trend: float  # Volume momentum
    
    # Price position
    bb_position: float  # Position within Bollinger Bands (-1 to 1)
    price_vs_sma_20: float  # Price / SMA20 - 1
    price_vs_ema_12: float  # Price / EMA12 - 1
    
    # Candlestick patterns
    body_ratio: float  # (close - open) / (high - low)
    upper_wick_ratio: float
    lower_wick_ratio: float
    
    # Recent price
    current_price: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array([
            self.price_change_1,
            self.price_change_3,
            self.price_change_6,
            self.price_change_12,
            self.volatility_6,
            self.volatility_12,
            self.atr_14,
            self.rsi_14,
            self.rsi_6,
            self.macd_signal,
            self.macd_histogram,
            self.volume_ratio_6,
            self.volume_trend,
            self.bb_position,
            self.price_vs_sma_20,
            self.price_vs_ema_12,
            self.body_ratio,
            self.upper_wick_ratio,
            self.lower_wick_ratio,
        ])
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get feature names for model."""
        return [
            "price_change_1",
            "price_change_3",
            "price_change_6",
            "price_change_12",
            "volatility_6",
            "volatility_12",
            "atr_14",
            "rsi_14",
            "rsi_6",
            "macd_signal",
            "macd_histogram",
            "volume_ratio_6",
            "volume_trend",
            "bb_position",
            "price_vs_sma_20",
            "price_vs_ema_12",
            "body_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
        ]


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI for the most recent candle."""
    if len(closes) < period + 1:
        return 50.0  # Neutral default
    
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


def compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Compute Exponential Moving Average."""
    ema = np.zeros_like(data)
    multiplier = 2 / (period + 1)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def compute_macd(closes: np.ndarray) -> tuple[float, float, float]:
    """
    Compute MACD, Signal, and Histogram.
    Returns: (macd, signal, histogram)
    """
    if len(closes) < 26:
        return 0.0, 0.0, 0.0
    
    ema_12 = compute_ema(closes, 12)
    ema_26 = compute_ema(closes, 26)
    macd_line = ema_12 - ema_26
    
    # Signal line (9-period EMA of MACD)
    if len(macd_line) >= 9:
        signal_line = compute_ema(macd_line, 9)
        signal = signal_line[-1]
    else:
        signal = macd_line[-1]
    
    macd = macd_line[-1]
    histogram = macd - signal
    
    # Normalize by price
    price = closes[-1]
    return float(macd / price), float(signal / price), float(histogram / price)


def compute_bollinger_position(closes: np.ndarray, period: int = 20, std_mult: float = 2.0) -> float:
    """
    Compute position within Bollinger Bands.
    Returns -1 at lower band, 0 at middle, +1 at upper band.
    """
    if len(closes) < period:
        return 0.0
    
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    
    if std == 0:
        return 0.0
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    current = closes[-1]
    
    # Normalize to -1 to +1 range
    band_width = upper - lower
    if band_width == 0:
        return 0.0
    
    position = (current - lower) / band_width * 2 - 1
    return float(np.clip(position, -2, 2))  # Allow slight overflow


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute normalized Average True Range."""
    if len(closes) < period + 1:
        return 0.0
    
    tr_list = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr_list.append(max(high_low, high_close, low_close))
    
    if len(tr_list) < period:
        return 0.0
    
    atr = np.mean(tr_list[-period:])
    # Normalize by current price
    return float(atr / closes[-1])


def extract_features(candle_data: CandleData) -> FeatureSet:
    """
    Extract all ML features from candle data.
    
    Requires at least 30 candles for reliable features.
    """
    closes = candle_data.closes
    highs = candle_data.highs
    lows = candle_data.lows
    volumes = candle_data.volumes
    opens = candle_data.opens
    
    current_price = closes[-1]
    
    # Price changes (returns)
    def pct_change(periods: int) -> float:
        if len(closes) <= periods:
            return 0.0
        return float((closes[-1] / closes[-periods - 1]) - 1)
    
    # Volatility (std of returns)
    def volatility(periods: int) -> float:
        if len(closes) <= periods:
            return 0.0
        returns = np.diff(closes[-periods - 1:]) / closes[-periods - 1:-1]
        return float(np.std(returns))
    
    # Volume analysis
    avg_volume_6 = np.mean(volumes[-7:-1]) if len(volumes) > 7 else np.mean(volumes)
    volume_ratio = float(volumes[-1] / avg_volume_6) if avg_volume_6 > 0 else 1.0
    
    volume_trend = 0.0
    if len(volumes) > 6:
        recent_vol = np.mean(volumes[-3:])
        older_vol = np.mean(volumes[-6:-3])
        if older_vol > 0:
            volume_trend = float((recent_vol / older_vol) - 1)
    
    # MACD
    macd, signal, histogram = compute_macd(closes)
    
    # Moving averages
    sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
    ema_12 = compute_ema(closes, 12)[-1] if len(closes) >= 12 else closes[-1]
    
    price_vs_sma = (current_price / sma_20) - 1 if sma_20 > 0 else 0
    price_vs_ema = (current_price / ema_12) - 1 if ema_12 > 0 else 0
    
    # Candlestick analysis (latest candle)
    candle_range = highs[-1] - lows[-1]
    body = closes[-1] - opens[-1]
    
    if candle_range > 0:
        body_ratio = body / candle_range
        upper_wick = (highs[-1] - max(opens[-1], closes[-1])) / candle_range
        lower_wick = (min(opens[-1], closes[-1]) - lows[-1]) / candle_range
    else:
        body_ratio = 0.0
        upper_wick = 0.0
        lower_wick = 0.0
    
    return FeatureSet(
        symbol=candle_data.symbol,
        timestamp=candle_data.candles[-1].timestamp.isoformat(),
        
        price_change_1=pct_change(1),
        price_change_3=pct_change(3),
        price_change_6=pct_change(6),
        price_change_12=pct_change(12),
        
        volatility_6=volatility(6),
        volatility_12=volatility(12),
        atr_14=compute_atr(highs, lows, closes, 14),
        
        rsi_14=compute_rsi(closes, 14),
        rsi_6=compute_rsi(closes, 6),
        macd_signal=macd - signal,  # Normalized MACD diff
        macd_histogram=histogram,
        
        volume_ratio_6=volume_ratio,
        volume_trend=volume_trend,
        
        bb_position=compute_bollinger_position(closes),
        price_vs_sma_20=float(price_vs_sma),
        price_vs_ema_12=float(price_vs_ema),
        
        body_ratio=float(body_ratio),
        upper_wick_ratio=float(upper_wick),
        lower_wick_ratio=float(lower_wick),
        
        current_price=float(current_price),
    )


def extract_all_features(
    candle_data: dict[str, CandleData]
) -> dict[str, FeatureSet]:
    """Extract features for all symbols."""
    return {
        symbol: extract_features(data)
        for symbol, data in candle_data.items()
    }
