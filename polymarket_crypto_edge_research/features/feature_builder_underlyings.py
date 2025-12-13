"""
Feature builder for underlying assets (BTC/ETH/SOL).
Computes 40+ technical indicators for 15-minute direction prediction.
"""

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core.logging_utils import get_logger
from data.schemas import Candle

logger = get_logger(__name__)

Symbol = Literal["BTC", "ETH", "SOL"]


class UnderlyingFeatures(BaseModel):
    """Features for a single underlying at a point in time."""
    
    symbol: Symbol
    timestamp: datetime
    
    # Price features
    close: float
    returns_1: float  # 1-candle return
    returns_3: float  # 3-candle return
    returns_6: float  # 6-candle (1.5h for 15m)
    returns_12: float  # 12-candle (3h)
    returns_24: float  # 24-candle (6h)
    returns_96: float  # 96-candle (24h)
    
    # Volatility
    volatility_6: float
    volatility_12: float
    volatility_24: float
    atr_14: float
    
    # Momentum
    rsi_6: float
    rsi_14: float
    rsi_24: float
    macd: float
    macd_signal: float
    macd_hist: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    cci_20: float
    
    # Trend
    sma_10: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    price_vs_sma_20: float
    price_vs_sma_50: float
    trend_strength: float
    
    # Volume
    volume_ratio_6: float
    volume_ratio_12: float
    volume_trend: float
    obv_slope: float
    
    # Bollinger Bands
    bb_position: float
    bb_width: float
    bb_squeeze: float
    
    # Candlestick
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    consecutive_up: int
    consecutive_down: int
    
    # Cross-asset (relative to BTC if not BTC)
    relative_strength: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dict for ML model."""
        return {
            "returns_1": self.returns_1,
            "returns_3": self.returns_3,
            "returns_6": self.returns_6,
            "returns_12": self.returns_12,
            "returns_24": self.returns_24,
            "returns_96": self.returns_96,
            "volatility_6": self.volatility_6,
            "volatility_12": self.volatility_12,
            "volatility_24": self.volatility_24,
            "atr_14": self.atr_14,
            "rsi_6": self.rsi_6,
            "rsi_14": self.rsi_14,
            "rsi_24": self.rsi_24,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_hist": self.macd_hist,
            "stoch_k": self.stoch_k,
            "stoch_d": self.stoch_d,
            "williams_r": self.williams_r,
            "cci_20": self.cci_20,
            "price_vs_sma_20": self.price_vs_sma_20,
            "price_vs_sma_50": self.price_vs_sma_50,
            "trend_strength": self.trend_strength,
            "volume_ratio_6": self.volume_ratio_6,
            "volume_ratio_12": self.volume_ratio_12,
            "volume_trend": self.volume_trend,
            "obv_slope": self.obv_slope,
            "bb_position": self.bb_position,
            "bb_width": self.bb_width,
            "bb_squeeze": self.bb_squeeze,
            "body_ratio": self.body_ratio,
            "upper_wick_ratio": self.upper_wick_ratio,
            "lower_wick_ratio": self.lower_wick_ratio,
            "consecutive_up": float(self.consecutive_up),
            "consecutive_down": float(self.consecutive_down),
            "relative_strength": self.relative_strength,
        }


class UnderlyingsFeatureBuilder:
    """
    Builds features for underlying assets from candle data.
    """
    
    def __init__(self):
        self._feature_names = list(UnderlyingFeatures.model_fields.keys())
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get list of feature names."""
        return [
            "returns_1", "returns_3", "returns_6", "returns_12", "returns_24", "returns_96",
            "volatility_6", "volatility_12", "volatility_24", "atr_14",
            "rsi_6", "rsi_14", "rsi_24",
            "macd", "macd_signal", "macd_hist",
            "stoch_k", "stoch_d", "williams_r", "cci_20",
            "price_vs_sma_20", "price_vs_sma_50", "trend_strength",
            "volume_ratio_6", "volume_ratio_12", "volume_trend", "obv_slope",
            "bb_position", "bb_width", "bb_squeeze",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "consecutive_up", "consecutive_down",
            "relative_strength",
        ]
    
    def build(
        self,
        symbol: Symbol,
        candles: list[Candle],
        btc_candles: list[Candle] | None = None
    ) -> UnderlyingFeatures | None:
        """
        Build features from candle data.
        
        Args:
            symbol: Asset symbol
            candles: List of candles (oldest first)
            btc_candles: BTC candles for relative strength (optional)
            
        Returns:
            UnderlyingFeatures or None if insufficient data
        """
        if len(candles) < 100:
            logger.warning(f"Insufficient candles for {symbol}: {len(candles)}")
            return None
        
        # Convert to numpy arrays
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        opens = np.array([c.open for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        # Current values
        close = closes[-1]
        timestamp = candles[-1].timestamp
        
        # Returns
        returns_1 = self._pct_change(closes, 1)
        returns_3 = self._pct_change(closes, 3)
        returns_6 = self._pct_change(closes, 6)
        returns_12 = self._pct_change(closes, 12)
        returns_24 = self._pct_change(closes, 24)
        returns_96 = self._pct_change(closes, 96)
        
        # Volatility
        volatility_6 = self._volatility(closes, 6)
        volatility_12 = self._volatility(closes, 12)
        volatility_24 = self._volatility(closes, 24)
        atr_14 = self._atr(highs, lows, closes, 14)
        
        # RSI
        rsi_6 = self._rsi(closes, 6)
        rsi_14 = self._rsi(closes, 14)
        rsi_24 = self._rsi(closes, 24)
        
        # MACD
        macd, signal, hist = self._macd(closes)
        
        # Stochastic
        stoch_k, stoch_d = self._stochastic(highs, lows, closes)
        
        # Williams %R
        williams_r = self._williams_r(highs, lows, closes)
        
        # CCI
        cci_20 = self._cci(highs, lows, closes, 20)
        
        # Moving averages
        sma_10 = np.mean(closes[-10:])
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        ema_12 = self._ema(closes, 12)[-1]
        ema_26 = self._ema(closes, 26)[-1]
        
        price_vs_sma_20 = (close / sma_20) - 1 if sma_20 > 0 else 0
        price_vs_sma_50 = (close / sma_50) - 1 if sma_50 > 0 else 0
        
        # Trend strength (ADX-like)
        trend_strength = self._trend_strength(closes, 14)
        
        # Volume
        avg_vol_6 = np.mean(volumes[-7:-1]) if len(volumes) > 7 else np.mean(volumes)
        avg_vol_12 = np.mean(volumes[-13:-1]) if len(volumes) > 13 else np.mean(volumes)
        volume_ratio_6 = volumes[-1] / avg_vol_6 if avg_vol_6 > 0 else 1.0
        volume_ratio_12 = volumes[-1] / avg_vol_12 if avg_vol_12 > 0 else 1.0
        
        recent_vol = np.mean(volumes[-3:])
        older_vol = np.mean(volumes[-6:-3]) if len(volumes) > 6 else recent_vol
        volume_trend = (recent_vol / older_vol) - 1 if older_vol > 0 else 0
        
        obv_slope = self._obv_slope(closes, volumes, 10)
        
        # Bollinger Bands
        bb_pos, bb_width, bb_squeeze = self._bollinger(closes, 20, 2.0)
        
        # Candlestick
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
        
        # Consecutive candles
        consecutive_up, consecutive_down = self._consecutive_candles(closes)
        
        # Relative strength vs BTC
        relative_strength = 0.0
        if btc_candles and symbol != "BTC" and len(btc_candles) >= 24:
            btc_closes = np.array([c.close for c in btc_candles])
            btc_return = self._pct_change(btc_closes, 24)
            asset_return = returns_24
            relative_strength = asset_return - btc_return
        
        return UnderlyingFeatures(
            symbol=symbol,
            timestamp=timestamp,
            close=close,
            returns_1=returns_1,
            returns_3=returns_3,
            returns_6=returns_6,
            returns_12=returns_12,
            returns_24=returns_24,
            returns_96=returns_96,
            volatility_6=volatility_6,
            volatility_12=volatility_12,
            volatility_24=volatility_24,
            atr_14=atr_14,
            rsi_6=rsi_6,
            rsi_14=rsi_14,
            rsi_24=rsi_24,
            macd=macd,
            macd_signal=signal,
            macd_hist=hist,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            williams_r=williams_r,
            cci_20=cci_20,
            sma_10=sma_10,
            sma_20=sma_20,
            sma_50=sma_50,
            ema_12=ema_12,
            ema_26=ema_26,
            price_vs_sma_20=price_vs_sma_20,
            price_vs_sma_50=price_vs_sma_50,
            trend_strength=trend_strength,
            volume_ratio_6=volume_ratio_6,
            volume_ratio_12=volume_ratio_12,
            volume_trend=volume_trend,
            obv_slope=obv_slope,
            bb_position=bb_pos,
            bb_width=bb_width,
            bb_squeeze=bb_squeeze,
            body_ratio=body_ratio,
            upper_wick_ratio=upper_wick,
            lower_wick_ratio=lower_wick,
            consecutive_up=consecutive_up,
            consecutive_down=consecutive_down,
            relative_strength=relative_strength,
        )
    
    def _pct_change(self, arr: np.ndarray, periods: int) -> float:
        """Calculate percentage change."""
        if len(arr) <= periods:
            return 0.0
        return float((arr[-1] / arr[-periods - 1]) - 1)
    
    def _volatility(self, closes: np.ndarray, period: int) -> float:
        """Calculate volatility (std of returns)."""
        if len(closes) <= period:
            return 0.0
        returns = np.diff(closes[-period - 1:]) / closes[-period - 1:-1]
        return float(np.std(returns))
    
    def _rsi(self, closes: np.ndarray, period: int) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _macd(self, closes: np.ndarray) -> tuple[float, float, float]:
        """Calculate MACD, Signal, Histogram."""
        if len(closes) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        macd_line = ema_12 - ema_26
        
        signal_line = self._ema(macd_line, 9) if len(macd_line) >= 9 else macd_line
        
        macd = macd_line[-1]
        signal = signal_line[-1]
        histogram = macd - signal
        
        # Normalize by price
        price = closes[-1]
        return float(macd / price), float(signal / price), float(histogram / price)
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Calculate ATR (normalized)."""
        if len(closes) < period + 1:
            return 0.0
        
        tr_list = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr_list.append(max(high_low, high_close, low_close))
        
        atr = np.mean(tr_list[-period:])
        return float(atr / closes[-1])
    
    def _stochastic(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
        """Calculate Stochastic K and D."""
        if len(closes) < k_period:
            return 50.0, 50.0
        
        lowest_low = np.min(lows[-k_period:])
        highest_high = np.max(highs[-k_period:])
        
        if highest_high == lowest_low:
            return 50.0, 50.0
        
        k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)
        
        # Simple D (SMA of K)
        d = k  # Simplified
        
        return float(k), float(d)
    
    def _williams_r(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Williams %R."""
        if len(closes) < period:
            return -50.0
        
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        
        if highest_high == lowest_low:
            return -50.0
        
        wr = -100 * (highest_high - closes[-1]) / (highest_high - lowest_low)
        return float(wr)
    
    def _cci(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Calculate Commodity Channel Index."""
        if len(closes) < period:
            return 0.0
        
        tp = (highs + lows + closes) / 3
        sma = np.mean(tp[-period:])
        mad = np.mean(np.abs(tp[-period:] - sma))
        
        if mad == 0:
            return 0.0
        
        cci = (tp[-1] - sma) / (0.015 * mad)
        return float(cci) / 100  # Normalize
    
    def _trend_strength(self, closes: np.ndarray, period: int) -> float:
        """Calculate trend strength (0-1)."""
        if len(closes) < period:
            return 0.5
        
        # Use linear regression slope as trend indicator
        x = np.arange(period)
        y = closes[-period:]
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by price and period
        normalized = slope * period / closes[-1]
        
        # Convert to 0-1 range
        return float(np.clip(0.5 + normalized * 10, 0, 1))
    
    def _obv_slope(self, closes: np.ndarray, volumes: np.ndarray, period: int) -> float:
        """Calculate OBV slope."""
        if len(closes) < period + 1:
            return 0.0
        
        obv = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        
        obv = np.array(obv[-period:])
        x = np.arange(len(obv))
        
        slope = np.polyfit(x, obv, 1)[0]
        
        # Normalize
        return float(slope / (np.mean(volumes[-period:]) + 1e-10))
    
    def _bollinger(self, closes: np.ndarray, period: int, std_mult: float) -> tuple[float, float, float]:
        """Calculate Bollinger Band position, width, and squeeze."""
        if len(closes) < period:
            return 0.0, 0.0, 0.0
        
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        
        if std == 0:
            return 0.0, 0.0, 1.0
        
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        
        # Position (-1 to 1)
        band_width = upper - lower
        position = ((closes[-1] - lower) / band_width) * 2 - 1 if band_width > 0 else 0
        
        # Width (normalized)
        width = band_width / sma if sma > 0 else 0
        
        # Squeeze (compare to historical width)
        historical_widths = []
        for i in range(20, len(closes)):
            h_sma = np.mean(closes[i-period:i])
            h_std = np.std(closes[i-period:i])
            h_width = (2 * std_mult * h_std) / h_sma if h_sma > 0 else 0
            historical_widths.append(h_width)
        
        if historical_widths:
            avg_width = np.mean(historical_widths)
            squeeze = width / avg_width if avg_width > 0 else 1.0
        else:
            squeeze = 1.0
        
        return float(np.clip(position, -2, 2)), float(width), float(squeeze)
    
    def _consecutive_candles(self, closes: np.ndarray) -> tuple[int, int]:
        """Count consecutive up/down candles."""
        up_count = 0
        down_count = 0
        
        # Count consecutive up
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i-1]:
                up_count += 1
            else:
                break
        
        # Count consecutive down
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i-1]:
                down_count += 1
            else:
                break
        
        return up_count, down_count


def build_underlying_features(
    symbol: Symbol,
    candles: list[Candle],
    btc_candles: list[Candle] | None = None
) -> UnderlyingFeatures | None:
    """Convenience function to build features."""
    builder = UnderlyingsFeatureBuilder()
    return builder.build(symbol, candles, btc_candles)
