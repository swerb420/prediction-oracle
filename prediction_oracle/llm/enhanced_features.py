"""
Enhanced Feature Engineering with Multi-Venue and Whale Signals
=================================================================

Extends base feature engineering with:
1. Cross-venue features (arb spreads, depth imbalance)
2. Whale consensus features (Polymarket trader signals)
3. Combined feature set for enhanced ML predictions

Target: 63%+ win rate through additional alpha sources.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from .crypto_data import CandleData, CryptoSymbol
from .feature_engineering import FeatureSet, extract_features
from .multi_venue_client import CrossVenueFeatures, get_cross_venue_snapshot
from .poly_whale_client import WhaleConsensus, get_whale_signals

logger = logging.getLogger(__name__)


class EnhancedFeatureSet(BaseModel):
    """
    Extended feature set with multi-venue and whale signals.
    
    Combines:
    - Technical features (from base FeatureSet)
    - Cross-venue features (arb, depth, consensus)
    - Whale features (Polymarket trader signals)
    """
    symbol: CryptoSymbol
    timestamp: str
    
    # === Base Technical Features ===
    price_change_1: float
    price_change_3: float
    price_change_6: float
    price_change_12: float
    volatility_6: float
    volatility_12: float
    atr_14: float
    rsi_14: float
    rsi_6: float
    macd_signal: float
    macd_histogram: float
    volume_ratio_6: float
    volume_trend: float
    bb_position: float
    price_vs_sma_20: float
    price_vs_ema_12: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    current_price: float
    
    # === Cross-Venue Features ===
    venue_price_std: float  # Price variance across venues (manipulation signal)
    venue_arb_spread_bps: float  # Max arbitrage opportunity
    venue_depth_imbalance: float  # Aggregate bid/ask imbalance
    venue_consensus: float  # -1 to 1, venue agreement on direction
    venue_slippage_10k_bps: float  # Estimated slippage for $10k trade
    venue_count: int  # Number of venues with data
    
    # === Whale Features ===
    whale_consensus: float  # -1 (bearish) to +1 (bullish)
    whale_volume_consensus: float  # Volume-weighted consensus
    whale_participation: float  # % of tracked whales active
    whale_trade_velocity: float  # Trades per hour
    whale_top10_consensus: float  # Top 10 whales' consensus
    whale_bullish_count: int
    whale_bearish_count: int
    
    # === Combined Signals ===
    signal_alignment: float  # Agreement between TA, venue, whale signals
    clean_score: float  # 0-1, data quality score
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array([
            # Technical
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
            # Venue
            self.venue_price_std,
            self.venue_arb_spread_bps,
            self.venue_depth_imbalance,
            self.venue_consensus,
            self.venue_slippage_10k_bps,
            # Whale
            self.whale_consensus,
            self.whale_volume_consensus,
            self.whale_participation,
            self.whale_trade_velocity,
            self.whale_top10_consensus,
            # Combined
            self.signal_alignment,
            self.clean_score,
        ])
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get feature names for model."""
        return [
            # Technical (19)
            "price_change_1", "price_change_3", "price_change_6", "price_change_12",
            "volatility_6", "volatility_12", "atr_14",
            "rsi_14", "rsi_6", "macd_signal", "macd_histogram",
            "volume_ratio_6", "volume_trend",
            "bb_position", "price_vs_sma_20", "price_vs_ema_12",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            # Venue (5)
            "venue_price_std", "venue_arb_spread_bps", "venue_depth_imbalance",
            "venue_consensus", "venue_slippage_10k_bps",
            # Whale (5)
            "whale_consensus", "whale_volume_consensus", "whale_participation",
            "whale_trade_velocity", "whale_top10_consensus",
            # Combined (2)
            "signal_alignment", "clean_score",
        ]
    
    @classmethod
    def from_components(
        cls,
        base: FeatureSet,
        venue: CrossVenueFeatures | None,
        whale: WhaleConsensus | None,
    ) -> "EnhancedFeatureSet":
        """
        Combine base features with venue and whale signals.
        
        Args:
            base: Base technical features
            venue: Cross-venue features (optional)
            whale: Whale consensus features (optional)
            
        Returns:
            EnhancedFeatureSet with all features
        """
        # Default venue values
        if venue:
            venue_price_std = venue.price_std
            venue_arb_spread = venue.max_arb_spread_bps
            venue_imbalance = venue.aggregate_imbalance
            venue_cons = venue.venue_consensus
            venue_slippage = venue.avg_slippage_10k_bps
            venue_count = venue.bullish_venues + venue.bearish_venues
        else:
            venue_price_std = 0.0
            venue_arb_spread = 0.0
            venue_imbalance = 1.0
            venue_cons = 0.0
            venue_slippage = 10.0  # Default slippage
            venue_count = 0
        
        # Default whale values
        if whale:
            whale_cons = whale.consensus_score
            whale_vol_cons = whale.volume_weighted_score
            whale_part = whale.participation_rate
            whale_vel = whale.trade_velocity
            whale_top10 = whale.top_10_consensus
            whale_bull = whale.bullish_count
            whale_bear = whale.bearish_count
        else:
            whale_cons = 0.0
            whale_vol_cons = 0.0
            whale_part = 0.0
            whale_vel = 0.0
            whale_top10 = 0.0
            whale_bull = 0
            whale_bear = 0
        
        # Compute signal alignment
        # Technical signal: based on RSI, MACD, BB position
        ta_signal = 0.0
        if base.rsi_14 < 30:
            ta_signal += 0.5  # Oversold = bullish
        elif base.rsi_14 > 70:
            ta_signal -= 0.5  # Overbought = bearish
        
        if base.macd_signal > 0:
            ta_signal += 0.3
        elif base.macd_signal < 0:
            ta_signal -= 0.3
        
        if base.bb_position < -0.5:
            ta_signal += 0.2  # Near lower band = bullish
        elif base.bb_position > 0.5:
            ta_signal -= 0.2  # Near upper band = bearish
        
        ta_signal = np.clip(ta_signal, -1, 1)
        
        # Signal alignment: how well do TA, venue, and whale agree?
        signals = [ta_signal]
        if venue:
            signals.append(venue_cons)
        if whale:
            signals.append(whale_cons)
        
        if len(signals) > 1:
            # Compute average correlation
            signal_mean = np.mean(signals)
            signal_alignment = float(np.sign(signal_mean) * min(abs(signal_mean), 1))
        else:
            signal_alignment = ta_signal
        
        # Clean score: data quality indicator
        # Higher if we have more data sources agreeing
        clean_score = 0.5  # Base score
        if venue and venue_count >= 3:
            clean_score += 0.2  # Multiple venues
        if whale and whale_part > 0.3:
            clean_score += 0.2  # Good whale participation
        if abs(signal_alignment) > 0.5:
            clean_score += 0.1  # Strong agreement
        clean_score = min(clean_score, 1.0)
        
        return cls(
            symbol=base.symbol,  # type: ignore
            timestamp=base.timestamp,
            
            # Technical
            price_change_1=base.price_change_1,
            price_change_3=base.price_change_3,
            price_change_6=base.price_change_6,
            price_change_12=base.price_change_12,
            volatility_6=base.volatility_6,
            volatility_12=base.volatility_12,
            atr_14=base.atr_14,
            rsi_14=base.rsi_14,
            rsi_6=base.rsi_6,
            macd_signal=base.macd_signal,
            macd_histogram=base.macd_histogram,
            volume_ratio_6=base.volume_ratio_6,
            volume_trend=base.volume_trend,
            bb_position=base.bb_position,
            price_vs_sma_20=base.price_vs_sma_20,
            price_vs_ema_12=base.price_vs_ema_12,
            body_ratio=base.body_ratio,
            upper_wick_ratio=base.upper_wick_ratio,
            lower_wick_ratio=base.lower_wick_ratio,
            current_price=base.current_price,
            
            # Venue
            venue_price_std=venue_price_std,
            venue_arb_spread_bps=venue_arb_spread,
            venue_depth_imbalance=venue_imbalance,
            venue_consensus=venue_cons,
            venue_slippage_10k_bps=venue_slippage,
            venue_count=venue_count,
            
            # Whale
            whale_consensus=whale_cons,
            whale_volume_consensus=whale_vol_cons,
            whale_participation=whale_part,
            whale_trade_velocity=whale_vel,
            whale_top10_consensus=whale_top10,
            whale_bullish_count=whale_bull,
            whale_bearish_count=whale_bear,
            
            # Combined
            signal_alignment=signal_alignment,
            clean_score=clean_score,
        )


async def extract_enhanced_features(
    candle_data: CandleData,
    include_venue: bool = True,
    include_whale: bool = True,
    whale_hours_back: int = 6,
) -> EnhancedFeatureSet:
    """
    Extract enhanced features including venue and whale signals.
    
    Args:
        candle_data: OHLCV candle data
        include_venue: Whether to fetch cross-venue data
        include_whale: Whether to fetch whale consensus
        whale_hours_back: Time window for whale analysis
        
    Returns:
        EnhancedFeatureSet with all features
    """
    # Base technical features
    base = extract_features(candle_data)
    symbol = candle_data.symbol
    
    # Helper for None coroutine
    async def return_none():
        return None
    
    # Fetch venue and whale data in parallel
    tasks = []
    
    if include_venue:
        tasks.append(get_cross_venue_snapshot(symbol))
    else:
        tasks.append(return_none())
    
    if include_whale:
        tasks.append(get_whale_signals(symbol, whale_hours_back))
    else:
        tasks.append(return_none())
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    venue = results[0] if not isinstance(results[0], Exception) else None
    whale = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
    
    return EnhancedFeatureSet.from_components(base, venue, whale)


async def extract_enhanced_features_batch(
    candle_data: dict[CryptoSymbol, CandleData],
    include_venue: bool = True,
    include_whale: bool = True,
) -> dict[CryptoSymbol, EnhancedFeatureSet]:
    """
    Extract enhanced features for multiple symbols.
    
    Args:
        candle_data: Dict of symbol -> candle data
        include_venue: Whether to fetch cross-venue data
        include_whale: Whether to fetch whale consensus
        
    Returns:
        Dict of symbol -> EnhancedFeatureSet
    """
    tasks = {
        symbol: extract_enhanced_features(data, include_venue, include_whale)
        for symbol, data in candle_data.items()
    }
    
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    features = {}
    for symbol, result in zip(tasks.keys(), results):
        if isinstance(result, EnhancedFeatureSet):
            features[symbol] = result
        else:
            logger.warning(f"Failed to extract features for {symbol}: {result}")
    
    return features


class FeatureQualityFilter:
    """
    Filter for trade signals based on feature quality.
    
    Only trade when:
    - clean_score > threshold (good data quality)
    - whale_consensus aligns with signal
    - venue_consensus supports direction
    """
    
    def __init__(
        self,
        min_clean_score: float = 0.7,
        min_whale_alignment: float = 0.3,
        min_venue_agreement: int = 2,
    ):
        """
        Initialize quality filter.
        
        Args:
            min_clean_score: Minimum clean_score to trade
            min_whale_alignment: Minimum whale consensus alignment
            min_venue_agreement: Minimum venues agreeing on direction
        """
        self.min_clean_score = min_clean_score
        self.min_whale_alignment = min_whale_alignment
        self.min_venue_agreement = min_venue_agreement
    
    def should_trade(
        self,
        features: EnhancedFeatureSet,
        predicted_direction: str,  # "UP" or "DOWN"
    ) -> tuple[bool, list[str]]:
        """
        Check if trade should be taken based on feature quality.
        
        Args:
            features: Enhanced features
            predicted_direction: ML predicted direction
            
        Returns:
            (should_trade, list of reasons if not trading)
        """
        reasons = []
        
        # Check clean score
        if features.clean_score < self.min_clean_score:
            reasons.append(f"Low clean_score: {features.clean_score:.2f} < {self.min_clean_score}")
        
        # Check whale alignment
        expected_whale = 1.0 if predicted_direction == "UP" else -1.0
        whale_align = features.whale_consensus * expected_whale
        
        if whale_align < -self.min_whale_alignment:
            reasons.append(f"Whale consensus opposes: {features.whale_consensus:.2f}")
        
        # Check venue consensus
        expected_venue = 1.0 if predicted_direction == "UP" else -1.0
        venue_align = features.venue_consensus * expected_venue
        
        if venue_align < -0.3 and features.venue_count >= self.min_venue_agreement:
            reasons.append(f"Venue consensus opposes: {features.venue_consensus:.2f}")
        
        # Check signal alignment
        if abs(features.signal_alignment) < 0.2:
            reasons.append(f"Weak signal alignment: {features.signal_alignment:.2f}")
        
        should_trade = len(reasons) == 0
        return should_trade, reasons
    
    def get_confidence_adjustment(
        self,
        features: EnhancedFeatureSet,
        base_confidence: float,
    ) -> float:
        """
        Adjust confidence based on feature quality.
        
        Args:
            features: Enhanced features
            base_confidence: Original ML confidence
            
        Returns:
            Adjusted confidence (may be higher or lower)
        """
        adjustment = 0.0
        
        # Boost for strong whale consensus
        if abs(features.whale_consensus) > 0.6:
            adjustment += 0.05 * np.sign(features.whale_consensus)
        
        # Boost for strong venue consensus
        if abs(features.venue_consensus) > 0.5 and features.venue_count >= 3:
            adjustment += 0.03 * np.sign(features.venue_consensus)
        
        # Boost for high clean score
        if features.clean_score > 0.8:
            adjustment += 0.02
        
        # Penalty for low participation
        if features.whale_participation < 0.2:
            adjustment -= 0.02
        
        # Apply adjustment
        adjusted = base_confidence + adjustment
        return float(np.clip(adjusted, 0.5, 0.95))
