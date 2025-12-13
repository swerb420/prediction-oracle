"""
15-minute price direction trading policy.
Combines ML predictions with Grok regime analysis for crypto trading.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class SignalStrength(str, Enum):
    """Signal strength classification."""
    
    STRONG_LONG = "strong_long"
    LONG = "long"
    WEAK_LONG = "weak_long"
    NEUTRAL = "neutral"
    WEAK_SHORT = "weak_short"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


class DirectionSignal(BaseModel):
    """Trading signal for 15m direction prediction."""
    
    timestamp: datetime
    symbol: str
    
    # Signal
    direction: str  # "long", "short", "neutral"
    strength: SignalStrength = SignalStrength.NEUTRAL
    
    # Probabilities
    prob_up: float = Field(ge=0.0, le=1.0)
    prob_down: float = Field(ge=0.0, le=1.0)
    
    # Confidence metrics
    ml_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    regime_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    combined_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Position sizing
    recommended_size: float = Field(default=0.0, ge=0.0, le=1.0)  # 0-1 of max position
    kelly_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Regime context
    regime: str = "unknown"
    regime_aligned: bool = True  # Signal aligns with regime
    
    # Risk flags
    high_volatility: bool = False
    avoid_trade: bool = False
    reason: str = ""
    
    # Entry/exit levels
    entry_price: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    
    @property
    def is_tradeable(self) -> bool:
        """Check if signal is actionable."""
        return (
            self.direction != "neutral" and
            not self.avoid_trade and
            self.combined_confidence >= 0.55 and
            self.recommended_size > 0
        )
    
    @property
    def expected_value(self) -> float:
        """Calculate expected value of trade."""
        if self.direction == "long":
            return self.prob_up - 0.5
        elif self.direction == "short":
            return self.prob_down - 0.5
        return 0.0


class DirectionPolicy:
    """
    15-minute direction trading policy.
    
    Combines:
    - ML model predictions (LightGBM)
    - Grok regime classification
    - Technical indicator validation
    - Kelly criterion position sizing
    """
    
    def __init__(
        self,
        ml_weight: float = 0.7,
        regime_weight: float = 0.3,
        min_confidence: float = 0.55,
        max_kelly_fraction: float = 0.25,  # Cap Kelly at 25%
        volatility_adjustment: bool = True
    ):
        self.ml_weight = ml_weight
        self.regime_weight = regime_weight
        self.min_confidence = min_confidence
        self.max_kelly_fraction = max_kelly_fraction
        self.volatility_adjustment = volatility_adjustment
    
    def generate_signal(
        self,
        symbol: str,
        ml_prediction: dict[str, Any],
        regime_classification: dict[str, Any] | None = None,
        current_price: float | None = None,
        indicators: dict[str, float] | None = None
    ) -> DirectionSignal:
        """
        Generate trading signal from ML prediction and regime analysis.
        
        Args:
            symbol: Asset symbol
            ml_prediction: Dict with prob_up, prob_down, confidence
            regime_classification: Optional regime analysis from Grok
            current_price: Current asset price
            indicators: Technical indicators
            
        Returns:
            DirectionSignal
        """
        from core.time_utils import now_utc
        
        timestamp = now_utc()
        
        # Extract ML prediction
        prob_up = ml_prediction.get("prob_up", 0.5)
        prob_down = ml_prediction.get("prob_down", 0.5)
        ml_confidence = ml_prediction.get("confidence", 0.5)
        
        # Extract regime info
        regime = "unknown"
        regime_confidence = 0.5
        regime_sentiment = 0.5
        high_volatility = False
        avoid_trade = False
        avoid_reason = ""
        
        if regime_classification:
            regime = regime_classification.get("regime", "unknown")
            regime_confidence = regime_classification.get("regime_confidence", 0.5)
            regime_sentiment = regime_classification.get("sentiment_score", 0.5)
            high_volatility = regime_classification.get("is_volatile", False)
            avoid_trade = regime_classification.get("avoid_trading", False)
            
            if avoid_trade:
                avoid_reason = "Regime classifier recommends avoiding trades"
        
        # Combine predictions with regime
        combined_prob_up = self._combine_signals(
            prob_up, regime_sentiment,
            self.ml_weight, self.regime_weight
        )
        combined_prob_down = 1 - combined_prob_up
        
        # Combined confidence
        combined_confidence = (
            self.ml_weight * ml_confidence +
            self.regime_weight * regime_confidence
        )
        
        # Check regime alignment
        regime_aligned = self._check_regime_alignment(
            prob_up > 0.5, regime, regime_sentiment
        )
        
        if not regime_aligned:
            combined_confidence *= 0.7  # Reduce confidence if misaligned
        
        # Determine direction
        if combined_prob_up > 0.5 + self.min_confidence / 2:
            direction = "long"
        elif combined_prob_down > 0.5 + self.min_confidence / 2:
            direction = "short"
        else:
            direction = "neutral"
        
        # Determine signal strength
        strength = self._classify_strength(combined_prob_up, combined_confidence)
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly(
            combined_prob_up if direction == "long" else combined_prob_down,
            win_rate=combined_confidence
        )
        
        # Adjust for volatility
        if self.volatility_adjustment and high_volatility:
            kelly_fraction *= 0.5
        
        # Cap Kelly fraction
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        
        # Determine position size
        if combined_confidence >= self.min_confidence and not avoid_trade:
            recommended_size = kelly_fraction
        else:
            recommended_size = 0.0
        
        # Calculate entry/exit levels
        entry_price = current_price
        take_profit = None
        stop_loss = None
        
        if current_price and indicators:
            atr = indicators.get("atr_14", current_price * 0.02)
            
            if direction == "long":
                take_profit = current_price + 2 * atr
                stop_loss = current_price - 1.5 * atr
            elif direction == "short":
                take_profit = current_price - 2 * atr
                stop_loss = current_price + 1.5 * atr
        
        return DirectionSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strength=strength,
            prob_up=combined_prob_up,
            prob_down=combined_prob_down,
            ml_confidence=ml_confidence,
            regime_confidence=regime_confidence,
            combined_confidence=combined_confidence,
            recommended_size=recommended_size,
            kelly_fraction=kelly_fraction,
            regime=regime,
            regime_aligned=regime_aligned,
            high_volatility=high_volatility,
            avoid_trade=avoid_trade,
            reason=avoid_reason,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss
        )
    
    def _combine_signals(
        self,
        ml_prob: float,
        regime_sentiment: float,
        ml_weight: float,
        regime_weight: float
    ) -> float:
        """Combine ML probability with regime sentiment."""
        # Normalize weights
        total_weight = ml_weight + regime_weight
        ml_w = ml_weight / total_weight
        regime_w = regime_weight / total_weight
        
        return ml_w * ml_prob + regime_w * regime_sentiment
    
    def _check_regime_alignment(
        self,
        ml_bullish: bool,
        regime: str,
        sentiment: float
    ) -> bool:
        """Check if ML prediction aligns with regime."""
        bullish_regimes = {"bull_trending", "breakout", "low_volatility"}
        bearish_regimes = {"bear_trending", "breakdown"}
        
        if ml_bullish:
            if regime in bearish_regimes or sentiment < 0.3:
                return False
        else:
            if regime in bullish_regimes or sentiment > 0.7:
                return False
        
        return True
    
    def _classify_strength(
        self,
        prob_up: float,
        confidence: float
    ) -> SignalStrength:
        """Classify signal strength."""
        prob_diff = abs(prob_up - 0.5)
        
        if prob_up > 0.5:
            if prob_diff > 0.25 and confidence > 0.7:
                return SignalStrength.STRONG_LONG
            elif prob_diff > 0.15 and confidence > 0.6:
                return SignalStrength.LONG
            elif prob_diff > 0.05:
                return SignalStrength.WEAK_LONG
        elif prob_up < 0.5:
            if prob_diff > 0.25 and confidence > 0.7:
                return SignalStrength.STRONG_SHORT
            elif prob_diff > 0.15 and confidence > 0.6:
                return SignalStrength.SHORT
            elif prob_diff > 0.05:
                return SignalStrength.WEAK_SHORT
        
        return SignalStrength.NEUTRAL
    
    def _calculate_kelly(
        self,
        win_prob: float,
        win_rate: float,
        odds: float = 1.0  # 1:1 odds
    ) -> float:
        """
        Calculate Kelly criterion fraction.
        
        Kelly = (bp - q) / b
        where b = odds, p = win probability, q = 1 - p
        """
        if win_prob <= 0 or odds <= 0:
            return 0.0
        
        q = 1 - win_prob
        kelly = (odds * win_prob - q) / odds
        
        # Adjust by confidence
        kelly *= win_rate
        
        # Never bet more than win probability suggests
        kelly = max(0, min(kelly, win_prob - 0.5))
        
        return kelly
    
    def evaluate_trade(
        self,
        signal: DirectionSignal,
        actual_direction: str,  # "up" or "down"
        actual_return: float
    ) -> dict[str, Any]:
        """
        Evaluate a completed trade.
        
        Args:
            signal: Original signal
            actual_direction: What actually happened
            actual_return: Actual percentage return
            
        Returns:
            Evaluation dict
        """
        predicted_up = signal.direction == "long"
        actual_up = actual_direction == "up"
        
        correct = predicted_up == actual_up
        
        # Calculate P&L
        if signal.direction == "long":
            pnl = actual_return * signal.recommended_size
        elif signal.direction == "short":
            pnl = -actual_return * signal.recommended_size
        else:
            pnl = 0.0
        
        return {
            "timestamp": signal.timestamp,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "strength": signal.strength.value,
            "confidence": signal.combined_confidence,
            "correct": correct,
            "actual_direction": actual_direction,
            "actual_return": actual_return,
            "position_size": signal.recommended_size,
            "pnl": pnl,
            "regime": signal.regime,
            "regime_aligned": signal.regime_aligned
        }


def create_direction_policy(**kwargs) -> DirectionPolicy:
    """Factory function to create direction policy."""
    return DirectionPolicy(**kwargs)
