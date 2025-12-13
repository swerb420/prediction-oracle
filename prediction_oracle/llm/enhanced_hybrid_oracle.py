"""
Enhanced Hybrid Oracle with Whale and Venue Signals
====================================================

The ultimate prediction system combining:
1. Enhanced ML features (31 dimensions)
2. Multi-venue price/depth data
3. Polymarket whale consensus
4. Grok 4.1 Fast validation with full context

Flow:
1. Fetch candle data + venue prices + whale trades
2. Extract enhanced features
3. ML prediction with quality gate
4. Grok validation with whale/venue context
5. Final signal with confidence calibration

Target: 63%+ win rate with high confidence trades.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from .crypto_data import CandleData, CryptoSymbol, get_fetcher
from .enhanced_features import EnhancedFeatureSet, extract_enhanced_features
from .enhanced_grok_provider import (
    EnhancedGrokProvider,
    EnhancedGrokValidation,
    create_enhanced_grok_provider,
)
from .enhanced_ml_predictor import (
    EnhancedCryptoMLPredictor,
    EnhancedMLPrediction,
    get_enhanced_predictor,
)
from .multi_venue_client import CrossVenueFeatures, get_cross_venue_snapshot
from .poly_whale_client import WhaleConsensus, get_whale_signals

logger = logging.getLogger(__name__)

TradeSignal = Literal["LONG", "SHORT", "HOLD"]


class EnhancedHybridPrediction(BaseModel):
    """
    Complete prediction output with all signals.
    """
    symbol: CryptoSymbol
    timestamp: datetime
    
    # ML prediction
    ml_direction: str
    ml_confidence: float
    ml_probability_up: float
    ml_quality_gate_passed: bool
    
    # Enhanced signals
    whale_consensus: float
    venue_consensus: float
    signal_alignment: float
    clean_score: float
    
    # Grok validation
    grok_agrees: bool
    grok_confidence: float
    grok_reasoning: str
    grok_recommendation: str
    grok_regime: str
    grok_whale_align: float
    grok_venue_align: float
    
    # Final decision
    final_signal: TradeSignal
    final_confidence: float
    should_trade: bool
    
    # Context
    current_price: float
    risk_factors: list[str]
    trade_rationale: str


class EnhancedHybridOracle:
    """
    Enhanced Hybrid Oracle combining ML + Whale + Venue + Grok.
    
    Entry conditions for high-confidence trades:
    1. ML confidence > threshold
    2. Quality gate passed (clean_score > 0.7, whale aligns)
    3. Grok agrees or provides strong alternative
    4. Final confidence > trade threshold
    
    Exit conditions:
    - Signal reversal
    - Time-based (N candles)
    - Stop loss/take profit
    """
    
    def __init__(
        self,
        xai_api_key: str | None = None,
        ml_confidence_threshold: float = 0.55,
        trade_confidence_threshold: float = 0.60,
        min_clean_score: float = 0.6,
        use_grok_validation: bool = True,
        use_whale_signals: bool = True,
        use_venue_data: bool = True,
    ):
        """
        Initialize enhanced hybrid oracle.
        
        Args:
            xai_api_key: xAI API key (or XAI_API_KEY env var)
            ml_confidence_threshold: Min ML confidence to consider
            trade_confidence_threshold: Min final confidence to trade
            min_clean_score: Min data quality score
            use_grok_validation: Enable Grok validation
            use_whale_signals: Enable whale tracking
            use_venue_data: Enable multi-venue data
        """
        self.xai_api_key = xai_api_key or os.getenv("XAI_API_KEY")
        self.ml_threshold = ml_confidence_threshold
        self.trade_threshold = trade_confidence_threshold
        self.min_clean_score = min_clean_score
        self.use_grok = use_grok_validation
        self.use_whale = use_whale_signals
        self.use_venue = use_venue_data
        
        self.ml_predictor: EnhancedCryptoMLPredictor | None = None
        self.grok_provider: EnhancedGrokProvider | None = None
        self._initialized = False
    
    async def initialize(self, retrain_ml: bool = False):
        """Initialize all components."""
        logger.info("Initializing Enhanced Hybrid Oracle...")
        
        # ML predictor
        self.ml_predictor = get_enhanced_predictor()
        await self.ml_predictor.initialize(retrain=retrain_ml)
        
        # Grok provider (context manager used per-call)
        if self.use_grok and self.xai_api_key:
            logger.info("Grok 4.1 validation enabled with whale/venue context")
        else:
            logger.warning("Grok validation disabled")
            self.use_grok = False
        
        self._initialized = True
        logger.info("Enhanced Hybrid Oracle initialized")
    
    async def predict(self, symbol: CryptoSymbol) -> EnhancedHybridPrediction:
        """
        Get enhanced hybrid prediction for a symbol.
        
        Full flow:
        1. Fetch all data in parallel
        2. ML prediction with quality gate
        3. Grok validation if enabled
        4. Combine for final signal
        """
        if not self._initialized:
            await self.initialize()
        
        # Helper for None coroutine
        async def return_none():
            return None
        
        # Step 1: Fetch data
        fetcher = get_fetcher()
        candle_task = fetcher.get_candles(symbol, "15m", 50)
        
        # Fetch whale and venue in parallel
        tasks = [candle_task]
        
        if self.use_whale:
            tasks.append(get_whale_signals(symbol, hours_back=6))
        else:
            tasks.append(return_none())
        
        if self.use_venue:
            tasks.append(get_cross_venue_snapshot(symbol))
        else:
            tasks.append(return_none())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        candle_data = results[0] if not isinstance(results[0], Exception) else None
        whale = results[1] if not isinstance(results[1], Exception) else None
        venue = results[2] if not isinstance(results[2], Exception) else None
        
        if not candle_data:
            raise RuntimeError(f"Failed to fetch candle data for {symbol}")
        
        current_price = candle_data.candles[-1].close
        
        # Step 2: Extract enhanced features
        features = await extract_enhanced_features(
            candle_data,
            include_venue=self.use_venue,
            include_whale=self.use_whale,
        )
        
        # Step 3: ML prediction
        ml_pred = self.ml_predictor.predictor.predict(features)
        
        # Step 4: Grok validation (if enabled and worth it)
        if (
            self.use_grok
            and self.xai_api_key
            and ml_pred.confidence >= self.ml_threshold
            and ml_pred.clean_score >= self.min_clean_score * 0.8  # Slightly relaxed
        ):
            async with EnhancedGrokProvider(self.xai_api_key) as grok:
                grok_result = await grok.validate_prediction(
                    symbol=symbol,
                    ml_direction=ml_pred.direction,
                    ml_confidence=ml_pred.confidence,
                    current_price=current_price,
                    features=features,
                    whale=whale,
                    venue=venue,
                    recent_price_action=self._describe_price_action(candle_data),
                )
        else:
            # Skip Grok, use ML only
            grok_result = EnhancedGrokValidation(
                symbol=symbol,
                ml_direction=ml_pred.direction,
                ml_confidence=ml_pred.confidence,
                grok_agrees=True,
                grok_confidence=ml_pred.confidence,
                adjusted_confidence=ml_pred.confidence,
                reasoning="Grok validation skipped",
                market_context="",
                risk_factors=[],
                final_recommendation="TRADE" if ml_pred.confidence >= self.trade_threshold else "SKIP",
                whale_align=features.whale_consensus if ml_pred.direction == "UP" else -features.whale_consensus,
                venue_align=features.venue_consensus if ml_pred.direction == "UP" else -features.venue_consensus,
                clean_score=features.clean_score,
                regime="ranging",
                regime_confidence=0.5,
                latency_ms=0,
            )
        
        # Step 5: Combine signals for final decision
        final_signal, final_confidence, should_trade, rationale = self._combine_signals(
            ml_pred, grok_result, features
        )
        
        return EnhancedHybridPrediction(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            
            ml_direction=ml_pred.direction,
            ml_confidence=ml_pred.confidence,
            ml_probability_up=ml_pred.probability_up,
            ml_quality_gate_passed=ml_pred.quality_gate_passed,
            
            whale_consensus=features.whale_consensus,
            venue_consensus=features.venue_consensus,
            signal_alignment=features.signal_alignment,
            clean_score=features.clean_score,
            
            grok_agrees=grok_result.grok_agrees,
            grok_confidence=grok_result.grok_confidence,
            grok_reasoning=grok_result.reasoning,
            grok_recommendation=grok_result.final_recommendation,
            grok_regime=grok_result.regime,
            grok_whale_align=grok_result.whale_align,
            grok_venue_align=grok_result.venue_align,
            
            final_signal=final_signal,
            final_confidence=final_confidence,
            should_trade=should_trade,
            
            current_price=current_price,
            risk_factors=grok_result.risk_factors,
            trade_rationale=rationale,
        )
    
    async def predict_all(self) -> dict[CryptoSymbol, EnhancedHybridPrediction]:
        """Get predictions for all symbols."""
        symbols: list[CryptoSymbol] = ["BTC", "ETH", "SOL", "XRP"]
        
        tasks = [self.predict(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        predictions = {}
        for sym, result in zip(symbols, results):
            if isinstance(result, EnhancedHybridPrediction):
                predictions[sym] = result
            else:
                logger.error(f"Failed to predict {sym}: {result}")
        
        return predictions
    
    def _describe_price_action(self, candle_data: CandleData) -> str:
        """Generate price action description for Grok."""
        candles = candle_data.candles[-5:]
        if len(candles) < 5:
            return ""
        
        changes = []
        for i in range(1, len(candles)):
            pct = (candles[i].close / candles[i-1].close - 1) * 100
            changes.append(f"{pct:+.2f}%")
        
        current = candles[-1]
        high_low_range = (current.high - current.low) / current.close * 100
        
        return f"Last 4 candles: {', '.join(changes)}. Current candle range: {high_low_range:.2f}%"
    
    def _combine_signals(
        self,
        ml_pred: EnhancedMLPrediction,
        grok_result: EnhancedGrokValidation,
        features: EnhancedFeatureSet,
    ) -> tuple[TradeSignal, float, bool, str]:
        """
        Combine ML + Grok + Whale + Venue into final signal.
        
        Returns: (signal, confidence, should_trade, rationale)
        """
        rationale_parts = []
        
        # Base signal from ML
        if ml_pred.direction == "UP":
            base_signal: TradeSignal = "LONG"
        elif ml_pred.direction == "DOWN":
            base_signal = "SHORT"
        else:
            base_signal = "HOLD"
        
        # Start with ML confidence
        confidence = ml_pred.confidence
        
        # Grok adjustment
        if grok_result.grok_agrees:
            # Boost confidence
            confidence = (confidence + grok_result.adjusted_confidence) / 2
            rationale_parts.append("Grok confirms")
        else:
            # Reduce confidence
            confidence = confidence * 0.8
            rationale_parts.append(f"Grok disagrees: {grok_result.reasoning[:50]}")
            
            # Check if Grok suggests reversal
            if grok_result.final_recommendation == "REVERSE":
                base_signal = "SHORT" if base_signal == "LONG" else "LONG"
                confidence = grok_result.grok_confidence
                rationale_parts.append("Signal reversed per Grok")
        
        # Whale consensus boost/penalty
        expected_whale = 1.0 if base_signal == "LONG" else -1.0
        whale_align = features.whale_consensus * expected_whale
        
        if whale_align > 0.5:
            confidence += 0.03
            rationale_parts.append(f"Whales align ({features.whale_consensus:+.2f})")
        elif whale_align < -0.3:
            confidence -= 0.05
            rationale_parts.append(f"Whales oppose ({features.whale_consensus:+.2f})")
        
        # Venue consensus
        expected_venue = 1.0 if base_signal == "LONG" else -1.0
        venue_align = features.venue_consensus * expected_venue
        
        if venue_align > 0.3 and features.venue_count >= 3:
            confidence += 0.02
            rationale_parts.append("Venues align")
        elif venue_align < -0.3 and features.venue_count >= 3:
            confidence -= 0.03
            rationale_parts.append("Venues oppose")
        
        # Clean score adjustment
        if features.clean_score < self.min_clean_score:
            confidence = min(confidence, 0.55)
            rationale_parts.append(f"Low data quality ({features.clean_score:.2f})")
        
        # Clamp confidence
        confidence = float(max(0.5, min(0.95, confidence)))
        
        # Determine if we should trade
        should_trade = (
            confidence >= self.trade_threshold
            and ml_pred.quality_gate_passed
            and grok_result.final_recommendation != "SKIP"
            and base_signal != "HOLD"
        )
        
        if not should_trade and base_signal != "HOLD":
            base_signal = "HOLD"
            if confidence < self.trade_threshold:
                rationale_parts.append(f"Confidence too low ({confidence:.1%})")
            if not ml_pred.quality_gate_passed:
                rationale_parts.append("Quality gate failed")
            if grok_result.final_recommendation == "SKIP":
                rationale_parts.append("Grok recommends skip")
        
        rationale = " | ".join(rationale_parts) if rationale_parts else "Standard entry"
        
        return base_signal, confidence, should_trade, rationale


# Factory and singleton
_enhanced_oracle: EnhancedHybridOracle | None = None


def get_enhanced_oracle(
    xai_api_key: str | None = None,
) -> EnhancedHybridOracle:
    """Get or create enhanced oracle instance."""
    global _enhanced_oracle
    if _enhanced_oracle is None:
        _enhanced_oracle = EnhancedHybridOracle(xai_api_key=xai_api_key)
    return _enhanced_oracle


async def predict_with_full_context(
    symbol: CryptoSymbol | None = None,
) -> dict[CryptoSymbol, EnhancedHybridPrediction] | EnhancedHybridPrediction:
    """
    Convenience function for enhanced hybrid predictions.
    
    Args:
        symbol: Specific symbol or None for all
    """
    oracle = get_enhanced_oracle()
    
    if symbol:
        return await oracle.predict(symbol)
    else:
        return await oracle.predict_all()
