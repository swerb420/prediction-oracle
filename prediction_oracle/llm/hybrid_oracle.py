"""
Hybrid ML + LLM Oracle for crypto price prediction.
Combines ML predictions with Grok 4.1 Fast validation for high win rate.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from .crypto_data import CryptoSymbol, fetch_crypto_data
from .feature_engineering import extract_features, FeatureSet
from .ml_predictor import CryptoMLPredictor, MLPrediction, get_predictor
from .grok41_provider import Grok41FastProvider, GrokValidation, create_grok41_provider

logger = logging.getLogger(__name__)

TradeSignal = Literal["LONG", "SHORT", "HOLD"]


class HybridPrediction(BaseModel):
    """Combined ML + LLM prediction."""
    symbol: CryptoSymbol
    timestamp: datetime
    
    # ML prediction
    ml_direction: str
    ml_confidence: float
    ml_probability_up: float
    
    # Grok validation
    grok_agrees: bool
    grok_confidence: float
    grok_reasoning: str
    grok_recommendation: str
    
    # Final decision
    final_signal: TradeSignal
    final_confidence: float
    should_trade: bool
    
    # Context
    current_price: float
    risk_factors: list[str]
    features_summary: dict[str, float]


class HybridCryptoOracle:
    """
    Hybrid Oracle combining ML predictions with LLM validation.
    
    Flow:
    1. Fetch latest 15-min candle data
    2. Extract ML features (RSI, MACD, etc.)
    3. Get ML prediction (direction + confidence)
    4. Validate with Grok 4.1 Fast
    5. Combine signals for final decision
    
    Only trades when:
    - ML confidence > threshold
    - Grok agrees or provides higher confidence
    - Final combined confidence > trade threshold
    """
    
    def __init__(
        self,
        xai_api_key: str | None = None,
        ml_confidence_threshold: float = 0.55,
        trade_confidence_threshold: float = 0.60,
        use_grok_validation: bool = True
    ):
        """
        Initialize the Hybrid Oracle.
        
        Args:
            xai_api_key: xAI API key for Grok. Defaults to XAI_API_KEY env var.
            ml_confidence_threshold: Minimum ML confidence to consider
            trade_confidence_threshold: Minimum final confidence to trade
            use_grok_validation: Whether to use Grok for validation
        """
        self.xai_api_key = xai_api_key or os.getenv("XAI_API_KEY")
        self.ml_threshold = ml_confidence_threshold
        self.trade_threshold = trade_confidence_threshold
        self.use_grok = use_grok_validation
        
        self.ml_predictor: CryptoMLPredictor | None = None
        self.grok_provider: Grok41FastProvider | None = None
        self._initialized = False
    
    async def initialize(self, retrain_ml: bool = False):
        """Initialize ML models and Grok provider."""
        logger.info("Initializing Hybrid Crypto Oracle...")
        
        # Initialize ML predictor
        self.ml_predictor = get_predictor()
        await self.ml_predictor.initialize(retrain=retrain_ml)
        
        # Initialize Grok provider
        if self.use_grok and self.xai_api_key:
            self.grok_provider = create_grok41_provider(self.xai_api_key)
            logger.info("Grok 4.1 Fast validation enabled")
        else:
            logger.warning("Grok validation disabled (no API key)")
            self.use_grok = False
        
        self._initialized = True
        logger.info("Hybrid Oracle initialized successfully")
    
    async def predict(self, symbol: CryptoSymbol) -> HybridPrediction:
        """
        Get hybrid prediction for a single symbol.
        
        Args:
            symbol: BTC, ETH, or SOL
            
        Returns:
            HybridPrediction with final trading signal
        """
        if not self._initialized:
            await self.initialize()
        
        # Step 1: Get ML prediction
        ml_pred = await self.ml_predictor.predict_symbol(symbol)
        
        # Step 2: Get current data for context
        from .crypto_data import get_fetcher
        fetcher = get_fetcher()
        candle_data = await fetcher.get_candles(symbol, "15m", 5)
        current_price = candle_data.candles[-1].close
        
        # Step 3: Prepare features for Grok
        features = extract_features(candle_data) if len(candle_data.candles) >= 5 else None
        features_dict = ml_pred.features_summary
        
        # Add more context if we have full features
        if features:
            features_dict.update({
                "rsi_14": features.rsi_14,
                "macd_signal": features.macd_signal,
                "bb_position": features.bb_position,
                "volume_ratio_6": features.volume_ratio_6,
                "price_change_3": features.price_change_3,
            })
        
        # Step 4: Grok validation (if enabled and ML confidence meets threshold)
        if self.use_grok and self.grok_provider and ml_pred.confidence >= self.ml_threshold:
            # Build price action description
            price_action = self._describe_price_action(candle_data)
            
            grok_result = await self.grok_provider.validate_prediction(
                symbol=symbol,
                ml_direction=ml_pred.direction,
                ml_confidence=ml_pred.confidence,
                current_price=current_price,
                features=features_dict,
                recent_price_action=price_action
            )
        else:
            # Skip Grok, use ML only
            grok_result = GrokValidation(
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
                latency_ms=0
            )
        
        # Step 5: Combine signals for final decision
        final_signal, final_confidence, should_trade = self._combine_signals(
            ml_pred, grok_result
        )
        
        return HybridPrediction(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            
            ml_direction=ml_pred.direction,
            ml_confidence=ml_pred.confidence,
            ml_probability_up=ml_pred.probability_up,
            
            grok_agrees=grok_result.grok_agrees,
            grok_confidence=grok_result.grok_confidence,
            grok_reasoning=grok_result.reasoning,
            grok_recommendation=grok_result.final_recommendation,
            
            final_signal=final_signal,
            final_confidence=final_confidence,
            should_trade=should_trade,
            
            current_price=current_price,
            risk_factors=grok_result.risk_factors,
            features_summary=features_dict
        )
    
    async def predict_all(self) -> dict[CryptoSymbol, HybridPrediction]:
        """Get predictions for all supported symbols."""
        if not self._initialized:
            await self.initialize()
        
        symbols: list[CryptoSymbol] = ["BTC", "ETH", "SOL", "XRP"]
        
        # Run predictions in parallel
        tasks = [self.predict(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        predictions = {}
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to predict {sym}: {result}")
            else:
                predictions[sym] = result
        
        return predictions
    
    def _describe_price_action(self, candle_data) -> str:
        """Generate a brief description of recent price action."""
        if len(candle_data.candles) < 3:
            return ""
        
        candles = candle_data.candles[-3:]
        changes = []
        for i in range(1, len(candles)):
            pct = (candles[i].close / candles[i-1].close - 1) * 100
            direction = "up" if pct > 0 else "down"
            changes.append(f"{direction} {abs(pct):.2f}%")
        
        return f"Last 3 candles: {', '.join(changes)}"
    
    def _combine_signals(
        self,
        ml_pred: MLPrediction,
        grok_result: GrokValidation
    ) -> tuple[TradeSignal, float, bool]:
        """
        Combine ML and Grok signals into final trading decision.
        
        Returns:
            (signal, confidence, should_trade)
        """
        # If Grok says REVERSE, go opposite
        if grok_result.final_recommendation == "REVERSE":
            if ml_pred.direction == "UP":
                signal = "SHORT"
            elif ml_pred.direction == "DOWN":
                signal = "LONG"
            else:
                signal = "HOLD"
            final_confidence = grok_result.adjusted_confidence
            should_trade = final_confidence >= self.trade_threshold
            
        # If Grok says SKIP, don't trade
        elif grok_result.final_recommendation == "SKIP":
            signal = "HOLD"
            final_confidence = min(ml_pred.confidence, grok_result.grok_confidence)
            should_trade = False
            
        # Grok agrees - combine confidences
        else:
            if ml_pred.direction == "UP":
                signal = "LONG"
            elif ml_pred.direction == "DOWN":
                signal = "SHORT"
            else:
                signal = "HOLD"
            
            # Weighted average of confidences (Grok gets more weight when it agrees)
            if grok_result.grok_agrees:
                final_confidence = (
                    ml_pred.confidence * 0.4 + 
                    grok_result.adjusted_confidence * 0.6
                )
            else:
                # Grok disagrees but still says TRADE - reduce confidence
                final_confidence = min(
                    ml_pred.confidence,
                    grok_result.adjusted_confidence
                ) * 0.9
            
            should_trade = final_confidence >= self.trade_threshold and signal != "HOLD"
        
        return signal, final_confidence, should_trade
    
    async def close(self):
        """Cleanup resources."""
        if self.grok_provider:
            await self.grok_provider.close()


# Singleton instance
_oracle: HybridCryptoOracle | None = None


def get_oracle(xai_api_key: str | None = None) -> HybridCryptoOracle:
    """Get or create global oracle instance."""
    global _oracle
    if _oracle is None:
        _oracle = HybridCryptoOracle(xai_api_key=xai_api_key)
    return _oracle


async def get_trading_signals(
    xai_api_key: str | None = None
) -> dict[CryptoSymbol, HybridPrediction]:
    """
    Convenience function to get trading signals for all symbols.
    
    Args:
        xai_api_key: Optional xAI API key
        
    Returns:
        Dict mapping symbols to their hybrid predictions
    """
    oracle = get_oracle(xai_api_key)
    return await oracle.predict_all()
