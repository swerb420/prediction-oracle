"""
Grok-powered market regime classifier.
Identifies bull/bear/range/volatile regimes for trading adaptation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger
from features.feature_builder_grok import MarketRegime, SentimentLevel
from llm.grok_client import GrokClient, GrokRequest, SYSTEM_PROMPTS

logger = get_logger(__name__)


class RegimeClassification(BaseModel):
    """Result of regime classification."""
    
    timestamp: datetime
    symbol: str
    
    # Primary classification
    regime: MarketRegime
    regime_confidence: float = Field(ge=0.0, le=1.0)
    regime_explanation: str = ""
    
    # Sub-classifications
    trend_direction: str = "neutral"  # up, down, neutral
    trend_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    volatility_level: str = "normal"  # low, normal, high, extreme
    
    # Sentiment
    sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    sentiment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Market conditions
    key_levels: list[float] = Field(default_factory=list)
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    
    # Risk factors
    risk_factors: list[str] = Field(default_factory=list)
    bullish_catalysts: list[str] = Field(default_factory=list)
    bearish_catalysts: list[str] = Field(default_factory=list)
    
    # Trading recommendations
    recommended_position_size: float = Field(default=1.0, ge=0.0, le=1.0)
    avoid_trading: bool = False
    
    # Grok metadata
    tokens_used: int = 0
    latency_ms: float = 0.0
    
    @property
    def is_trending(self) -> bool:
        return self.regime in (MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING)
    
    @property
    def is_volatile(self) -> bool:
        return self.regime == MarketRegime.HIGH_VOLATILITY or self.volatility_level in ("high", "extreme")


class RegimeClassifier:
    """
    Uses Grok to classify market regimes.
    
    Batches requests every 15-30 minutes to minimize API calls.
    Uses technical indicators as context for better accuracy.
    """
    
    PROMPT_TEMPLATE = """Classify the current market regime for {symbol}.

Price Data (last 24 hours):
- Current Price: ${current_price:,.2f}
- 24h High: ${high_24h:,.2f}
- 24h Low: ${low_24h:,.2f}
- 24h Change: {change_24h:+.2f}%
- 24h Volume: ${volume_24h:,.0f}

Technical Indicators:
{indicators_text}

Recent Price Action:
{price_action}

Respond with ONLY this JSON structure:
{{
    "regime": "bull_trending|bear_trending|range_bound|high_volatility|low_volatility|breakout|breakdown",
    "regime_confidence": 0.0-1.0,
    "regime_explanation": "brief explanation",
    "trend_direction": "up|down|neutral",
    "trend_strength": 0.0-1.0,
    "volatility_level": "low|normal|high|extreme",
    "sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
    "sentiment_score": 0.0-1.0,
    "key_levels": [price1, price2],
    "support_levels": [support1, support2],
    "resistance_levels": [resistance1, resistance2],
    "risk_factors": ["factor1", "factor2"],
    "bullish_catalysts": ["catalyst1"],
    "bearish_catalysts": ["catalyst1"],
    "recommended_position_size": 0.0-1.0,
    "avoid_trading": true|false
}}"""

    def __init__(self, client: GrokClient | None = None):
        self.client = client
        self._cache: dict[str, tuple[datetime, RegimeClassification]] = {}
        self._cache_ttl_seconds = 900  # 15 minutes
    
    async def classify(
        self,
        symbol: str,
        price_data: dict[str, float],
        indicators: dict[str, float],
        price_action: str = ""
    ) -> RegimeClassification:
        """
        Classify market regime using Grok.
        
        Args:
            symbol: Asset symbol (BTC, ETH, SOL)
            price_data: Dict with current_price, high_24h, low_24h, etc.
            indicators: Technical indicator values
            price_action: Description of recent price action
            
        Returns:
            RegimeClassification
        """
        from core.time_utils import now_utc
        
        # Check cache
        cached = self._get_cached(symbol)
        if cached is not None:
            return cached
        
        # Format indicators
        indicator_lines = []
        for name, value in indicators.items():
            if isinstance(value, float):
                indicator_lines.append(f"- {name}: {value:.4f}")
            else:
                indicator_lines.append(f"- {name}: {value}")
        indicators_text = "\n".join(indicator_lines) if indicator_lines else "No indicators available"
        
        # Build prompt
        prompt = self.PROMPT_TEMPLATE.format(
            symbol=symbol,
            current_price=price_data.get("current_price", 0),
            high_24h=price_data.get("high_24h", 0),
            low_24h=price_data.get("low_24h", 0),
            change_24h=price_data.get("change_24h", 0),
            volume_24h=price_data.get("volume_24h", 0),
            indicators_text=indicators_text,
            price_action=price_action or "No specific price action noted"
        )
        
        # Ensure client exists
        if self.client is None:
            from llm.grok_client import create_grok_client
            self.client = create_grok_client()
        
        # Make request
        request = GrokRequest(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["regime"],
            max_tokens=600,
            temperature=0.1,
            request_type="regime",
            symbol=symbol
        )
        
        response = await self.client.complete(request)
        
        # Parse response
        if not response.success or response.parsed is None:
            logger.warning(f"Regime classification failed for {symbol}: {response.error or response.parse_error}")
            return self._fallback_classification(symbol, indicators)
        
        # Build classification
        data = response.parsed
        timestamp = now_utc()
        
        try:
            regime = MarketRegime(data.get("regime", "unknown").lower())
        except ValueError:
            regime = MarketRegime.UNKNOWN
        
        try:
            sentiment = SentimentLevel(data.get("sentiment", "neutral").lower())
        except ValueError:
            sentiment = SentimentLevel.NEUTRAL
        
        classification = RegimeClassification(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            regime_confidence=float(data.get("regime_confidence", 0.5)),
            regime_explanation=data.get("regime_explanation", ""),
            trend_direction=data.get("trend_direction", "neutral"),
            trend_strength=float(data.get("trend_strength", 0.5)),
            volatility_level=data.get("volatility_level", "normal"),
            sentiment=sentiment,
            sentiment_score=float(data.get("sentiment_score", 0.5)),
            key_levels=[float(x) for x in data.get("key_levels", []) if x],
            support_levels=[float(x) for x in data.get("support_levels", []) if x],
            resistance_levels=[float(x) for x in data.get("resistance_levels", []) if x],
            risk_factors=data.get("risk_factors", []),
            bullish_catalysts=data.get("bullish_catalysts", []),
            bearish_catalysts=data.get("bearish_catalysts", []),
            recommended_position_size=float(data.get("recommended_position_size", 1.0)),
            avoid_trading=data.get("avoid_trading", False),
            tokens_used=response.total_tokens,
            latency_ms=response.latency_ms
        )
        
        # Cache result
        self._cache[symbol] = (timestamp, classification)
        
        logger.info(
            f"Regime for {symbol}: {regime.value} "
            f"(confidence={classification.regime_confidence:.2f}, "
            f"tokens={response.total_tokens})"
        )
        
        return classification
    
    def _get_cached(self, symbol: str) -> RegimeClassification | None:
        """Get cached classification if still valid."""
        from core.time_utils import now_utc
        
        if symbol not in self._cache:
            return None
        
        cached_time, classification = self._cache[symbol]
        elapsed = (now_utc() - cached_time).total_seconds()
        
        if elapsed > self._cache_ttl_seconds:
            del self._cache[symbol]
            return None
        
        return classification
    
    def _fallback_classification(
        self,
        symbol: str,
        indicators: dict[str, float]
    ) -> RegimeClassification:
        """
        Fallback classification using indicators when Grok fails.
        Less accurate but doesn't require API.
        """
        from core.time_utils import now_utc
        
        # Simple rule-based regime detection
        rsi = indicators.get("rsi_14", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        atr_pct = indicators.get("atr_percent", 0.02)
        bb_pct = indicators.get("bb_percent", 0.5)
        
        # Determine regime
        if atr_pct > 0.04:
            regime = MarketRegime.HIGH_VOLATILITY
            volatility = "high"
        elif atr_pct < 0.01:
            regime = MarketRegime.LOW_VOLATILITY
            volatility = "low"
        elif rsi > 70 and macd_hist > 0:
            regime = MarketRegime.BULL_TRENDING
            volatility = "normal"
        elif rsi < 30 and macd_hist < 0:
            regime = MarketRegime.BEAR_TRENDING
            volatility = "normal"
        elif bb_pct > 1.0:
            regime = MarketRegime.BREAKOUT
            volatility = "high"
        elif bb_pct < 0.0:
            regime = MarketRegime.BREAKDOWN
            volatility = "high"
        else:
            regime = MarketRegime.RANGE_BOUND
            volatility = "normal"
        
        # Determine sentiment from RSI
        if rsi > 65:
            sentiment = SentimentLevel.BULLISH
            sentiment_score = min(1.0, rsi / 100)
        elif rsi < 35:
            sentiment = SentimentLevel.BEARISH
            sentiment_score = max(0.0, rsi / 100)
        else:
            sentiment = SentimentLevel.NEUTRAL
            sentiment_score = rsi / 100
        
        return RegimeClassification(
            timestamp=now_utc(),
            symbol=symbol,
            regime=regime,
            regime_confidence=0.5,  # Lower confidence for fallback
            regime_explanation="Fallback classification from indicators",
            trend_direction="up" if macd_hist > 0 else ("down" if macd_hist < 0 else "neutral"),
            trend_strength=abs(macd_hist) / 100 if macd_hist else 0.5,
            volatility_level=volatility,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            recommended_position_size=0.5,  # Conservative for fallback
            avoid_trading=atr_pct > 0.05  # Avoid in extreme volatility
        )
    
    async def classify_multiple(
        self,
        symbols: list[str],
        price_data: dict[str, dict[str, float]],
        indicators: dict[str, dict[str, float]]
    ) -> dict[str, RegimeClassification]:
        """
        Classify multiple symbols in batch.
        
        Args:
            symbols: List of symbols
            price_data: Price data per symbol
            indicators: Indicators per symbol
            
        Returns:
            Dict of symbol -> classification
        """
        results = {}
        
        for symbol in symbols:
            classification = await self.classify(
                symbol=symbol,
                price_data=price_data.get(symbol, {}),
                indicators=indicators.get(symbol, {})
            )
            results[symbol] = classification
        
        return results


async def classify_regime(
    symbol: str,
    price_data: dict[str, float],
    indicators: dict[str, float]
) -> RegimeClassification:
    """Convenience function for one-off regime classification."""
    classifier = RegimeClassifier()
    return await classifier.classify(symbol, price_data, indicators)
