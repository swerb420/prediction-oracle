"""
Grok-enhanced feature builder for regime detection and sentiment analysis.
Uses Grok 4.1 for regime classification, sentiment scoring, and semantic clustering.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification from Grok."""
    
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    UNKNOWN = "unknown"


class SentimentLevel(str, Enum):
    """Sentiment classification."""
    
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class GrokFeatures(BaseModel):
    """Features derived from Grok 4.1 analysis."""
    
    timestamp: datetime
    symbol: str
    
    # Regime detection
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    regime_duration_periods: int = 0
    regime_stability: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Sentiment scoring
    sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    sentiment_score: float = Field(default=0.5, ge=0.0, le=1.0)  # 0=bearish, 1=bullish
    news_sentiment: float = Field(default=0.5, ge=0.0, le=1.0)
    social_sentiment: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Market context
    key_events: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)
    
    # Probability adjustments from Grok
    direction_up_adjustment: float = Field(default=0.0, ge=-0.5, le=0.5)
    confidence_adjustment: float = Field(default=0.0, ge=-0.5, le=0.5)
    
    # Grok metadata
    grok_response_tokens: int = 0
    grok_latency_ms: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dict for ML model."""
        # Encode regime as one-hot
        regime_encoding = {
            f"regime_{r.value}": 1.0 if self.regime == r else 0.0
            for r in MarketRegime
            if r != MarketRegime.UNKNOWN
        }
        
        # Encode sentiment as numeric
        sentiment_map = {
            SentimentLevel.VERY_BEARISH: 0.0,
            SentimentLevel.BEARISH: 0.25,
            SentimentLevel.NEUTRAL: 0.5,
            SentimentLevel.BULLISH: 0.75,
            SentimentLevel.VERY_BULLISH: 1.0,
        }
        
        return {
            **regime_encoding,
            "regime_confidence": self.regime_confidence,
            "regime_duration_periods": float(self.regime_duration_periods),
            "regime_stability": self.regime_stability,
            "sentiment_numeric": sentiment_map.get(self.sentiment, 0.5),
            "sentiment_score": self.sentiment_score,
            "news_sentiment": self.news_sentiment,
            "social_sentiment": self.social_sentiment,
            "direction_up_adjustment": self.direction_up_adjustment,
            "confidence_adjustment": self.confidence_adjustment,
            "key_events_count": float(len(self.key_events)),
            "risk_factors_count": float(len(self.risk_factors)),
            "bullish_factors_count": float(len(self.bullish_factors)),
            "bearish_factors_count": float(len(self.bearish_factors)),
        }


class GrokMarketCluster(BaseModel):
    """Semantic market clustering from Grok."""
    
    cluster_id: str
    cluster_name: str
    market_ids: list[str]
    central_theme: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GrokFeatureBuilder:
    """
    Builds features from Grok 4.1 analysis.
    
    Uses Grok for:
    1. Regime detection (bull/bear/range/volatile)
    2. Sentiment scoring (news + social)
    3. Risk factor identification
    4. Market clustering for Polymarket
    """
    
    REGIME_PROMPT_TEMPLATE = """Analyze the current market regime for {symbol} based on:

Price Data (last 24h):
{price_summary}

Technical Indicators:
{indicators_summary}

Respond with ONLY valid JSON:
{{
    "regime": "bull_trending|bear_trending|range_bound|high_volatility|low_volatility|breakout|breakdown",
    "regime_confidence": 0.0-1.0,
    "regime_stability": 0.0-1.0,
    "sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
    "sentiment_score": 0.0-1.0,
    "direction_up_adjustment": -0.5 to 0.5,
    "key_events": ["event1", "event2"],
    "risk_factors": ["risk1"],
    "bullish_factors": ["factor1"],
    "bearish_factors": ["factor1"]
}}"""

    SENTIMENT_PROMPT_TEMPLATE = """Score the current market sentiment for {symbol}.

Recent Context:
{context}

Rate sentiment from 0.0 (extremely bearish) to 1.0 (extremely bullish).

Respond with ONLY valid JSON:
{{
    "sentiment_score": 0.0-1.0,
    "news_sentiment": 0.0-1.0,
    "social_sentiment": 0.0-1.0,
    "key_events": ["event1"],
    "explanation": "brief explanation"
}}"""

    CLUSTER_PROMPT_TEMPLATE = """Group these Polymarket questions into semantic clusters of related/duplicate markets:

Markets:
{markets_list}

Respond with ONLY valid JSON:
{{
    "clusters": [
        {{
            "cluster_id": "cluster_1",
            "cluster_name": "name describing theme",
            "market_ids": ["id1", "id2"],
            "central_theme": "what these markets have in common",
            "confidence": 0.0-1.0
        }}
    ]
}}"""

    def __init__(self):
        self._cached_regimes: dict[str, tuple[datetime, GrokFeatures]] = {}
        self._cache_ttl_seconds: int = 900  # 15 minutes
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get list of feature names."""
        regime_features = [f"regime_{r.value}" for r in MarketRegime if r != MarketRegime.UNKNOWN]
        
        return [
            *regime_features,
            "regime_confidence", "regime_duration_periods", "regime_stability",
            "sentiment_numeric", "sentiment_score", "news_sentiment", "social_sentiment",
            "direction_up_adjustment", "confidence_adjustment",
            "key_events_count", "risk_factors_count",
            "bullish_factors_count", "bearish_factors_count",
        ]
    
    def build_regime_prompt(
        self,
        symbol: str,
        price_data: dict[str, Any],
        indicators: dict[str, float]
    ) -> str:
        """Build regime detection prompt for Grok."""
        # Format price summary
        price_lines = []
        if "open" in price_data:
            price_lines.append(f"Open: ${price_data['open']:,.2f}")
        if "high" in price_data:
            price_lines.append(f"High: ${price_data['high']:,.2f}")
        if "low" in price_data:
            price_lines.append(f"Low: ${price_data['low']:,.2f}")
        if "close" in price_data:
            price_lines.append(f"Close: ${price_data['close']:,.2f}")
        if "volume" in price_data:
            price_lines.append(f"Volume: ${price_data['volume']:,.0f}")
        if "change_pct" in price_data:
            price_lines.append(f"24h Change: {price_data['change_pct']:+.2f}%")
        
        price_summary = "\n".join(price_lines) if price_lines else "No price data available"
        
        # Format indicators
        indicator_lines = []
        for key, value in indicators.items():
            if isinstance(value, float):
                indicator_lines.append(f"{key}: {value:.4f}")
            else:
                indicator_lines.append(f"{key}: {value}")
        
        indicators_summary = "\n".join(indicator_lines) if indicator_lines else "No indicators available"
        
        return self.REGIME_PROMPT_TEMPLATE.format(
            symbol=symbol,
            price_summary=price_summary,
            indicators_summary=indicators_summary
        )
    
    def build_sentiment_prompt(
        self,
        symbol: str,
        context: str
    ) -> str:
        """Build sentiment scoring prompt for Grok."""
        return self.SENTIMENT_PROMPT_TEMPLATE.format(
            symbol=symbol,
            context=context[:800]  # Limit context length
        )
    
    def build_cluster_prompt(
        self,
        markets: list[tuple[str, str]]  # (market_id, question) pairs
    ) -> str:
        """Build market clustering prompt for Grok."""
        markets_list = "\n".join(
            f"- {market_id}: {question}"
            for market_id, question in markets[:30]  # Limit to 30 markets
        )
        
        return self.CLUSTER_PROMPT_TEMPLATE.format(markets_list=markets_list)
    
    def parse_regime_response(
        self,
        symbol: str,
        response: dict[str, Any],
        latency_ms: float = 0.0,
        tokens: int = 0
    ) -> GrokFeatures:
        """Parse Grok regime response into features."""
        from core.time_utils import now_utc
        
        timestamp = now_utc()
        
        # Parse regime
        regime_str = response.get("regime", "unknown").lower()
        try:
            regime = MarketRegime(regime_str)
        except ValueError:
            regime = MarketRegime.UNKNOWN
            logger.warning(f"Unknown regime: {regime_str}")
        
        # Parse sentiment
        sentiment_str = response.get("sentiment", "neutral").lower()
        try:
            sentiment = SentimentLevel(sentiment_str)
        except ValueError:
            sentiment = SentimentLevel.NEUTRAL
            logger.warning(f"Unknown sentiment: {sentiment_str}")
        
        return GrokFeatures(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            regime_confidence=float(response.get("regime_confidence", 0.5)),
            regime_stability=float(response.get("regime_stability", 0.5)),
            sentiment=sentiment,
            sentiment_score=float(response.get("sentiment_score", 0.5)),
            news_sentiment=float(response.get("news_sentiment", 0.5)),
            social_sentiment=float(response.get("social_sentiment", 0.5)),
            direction_up_adjustment=float(response.get("direction_up_adjustment", 0.0)),
            key_events=response.get("key_events", []),
            risk_factors=response.get("risk_factors", []),
            bullish_factors=response.get("bullish_factors", []),
            bearish_factors=response.get("bearish_factors", []),
            grok_response_tokens=tokens,
            grok_latency_ms=latency_ms,
        )
    
    def parse_sentiment_response(
        self,
        symbol: str,
        response: dict[str, Any],
        base_features: GrokFeatures | None = None
    ) -> GrokFeatures:
        """Parse Grok sentiment response, optionally updating existing features."""
        from core.time_utils import now_utc
        
        if base_features:
            features = base_features.model_copy()
        else:
            features = GrokFeatures(
                timestamp=now_utc(),
                symbol=symbol,
            )
        
        # Update sentiment fields
        features.sentiment_score = float(response.get("sentiment_score", 0.5))
        features.news_sentiment = float(response.get("news_sentiment", 0.5))
        features.social_sentiment = float(response.get("social_sentiment", 0.5))
        
        # Derive sentiment level from score
        score = features.sentiment_score
        if score >= 0.8:
            features.sentiment = SentimentLevel.VERY_BULLISH
        elif score >= 0.6:
            features.sentiment = SentimentLevel.BULLISH
        elif score >= 0.4:
            features.sentiment = SentimentLevel.NEUTRAL
        elif score >= 0.2:
            features.sentiment = SentimentLevel.BEARISH
        else:
            features.sentiment = SentimentLevel.VERY_BEARISH
        
        # Add any new events
        new_events = response.get("key_events", [])
        features.key_events = list(set(features.key_events + new_events))
        
        return features
    
    def parse_cluster_response(
        self,
        response: dict[str, Any]
    ) -> list[GrokMarketCluster]:
        """Parse Grok clustering response."""
        clusters = []
        
        for cluster_data in response.get("clusters", []):
            try:
                cluster = GrokMarketCluster(
                    cluster_id=cluster_data.get("cluster_id", "unknown"),
                    cluster_name=cluster_data.get("cluster_name", "Unknown"),
                    market_ids=cluster_data.get("market_ids", []),
                    central_theme=cluster_data.get("central_theme", ""),
                    confidence=float(cluster_data.get("confidence", 0.5))
                )
                clusters.append(cluster)
            except Exception as e:
                logger.warning(f"Failed to parse cluster: {e}")
                continue
        
        return clusters
    
    def get_cached_features(
        self,
        symbol: str
    ) -> GrokFeatures | None:
        """Get cached features if still valid."""
        from core.time_utils import now_utc
        
        if symbol not in self._cached_regimes:
            return None
        
        cached_time, features = self._cached_regimes[symbol]
        elapsed = (now_utc() - cached_time).total_seconds()
        
        if elapsed > self._cache_ttl_seconds:
            del self._cached_regimes[symbol]
            return None
        
        return features
    
    def cache_features(
        self,
        symbol: str,
        features: GrokFeatures
    ) -> None:
        """Cache features for later use."""
        from core.time_utils import now_utc
        self._cached_regimes[symbol] = (now_utc(), features)
    
    def build_default_features(self, symbol: str) -> GrokFeatures:
        """Build default features when Grok is unavailable."""
        from core.time_utils import now_utc
        
        return GrokFeatures(
            timestamp=now_utc(),
            symbol=symbol,
        )
    
    def estimate_regime_from_indicators(
        self,
        symbol: str,
        indicators: dict[str, float]
    ) -> GrokFeatures:
        """
        Fallback: estimate regime from technical indicators without Grok.
        Less accurate but doesn't require API call.
        """
        from core.time_utils import now_utc
        
        features = GrokFeatures(
            timestamp=now_utc(),
            symbol=symbol,
        )
        
        # Simple rule-based regime detection
        rsi = indicators.get("rsi_14", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        bb_pct = indicators.get("bb_percent", 0.5)
        atr_pct = indicators.get("atr_percent", 0.02)
        
        # Volatility regime
        if atr_pct > 0.04:
            features.regime = MarketRegime.HIGH_VOLATILITY
            features.regime_confidence = 0.6
        elif atr_pct < 0.01:
            features.regime = MarketRegime.LOW_VOLATILITY
            features.regime_confidence = 0.6
        # Trend regime
        elif rsi > 70 and macd_hist > 0:
            features.regime = MarketRegime.BULL_TRENDING
            features.regime_confidence = 0.7
        elif rsi < 30 and macd_hist < 0:
            features.regime = MarketRegime.BEAR_TRENDING
            features.regime_confidence = 0.7
        # Breakout/Breakdown
        elif bb_pct > 1.0 and macd_hist > 0:
            features.regime = MarketRegime.BREAKOUT
            features.regime_confidence = 0.6
        elif bb_pct < 0.0 and macd_hist < 0:
            features.regime = MarketRegime.BREAKDOWN
            features.regime_confidence = 0.6
        else:
            features.regime = MarketRegime.RANGE_BOUND
            features.regime_confidence = 0.5
        
        # Simple sentiment from RSI
        if rsi > 65:
            features.sentiment = SentimentLevel.BULLISH
            features.sentiment_score = min(1.0, rsi / 100)
        elif rsi < 35:
            features.sentiment = SentimentLevel.BEARISH
            features.sentiment_score = max(0.0, rsi / 100)
        else:
            features.sentiment = SentimentLevel.NEUTRAL
            features.sentiment_score = rsi / 100
        
        return features


def build_grok_features(
    symbol: str,
    price_data: dict[str, Any] | None = None,
    indicators: dict[str, float] | None = None
) -> GrokFeatures:
    """
    Convenience function to build Grok features.
    Falls back to indicator-based estimation if no Grok response.
    """
    builder = GrokFeatureBuilder()
    
    if indicators:
        return builder.estimate_regime_from_indicators(symbol, indicators)
    
    return builder.build_default_features(symbol)
