"""
Enhanced Grok Interface with Whale and Venue Context
=====================================================

Extends the base Grok provider with:
1. Whale consensus summary in prompts
2. Cross-venue data for validation
3. Output whale_align and clean_score fields
4. Enhanced regime classification

Target: Better confidence calibration through additional context.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from .enhanced_features import EnhancedFeatureSet
from .multi_venue_client import CrossVenueFeatures
from .poly_whale_client import WhaleConsensus

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


class EnhancedGrokValidation(BaseModel):
    """Enhanced Grok validation response with whale/venue context."""
    symbol: str
    ml_direction: str
    ml_confidence: float
    
    # Grok's analysis
    grok_agrees: bool
    grok_confidence: float
    adjusted_confidence: float
    reasoning: str
    market_context: str
    risk_factors: list[str]
    final_recommendation: str  # "TRADE", "SKIP", "REVERSE"
    
    # Enhanced fields
    whale_align: float  # -1 to 1: how well does whale consensus align?
    venue_align: float  # -1 to 1: how well does venue consensus align?
    clean_score: float  # 0 to 1: overall data quality
    regime: str  # "trending", "ranging", "volatile", "news_driven"
    regime_confidence: float
    
    # Timing
    latency_ms: int


class EnhancedGrokProvider:
    """
    Enhanced Grok provider with whale and venue context.
    
    Takes ML predictions along with whale consensus and cross-venue data,
    then validates with Grok for final confidence calibration.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "grok-3-fast",
        timeout: float = 30.0,
    ):
        """
        Initialize enhanced Grok provider.
        
        Args:
            api_key: xAI API key
            model: Grok model to use (grok-3-fast recommended)
            timeout: HTTP timeout
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1"
        self._client: httpx.AsyncClient | None = None
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(5)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _build_whale_summary(self, whale: WhaleConsensus | None) -> str:
        """Build whale consensus summary for prompt."""
        if not whale:
            return "No whale data available."
        
        direction = "BULLISH" if whale.consensus_score > 0.2 else (
            "BEARISH" if whale.consensus_score < -0.2 else "NEUTRAL"
        )
        
        return f"""POLYMARKET WHALE SIGNALS:
- Consensus: {direction} ({whale.consensus_score:+.2f})
- Top 10 Whales: {whale.top_10_consensus:+.2f}
- Volume-Weighted: {whale.volume_weighted_score:+.2f}
- Active Whales: {whale.bullish_count} bullish, {whale.bearish_count} bearish
- Participation: {whale.participation_rate:.0%}
- Trade Velocity: {whale.trade_velocity:.1f}/hour"""
    
    def _build_venue_summary(self, venue: CrossVenueFeatures | None) -> str:
        """Build cross-venue summary for prompt."""
        if not venue:
            return "No cross-venue data available."
        
        direction = "BULLISH" if venue.venue_consensus > 0.2 else (
            "BEARISH" if venue.venue_consensus < -0.2 else "NEUTRAL"
        )
        
        arb_status = "YES" if venue.arb_opportunity else "NO"
        
        return f"""CROSS-VENUE ANALYSIS:
- Price Consensus: ${venue.avg_mid_price:,.2f} (±${venue.price_std:.2f})
- Depth Imbalance: {venue.aggregate_imbalance:.2f} ({direction})
- Arbitrage Opportunity: {arb_status} ({venue.max_arb_spread_bps:.1f} bps)
- Est. Slippage ($10k): {venue.avg_slippage_10k_bps:.1f} bps
- Venues: {venue.bullish_venues} bullish, {venue.bearish_venues} bearish"""
    
    async def validate_prediction(
        self,
        symbol: CryptoSymbol,
        ml_direction: str,
        ml_confidence: float,
        current_price: float,
        features: EnhancedFeatureSet | dict[str, Any],
        whale: WhaleConsensus | None = None,
        venue: CrossVenueFeatures | None = None,
        recent_price_action: str = "",
    ) -> EnhancedGrokValidation:
        """
        Validate ML prediction with enhanced context.
        
        Args:
            symbol: Crypto symbol
            ml_direction: "UP", "DOWN", or "NEUTRAL"
            ml_confidence: ML model's confidence
            current_price: Current price
            features: Enhanced feature set
            whale: Whale consensus data
            venue: Cross-venue data
            recent_price_action: Description of recent action
            
        Returns:
            EnhancedGrokValidation with full analysis
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        start_time = time.time()
        
        # Build feature summary
        if isinstance(features, EnhancedFeatureSet):
            feat_dict = {
                "rsi_14": features.rsi_14,
                "macd_signal": features.macd_signal,
                "bb_position": features.bb_position,
                "volume_ratio_6": features.volume_ratio_6,
                "price_change_3": features.price_change_3,
                "signal_alignment": features.signal_alignment,
                "clean_score": features.clean_score,
            }
        else:
            feat_dict = features
        
        whale_summary = self._build_whale_summary(whale)
        venue_summary = self._build_venue_summary(venue)
        
        system_prompt = """You are an expert crypto trader assistant analyzing 15-minute predictions.

You have access to:
1. ML model prediction with technical indicators
2. Polymarket whale consensus (top 25 traders' positions)
3. Cross-venue data (arb spreads, depth imbalance across 4 CEXs)

Evaluate alignment between ML, whales, and venue signals. Consider:
- Do whales confirm the ML direction?
- Is there unusual arbitrage or depth imbalance?
- What's the current market regime?
- Any news or events affecting this?

Output ONLY valid JSON:
{
    "agrees_with_ml": true/false,
    "grok_confidence": 0.XX,
    "adjusted_confidence": 0.XX,
    "reasoning": "Brief explanation",
    "market_context": "Current conditions",
    "risk_factors": ["factor1", "factor2"],
    "recommendation": "TRADE" | "SKIP" | "REVERSE",
    "whale_align": X.XX,  // -1 to 1: whale agreement with ML
    "venue_align": X.XX,  // -1 to 1: venue agreement with ML
    "clean_score": 0.XX,  // 0 to 1: data quality
    "regime": "trending" | "ranging" | "volatile" | "news_driven",
    "regime_confidence": 0.XX
}

TRADE = confident, take position
SKIP = uncertain, avoid
REVERSE = ML is likely wrong"""

        user_prompt = f"""SYMBOL: {symbol}
CURRENT PRICE: ${current_price:,.2f}
ML PREDICTION: {ml_direction} ({ml_confidence:.1%} confidence)

TECHNICAL INDICATORS:
- RSI: {feat_dict.get('rsi_14', 'N/A')}
- MACD Signal: {feat_dict.get('macd_signal', 'N/A')}
- Bollinger Position: {feat_dict.get('bb_position', 'N/A')}
- Volume Ratio: {feat_dict.get('volume_ratio_6', 'N/A')}
- Signal Alignment: {feat_dict.get('signal_alignment', 'N/A')}

{whale_summary}

{venue_summary}

{f"RECENT ACTION: {recent_price_action}" if recent_price_action else ""}

Validate this 15-minute prediction with all available context."""

        async with self._semaphore:
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 600
                    }
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                if resp.status_code != 200:
                    logger.error(f"Grok API error {resp.status_code}: {resp.text}")
                    return self._fallback_validation(
                        symbol, ml_direction, ml_confidence, latency_ms,
                        whale, venue, f"API error: {resp.status_code}"
                    )
                
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = self._parse_response(content)
                
                return EnhancedGrokValidation(
                    symbol=symbol,
                    ml_direction=ml_direction,
                    ml_confidence=ml_confidence,
                    grok_agrees=parsed.get("agrees_with_ml", True),
                    grok_confidence=parsed.get("grok_confidence", ml_confidence),
                    adjusted_confidence=parsed.get("adjusted_confidence", ml_confidence),
                    reasoning=parsed.get("reasoning", ""),
                    market_context=parsed.get("market_context", ""),
                    risk_factors=parsed.get("risk_factors", []),
                    final_recommendation=parsed.get("recommendation", "SKIP"),
                    whale_align=parsed.get("whale_align", 0.0),
                    venue_align=parsed.get("venue_align", 0.0),
                    clean_score=parsed.get("clean_score", 0.5),
                    regime=parsed.get("regime", "ranging"),
                    regime_confidence=parsed.get("regime_confidence", 0.5),
                    latency_ms=latency_ms,
                )
                
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                logger.error(f"Grok validation error: {e}")
                return self._fallback_validation(
                    symbol, ml_direction, ml_confidence, latency_ms,
                    whale, venue, str(e)
                )
    
    def _fallback_validation(
        self,
        symbol: str,
        ml_direction: str,
        ml_confidence: float,
        latency_ms: int,
        whale: WhaleConsensus | None,
        venue: CrossVenueFeatures | None,
        error: str,
    ) -> EnhancedGrokValidation:
        """Return fallback validation when API fails."""
        # Compute alignments from raw data
        whale_align = 0.0
        if whale:
            expected = 1.0 if ml_direction == "UP" else -1.0
            whale_align = whale.consensus_score * expected
        
        venue_align = 0.0
        if venue:
            expected = 1.0 if ml_direction == "UP" else -1.0
            venue_align = venue.venue_consensus * expected
        
        clean_score = 0.5
        if whale and whale.participation_rate > 0.3:
            clean_score += 0.2
        if venue and venue.bullish_venues + venue.bearish_venues >= 3:
            clean_score += 0.2
        
        return EnhancedGrokValidation(
            symbol=symbol,
            ml_direction=ml_direction,
            ml_confidence=ml_confidence,
            grok_agrees=True,
            grok_confidence=ml_confidence * 0.8,
            adjusted_confidence=ml_confidence * 0.8,
            reasoning=f"Fallback: {error}",
            market_context="Unable to fetch context",
            risk_factors=["API unavailable", "Using ML + data alignment only"],
            final_recommendation="SKIP" if ml_confidence < 0.6 else "TRADE",
            whale_align=whale_align,
            venue_align=venue_align,
            clean_score=min(clean_score, 1.0),
            regime="ranging",
            regime_confidence=0.3,
            latency_ms=latency_ms,
        )
    
    def _parse_response(self, content: str) -> dict:
        """Parse JSON from Grok response."""
        # Extract JSON from code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            newline = content.find("\n", start)
            if newline != -1:
                start = newline + 1
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON, using regex extraction")
            return self._regex_parse(content)
    
    def _regex_parse(self, content: str) -> dict:
        """Fallback regex parsing for malformed JSON."""
        result = {
            "agrees_with_ml": True,
            "grok_confidence": 0.5,
            "adjusted_confidence": 0.5,
            "reasoning": "Failed to parse",
            "recommendation": "SKIP",
            "whale_align": 0.0,
            "venue_align": 0.0,
            "clean_score": 0.5,
            "regime": "ranging",
            "regime_confidence": 0.5,
        }
        
        # Extract booleans
        if re.search(r'"agrees_with_ml"\s*:\s*false', content, re.I):
            result["agrees_with_ml"] = False
        
        # Extract floats
        for field in ["grok_confidence", "adjusted_confidence", "whale_align", 
                      "venue_align", "clean_score", "regime_confidence"]:
            match = re.search(rf'"{field}"\s*:\s*(-?[\d.]+)', content)
            if match:
                try:
                    result[field] = float(match.group(1))
                except ValueError:
                    pass
        
        # Extract recommendation
        if "TRADE" in content:
            result["recommendation"] = "TRADE"
        elif "REVERSE" in content:
            result["recommendation"] = "REVERSE"
        
        # Extract regime
        for regime in ["trending", "ranging", "volatile", "news_driven"]:
            if regime in content.lower():
                result["regime"] = regime
                break
        
        # Extract reasoning
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', content)
        if reason_match:
            result["reasoning"] = reason_match.group(1)
        
        return result
    
    async def batch_validate(
        self,
        predictions: list[dict],
    ) -> list[EnhancedGrokValidation]:
        """Validate multiple predictions in parallel."""
        tasks = [
            self.validate_prediction(**pred)
            for pred in predictions
        ]
        return await asyncio.gather(*tasks)


def create_enhanced_grok_provider(api_key: str) -> EnhancedGrokProvider:
    """Factory function for enhanced Grok provider."""
    return EnhancedGrokProvider(api_key)
