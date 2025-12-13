"""
Grok 4.1 Fast provider for high-confidence validation.
Uses xAI's grok-3-fast model for quick, smart confidence checks.
"""

import asyncio
import json
import logging
import time
import re
from datetime import datetime

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GrokValidation(BaseModel):
    """Grok's validation response."""
    symbol: str
    ml_direction: str
    ml_confidence: float
    grok_agrees: bool
    grok_confidence: float
    adjusted_confidence: float
    reasoning: str
    market_context: str
    risk_factors: list[str]
    final_recommendation: str  # "TRADE", "SKIP", "REVERSE"
    latency_ms: int


class Grok41FastProvider:
    """
    Grok 4.1 Fast (grok-3-fast) provider for final confidence validation.
    
    Takes ML predictions and validates them with real-time context,
    news awareness, and market sentiment analysis.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "grok-3-fast",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1"
        self.client = httpx.AsyncClient(timeout=timeout)
        self._semaphore = asyncio.Semaphore(5)  # Allow 5 concurrent requests
        
    async def validate_prediction(
        self,
        symbol: str,
        ml_direction: str,
        ml_confidence: float,
        current_price: float,
        features: dict,
        recent_price_action: str = ""
    ) -> GrokValidation:
        """
        Validate ML prediction with Grok's analysis.
        
        Args:
            symbol: BTC, ETH, SOL
            ml_direction: "UP", "DOWN", or "NEUTRAL"
            ml_confidence: ML model's confidence (0.5-1.0)
            current_price: Current asset price
            features: Key features from ML model
            recent_price_action: Description of recent price movement
            
        Returns:
            GrokValidation with adjusted confidence and recommendation
        """
        start_time = time.time()
        
        system_prompt = """You are an expert crypto trader assistant. Your job is to validate ML predictions for 15-minute crypto price direction.

You have access to real-time market context and news. Be concise but thorough.

Evaluate the ML prediction considering:
1. Current market conditions and sentiment
2. Recent news that could affect price
3. Technical indicator alignment
4. Risk of false signals
5. Timeframe appropriateness (15 min is short-term)

Output ONLY valid JSON:
{
    "agrees_with_ml": true/false,
    "grok_confidence": 0.XX,
    "adjusted_confidence": 0.XX,
    "reasoning": "Brief explanation",
    "market_context": "Current market mood",
    "risk_factors": ["factor1", "factor2"],
    "recommendation": "TRADE" | "SKIP" | "REVERSE"
}

TRADE = agree with ML, confidence is high enough
SKIP = uncertain, don't trade this signal
REVERSE = ML is wrong, opposite direction likely"""

        user_prompt = f"""SYMBOL: {symbol}
CURRENT PRICE: ${current_price:,.2f}
ML PREDICTION: {ml_direction} ({ml_confidence:.1%} confidence)

KEY INDICATORS:
- RSI: {features.get('rsi_14', 'N/A')}
- MACD Signal: {features.get('macd_signal', 'N/A')}
- Bollinger Position: {features.get('bb_position', 'N/A')}
- Volume Ratio: {features.get('volume_ratio_6', 'N/A')}
- 3-candle change: {features.get('price_change_3', 'N/A'):.2%}

{f"RECENT ACTION: {recent_price_action}" if recent_price_action else ""}

Validate this 15-minute prediction. Consider any breaking news or market events."""

        async with self._semaphore:
            try:
                resp = await self.client.post(
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
                        "max_tokens": 500
                    }
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                if resp.status_code != 200:
                    error_text = resp.text
                    logger.error(f"Grok API error {resp.status_code}: {error_text}")
                    return self._fallback_validation(
                        symbol, ml_direction, ml_confidence, latency_ms,
                        f"API error: {resp.status_code}"
                    )
                
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                
                parsed = self._parse_response(content)
                
                return GrokValidation(
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
                    latency_ms=latency_ms
                )
                
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                logger.error(f"Grok validation error: {e}")
                return self._fallback_validation(
                    symbol, ml_direction, ml_confidence, latency_ms,
                    str(e)
                )
    
    def _fallback_validation(
        self,
        symbol: str,
        ml_direction: str,
        ml_confidence: float,
        latency_ms: int,
        error: str
    ) -> GrokValidation:
        """Return conservative fallback when API fails."""
        return GrokValidation(
            symbol=symbol,
            ml_direction=ml_direction,
            ml_confidence=ml_confidence,
            grok_agrees=True,
            grok_confidence=ml_confidence * 0.8,  # Reduce confidence on fallback
            adjusted_confidence=ml_confidence * 0.8,
            reasoning=f"Fallback: {error}",
            market_context="Unable to fetch market context",
            risk_factors=["API unavailable", "Using ML-only prediction"],
            final_recommendation="SKIP" if ml_confidence < 0.6 else "TRADE",
            latency_ms=latency_ms
        )
    
    def _parse_response(self, content: str) -> dict:
        """Parse JSON from Grok response."""
        # Try to extract JSON from code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            # Skip language identifier if present
            newline = content.find("\n", start)
            if newline != -1:
                start = newline + 1
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract individual fields
            logger.debug(f"Failed to parse JSON, trying regex extraction")
            
            result = {
                "agrees_with_ml": True,
                "grok_confidence": 0.5,
                "adjusted_confidence": 0.5,
                "reasoning": "Failed to parse response",
                "recommendation": "SKIP"
            }
            
            # Extract agrees_with_ml
            if "false" in content.lower() and "agrees" in content.lower():
                result["agrees_with_ml"] = False
            
            # Extract confidences
            conf_match = re.search(r'"grok_confidence"\s*:\s*(0\.\d+)', content)
            if conf_match:
                result["grok_confidence"] = float(conf_match.group(1))
            
            adj_match = re.search(r'"adjusted_confidence"\s*:\s*(0\.\d+)', content)
            if adj_match:
                result["adjusted_confidence"] = float(adj_match.group(1))
            
            # Extract recommendation
            if "TRADE" in content:
                result["recommendation"] = "TRADE"
            elif "REVERSE" in content:
                result["recommendation"] = "REVERSE"
            else:
                result["recommendation"] = "SKIP"
            
            # Extract reasoning
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', content)
            if reason_match:
                result["reasoning"] = reason_match.group(1)
            
            return result
    
    async def batch_validate(
        self,
        predictions: list[dict]
    ) -> list[GrokValidation]:
        """Validate multiple predictions in parallel."""
        tasks = [
            self.validate_prediction(**pred)
            for pred in predictions
        ]
        return await asyncio.gather(*tasks)
    
    async def close(self):
        await self.client.aclose()


# Factory function
def create_grok41_provider(api_key: str) -> Grok41FastProvider:
    """Create a Grok 4.1 Fast provider instance."""
    return Grok41FastProvider(api_key)
