"""
Grok Multi-Tier Provider - Fast analysis with smart deep search.

Models (by cost/capability):
- FAST: grok-4-1-fast-reasoning ($0.20/M) - Quick analysis every few seconds
- DEEP: grok-4-0709 ($3.00/M) - Deep search for unusual situations ONLY

Strategy:
- Use FAST model for rapid market scanning (every 2-5 seconds)
- Use DEEP model only for extreme situations (rare, expensive)
- Designed for catching entry opportunities quickly
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal
from dataclasses import dataclass

import httpx

from real_data_store import get_store, RealDataStore

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]

# XAI API Configuration
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
if not XAI_API_KEY:
    logger.warning("XAI_API_KEY not set - Grok features will be disabled")
XAI_BASE_URL = "https://api.x.ai/v1"

# Model tiers
GROK_FAST = "grok-4-1-fast-reasoning"  # $0.20/M - Quick analysis
GROK_DEEP = "grok-4-0709"               # $3.00/M - Deep search (rare)

# Cost estimates per 1K tokens
COST_FAST = 0.0002   # $0.20 per million
COST_DEEP = 0.003    # $3.00 per million


@dataclass
class GrokResponse:
    """Parsed response from Grok."""
    direction: str  # "UP" or "DOWN"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    key_factors: list[str]
    action: str  # "BUY", "SELL", "WAIT"
    urgency: str  # "immediate", "soon", "wait"
    model_used: str
    cost_estimate: float
    raw_response: str = ""


class GrokProvider:
    """
    Tiered Grok provider for rapid market analysis.
    
    FAST model: Called frequently for quick assessment
    DEEP model: Called rarely for unusual situations
    """
    
    # Rate limits
    MAX_FAST_PER_MINUTE = 60    # Fast model - very permissive
    MAX_DEEP_PER_HOUR = 5        # Deep model - expensive, use sparingly
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        store: Optional[RealDataStore] = None,
    ):
        self.api_key = api_key or XAI_API_KEY
        self.store = store or get_store()
        self.client: Optional[httpx.AsyncClient] = None
        
        # Rate limiting
        self.fast_calls: list[datetime] = []
        self.deep_calls: list[datetime] = []
        
        # Track last analysis per symbol (avoid redundant calls)
        self.last_analysis: dict[str, tuple[datetime, GrokResponse]] = {}
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=15.0)  # Fast timeout
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def _ensure_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=15.0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Rate Limiting
    # ─────────────────────────────────────────────────────────────────────────
    
    def _can_call_fast(self) -> bool:
        """Check if we can make a fast model call."""
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        self.fast_calls = [t for t in self.fast_calls if t > minute_ago]
        return len(self.fast_calls) < self.MAX_FAST_PER_MINUTE
    
    def _can_call_deep(self) -> bool:
        """Check if we can make a deep model call."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        self.deep_calls = [t for t in self.deep_calls if t > hour_ago]
        return len(self.deep_calls) < self.MAX_DEEP_PER_HOUR
    
    def _record_call(self, model: str):
        """Record a call for rate limiting."""
        now = datetime.now(timezone.utc)
        if model == GROK_FAST:
            self.fast_calls.append(now)
        else:
            self.deep_calls.append(now)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Deep Model Triggers (expensive - be selective)
    # ─────────────────────────────────────────────────────────────────────────
    
    def needs_deep_analysis(
        self,
        symbol: str,
        ml_confidence: float,
        ml_direction: str,
        market_direction: str,
        market_confidence: float,
        volume_ratio: float = 1.0,  # Current volume / average volume
    ) -> tuple[bool, str]:
        """
        Determine if we need expensive deep analysis.
        
        Only triggers for truly unusual situations.
        Returns (needs_deep, reason).
        """
        # Extreme divergence: ML very confident one way, market very confident other way
        if (ml_confidence >= 0.75 and market_confidence >= 0.65 
            and ml_direction != market_direction):
            return True, "extreme_divergence"
        
        # Unusual volume (5x+ normal)
        if volume_ratio >= 5.0:
            return True, "volume_spike"
        
        # Otherwise, fast model is sufficient
        return False, ""
    
    # ─────────────────────────────────────────────────────────────────────────
    # ENHANCED PROMPTS - Better structured for higher accuracy
    # ─────────────────────────────────────────────────────────────────────────
    
    def _build_fast_prompt(
        self,
        symbol: str,
        yes_price: float,
        ml_direction: str,
        ml_confidence: float,
        orderbook_data: dict = None,
        momentum_data: dict = None,
    ) -> str:
        """Build a structured prompt for fast analysis with chain-of-thought."""
        
        # Format orderbook info if available
        ob_info = ""
        if orderbook_data:
            imbalance = orderbook_data.get('imbalance', 0)
            ob_direction = "bullish (more buyers)" if imbalance > 0 else "bearish (more sellers)"
            ob_info = f"\n- Orderbook: {ob_direction}, imbalance={imbalance:+.2f} (-1=heavy sell, +1=heavy buy)"
        
        # Format momentum if available  
        mom_info = ""
        if momentum_data:
            change_5m = momentum_data.get('change_5min_pct', 0)
            change_1h = momentum_data.get('change_1h_pct', 0)
            mom_info = f"\n- Momentum: 5min={change_5m:+.2f}%, 1hr={change_1h:+.2f}%"
        
        # Wall Street hours context
        ws_info = ""
        try:
            import pytz
            from datetime import datetime
            est = pytz.timezone('America/New_York')
            now_est = datetime.now(est)
            if now_est.weekday() < 5:  # Weekday
                hour = now_est.hour
                if 9 <= hour < 11:
                    ws_info = "\n- ⚠️ WALL STREET HOURS: Market open (9:30-11 AM EST) typically sees institutional crypto selling. LEAN BEARISH."
                elif 11 <= hour < 16:
                    ws_info = "\n- 📊 US TRADING HOURS: Wall Street active. Watch for crypto correlation to stocks."
        except:
            pass
        
        market_implied = "UP" if yes_price > 0.5 else "DOWN"
        market_conf = abs(yes_price - 0.5) * 2  # 0-1 scale
        
        return f"""You are analyzing a {symbol}/USD 15-minute price prediction market on Polymarket.

## CURRENT DATA
- Market: YES={yes_price:.1%} (implies {market_implied} with {market_conf:.0%} confidence)
- ML Model: predicts {ml_direction} ({ml_confidence:.0%} confident){ob_info}{mom_info}{ws_info}

## YOUR TASK
Analyze whether {symbol} will go UP or DOWN in the next 15 minutes.

## THINK STEP BY STEP
1. What does the orderbook imbalance tell us about near-term direction?
2. Is momentum accelerating or fading?
3. Does the ML prediction align with market structure?
4. What's the VALUE proposition? (lower entry price = better value)

## CONFIDENCE CALIBRATION
- 50-55%: Slight edge, low conviction
- 55-65%: Moderate edge, tradeable
- 65-75%: Strong edge, high conviction  
- 75%+: Very strong edge, maximum size

## RESPOND IN JSON
```json
{{
    "direction": "UP" or "DOWN",
    "confidence": 0.50 to 0.85,
    "action": "BUY" or "WAIT",
    "urgency": "immediate" or "soon" or "wait",
    "key_factors": ["factor1", "factor2", "factor3"],
    "reasoning": "2-3 sentence explanation"
}}
```

Be decisive. If unsure, set confidence to 0.50 and action to "WAIT"."""

    def _build_deep_prompt(
        self,
        symbol: str,
        market_data: dict,
        ml_prediction: dict,
        reason: str,
    ) -> str:
        """Build a detailed prompt for deep analysis with full context."""
        now = datetime.now(timezone.utc)
        
        yes_price = market_data.get('yes_price', 0.5)
        no_price = market_data.get('no_price', 0.5)
        volume = market_data.get('volume', 0)
        
        ml_dir = ml_prediction.get('direction', 'UNKNOWN')
        ml_conf = ml_prediction.get('confidence', 0.5)
        
        return f"""# DEEP ANALYSIS: {symbol}/USD - {reason}

## Time Context
Current UTC: {now.strftime("%Y-%m-%d %H:%M:%S")}
Window: Next 15 minutes

## Market State
| Metric | Value |
|--------|-------|
| Polymarket YES | {yes_price:.3f} |
| Polymarket NO | {no_price:.3f} |
| 24h Volume | ${volume:,.0f} |
| Implied Direction | {"UP" if yes_price > 0.5 else "DOWN"} |

## ML Model Assessment
- Direction: {ml_dir}
- Confidence: {ml_conf:.1%}
- Trigger: {reason}

## ANALYSIS FRAMEWORK

### 1. Price Action Analysis
What does recent price action tell us about momentum and trend?

### 2. Order Flow
Is smart money accumulating or distributing based on orderbook?

### 3. Cross-Market Signals  
Are other crypto assets confirming or diverging?

### 4. Risk Assessment
What could make this trade fail? Where's the stop?

### 5. Value Proposition
At {yes_price:.1%} for YES, is there positive expected value?

## DECISION MATRIX
| Confidence | Position Size | Urgency |
|------------|--------------|---------|
| <55% | SKIP | - |
| 55-65% | Small (5%) | Soon |
| 65-75% | Medium (10%) | Now |
| 75%+ | Large (15%) | Immediate |

## RESPOND IN JSON
```json
{{
    "direction": "UP" or "DOWN",
    "confidence": 0.50 to 0.85,
    "action": "BUY" or "WAIT", 
    "urgency": "immediate" or "soon" or "wait",
    "key_factors": ["factor1", "factor2", "factor3"],
    "reasoning": "Detailed 3-5 sentence analysis",
    "risk_factors": ["risk1", "risk2"],
    "edge_source": "What gives us an edge here?"
}}
```"""
    
    def _build_batch_prompt(
        self,
        coins_data: list[dict],
        phase: str = "early",
    ) -> str:
        """Build a batch analysis prompt for multiple coins."""
        
        timing_context = {
            "early": "EARLY WINDOW (first 60s) - Odds still forming, look for mispricing",
            "late": "LATE WINDOW (last 30s) - Final call, momentum should be clear",
        }
        
        prompt = f"""# BATCH CRYPTO ANALYSIS - {timing_context.get(phase, 'STANDARD')}

You are analyzing 4 crypto prediction markets simultaneously.
Each market asks: "Will [COIN] go UP or DOWN in the next 15 minutes?"

## CURRENT MARKET STATE
"""
        for coin in coins_data:
            symbol = coin.get('symbol', '?')
            yes_price = coin.get('yes_price', 0.5)
            no_price = coin.get('no_price', 0.5)
            our_signal = coin.get('our_signal', '?')
            our_conf = coin.get('our_confidence', 0.5)
            ob_bias = coin.get('orderbook_bias', 0)
            mom_5m = coin.get('momentum_5m', 0)
            
            prompt += f"""
### {symbol}
- Market: YES={yes_price:.1%} | NO={no_price:.1%}
- Our Signal: {our_signal} ({our_conf:.0%})
- Orderbook: {ob_bias:+.2f} (neg=bearish, pos=bullish)
- 5min Momentum: {mom_5m:+.2f}%
"""
        
        prompt += """
## ANALYSIS APPROACH
For each coin:
1. Does orderbook support the direction?
2. Is momentum confirming?
3. Is the entry price offering value (<50% = good value for YES, >50% = good value for NO)?
4. Are signals aligned or conflicting?

## CONFIDENCE RULES
- If OB and momentum align: Higher confidence (65%+)
- If signals conflict: Lower confidence or WAIT
- If entry price unfavorable (>55%): Reduce confidence by 10%

## RESPOND - One line per coin, EXACT format:
BTC: [UP/DOWN], [confidence]%, [BUY/WAIT], [key reason]
ETH: [UP/DOWN], [confidence]%, [BUY/WAIT], [key reason]
SOL: [UP/DOWN], [confidence]%, [BUY/WAIT], [key reason]
XRP: [UP/DOWN], [confidence]%, [BUY/WAIT], [key reason]

Be decisive. Pick a direction even if uncertain (use lower confidence)."""
        
        return prompt

    # ─────────────────────────────────────────────────────────────────────────
    # API Calls
    # ─────────────────────────────────────────────────────────────────────────
    
    async def quick_check(
        self,
        symbol: CryptoSymbol,
        yes_price: float,
        ml_direction: str,
        ml_confidence: float,
        orderbook_data: dict = None,
        momentum_data: dict = None,
    ) -> Optional[GrokResponse]:
        """
        Enhanced quick check using FAST model with structured prompts.
        
        Designed for rapid scanning with better context.
        """
        if not self._can_call_fast():
            logger.debug("Fast model rate limited")
            return None
        
        await self._ensure_client()
        
        prompt = self._build_fast_prompt(
            symbol, yes_price, ml_direction, ml_confidence,
            orderbook_data=orderbook_data,
            momentum_data=momentum_data,
        )
        
        try:
            resp = await self.client.post(
                f"{XAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROK_FAST,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 150,  # Keep it short for speed
                }
            )
            
            self._record_call(GROK_FAST)
            
            if resp.status_code != 200:
                logger.warning(f"Fast Grok error: {resp.status_code}")
                return None
            
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            
            result = self._parse_response(raw, GROK_FAST)
            
            # Cache result
            self.last_analysis[symbol] = (datetime.now(timezone.utc), result)
            
            # Log
            self.store.log_grok_call(
                symbol=symbol,
                trigger_reasons=["quick_check"],
                prompt=prompt,
                response={
                    "direction": result.direction,
                    "confidence": result.confidence,
                    "action": result.action,
                },
                cost_estimate=result.cost_estimate,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fast Grok error: {e}")
            return None
    
    async def deep_search(
        self,
        symbol: CryptoSymbol,
        market_data: dict,
        ml_prediction: dict,
        reason: str,
    ) -> Optional[GrokResponse]:
        """
        Deep search using expensive model.
        
        Only call for unusual situations!
        """
        if not self._can_call_deep():
            logger.warning("Deep model rate limited")
            return None
        
        await self._ensure_client()
        
        prompt = self._build_deep_prompt(symbol, market_data, ml_prediction, reason)
        
        try:
            resp = await self.client.post(
                f"{XAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROK_DEEP,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 800,
                }
            )
            
            self._record_call(GROK_DEEP)
            
            if resp.status_code != 200:
                logger.error(f"Deep Grok error: {resp.status_code} - {resp.text}")
                return None
            
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            
            result = self._parse_response(raw, GROK_DEEP)
            
            # Log
            self.store.log_grok_call(
                symbol=symbol,
                trigger_reasons=["deep_search", reason],
                prompt=prompt,
                response={
                    "direction": result.direction,
                    "confidence": result.confidence,
                    "action": result.action,
                    "reasoning": result.reasoning,
                    "key_factors": result.key_factors,
                },
                cost_estimate=result.cost_estimate,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Deep Grok error: {e}")
            return None
    
    async def analyze(
        self,
        symbol: CryptoSymbol,
        market_data: dict,
        ml_prediction: dict,
    ) -> Optional[GrokResponse]:
        """
        Smart analyze: Uses fast model, upgrades to deep if needed.
        """
        yes_price = market_data.get("yes_price", 0.5)
        ml_direction = ml_prediction.get("direction", "UP")
        ml_confidence = ml_prediction.get("confidence", 0.5)
        market_direction = market_data.get("market_direction", "UP")
        market_confidence = max(yes_price, 1 - yes_price)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        
        # Check if deep analysis needed
        needs_deep, reason = self.needs_deep_analysis(
            symbol, ml_confidence, ml_direction,
            market_direction, market_confidence, volume_ratio
        )
        
        if needs_deep and self._can_call_deep():
            logger.info(f"Using DEEP model for {symbol}: {reason}")
            return await self.deep_search(symbol, market_data, ml_prediction, reason)
        
        # Otherwise use fast model
        return await self.quick_check(symbol, yes_price, ml_direction, ml_confidence)
    
    def _parse_response(self, raw: str, model: str) -> GrokResponse:
        """Parse Grok's JSON response with enhanced field extraction."""
        cost = COST_FAST if model == GROK_FAST else COST_DEEP
        
        try:
            # Extract JSON from response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                data = json.loads(raw[json_start:json_end])
                
                # Extract confidence - handle both float and percentage string
                conf = data.get("confidence", 0.5)
                if isinstance(conf, str):
                    conf = float(conf.replace('%', '')) / 100 if '%' in conf else float(conf)
                conf = max(0.5, min(0.85, float(conf)))  # Clamp to reasonable range
                
                return GrokResponse(
                    direction=data.get("direction", "UP").upper(),
                    confidence=conf,
                    reasoning=data.get("reasoning", ""),
                    key_factors=data.get("key_factors", []),
                    action=data.get("action", "WAIT").upper(),
                    urgency=data.get("urgency", "wait").lower(),
                    model_used=model,
                    cost_estimate=cost,
                    raw_response=raw,
                )
        except Exception as e:
            logger.warning(f"Parse error: {e}, raw: {raw[:200]}")
        
        # Fallback: try to extract direction from raw text
        direction = "UP"
        if "DOWN" in raw.upper():
            direction = "DOWN"
        
        return GrokResponse(
            direction=direction,
            confidence=0.5,
            reasoning=f"Parse failed: {raw[:100]}",
            key_factors=[],
            action="WAIT",
            urgency="wait",
            model_used=model,
            cost_estimate=cost,
            raw_response=raw,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_stats(self) -> dict:
        """Get usage stats."""
        now = datetime.now(timezone.utc)
        
        return {
            "fast_calls_last_min": len(self.fast_calls),
            "fast_limit": self.MAX_FAST_PER_MINUTE,
            "deep_calls_last_hour": len(self.deep_calls),
            "deep_limit": self.MAX_DEEP_PER_HOUR,
            "can_call_fast": self._can_call_fast(),
            "can_call_deep": self._can_call_deep(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(level=logging.INFO)
    
    async with GrokProvider() as grok:
        print("Testing FAST model (grok-4-1-fast-reasoning)...")
        
        result = await grok.quick_check(
            symbol="BTC",
            yes_price=0.48,
            ml_direction="UP",
            ml_confidence=0.65,
        )
        
        if result:
            print(f"\nFast Result:")
            print(f"  Direction: {result.direction}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Action: {result.action}")
            print(f"  Urgency: {result.urgency}")
            print(f"  Model: {result.model_used}")
            print(f"  Cost: ${result.cost_estimate:.4f}")
        else:
            print("Fast call failed")
        
        print(f"\nStats: {grok.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
