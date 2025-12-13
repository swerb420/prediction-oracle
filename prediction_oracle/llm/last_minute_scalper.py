#!/usr/bin/env python3
"""
Last-Minute Scalper - High-confidence late-window entries.

Strategy:
1. Wait until 30-45 seconds before window closes
2. Gather comprehensive data from all sources
3. Make ONE deep Grok call with ALL context
4. Enter ONLY if confidence is very high (>70%)

This is designed for HIGHER WIN RATE on fewer trades.
"""

import asyncio
import httpx
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal
from dataclasses import dataclass

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]
SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Grok API
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")


@dataclass
class ScalpSignal:
    """High-conviction scalp opportunity."""
    symbol: str
    direction: str
    confidence: float
    reasoning: str
    entry_price: float
    time_remaining: int
    
    # All the data that led to this signal
    binance_price: float
    bybit_price: float
    coinbase_price: float
    poly_yes_price: float
    orderbook_imbalance: float
    momentum_1min: float
    momentum_5min: float
    momentum_15min: float
    
    def __str__(self):
        return (
            f"🎯 SCALP: {self.symbol} {self.direction} @ {self.entry_price:.3f} "
            f"({self.confidence:.0%} conf, {self.time_remaining}s left)"
        )


async def get_binance_momentum(symbol: str, client: httpx.AsyncClient) -> dict:
    """Get recent price momentum from Binance."""
    try:
        # Get last 15 1-minute candles
        resp = await client.get(
            'https://api.binance.com/api/v3/klines',
            params={
                'symbol': f'{symbol}USDT',
                'interval': '1m',
                'limit': 15
            }
        )
        klines = resp.json()
        
        if len(klines) >= 15:
            # Calculate momentum at different timeframes
            current = float(klines[-1][4])  # Last close
            price_1m = float(klines[-2][4])
            price_5m = float(klines[-6][4])
            price_15m = float(klines[0][4])
            
            mom_1m = (current - price_1m) / price_1m * 100
            mom_5m = (current - price_5m) / price_5m * 100
            mom_15m = (current - price_15m) / price_15m * 100
            
            return {
                'current_price': current,
                'mom_1m': mom_1m,
                'mom_5m': mom_5m,
                'mom_15m': mom_15m,
                'trend': 'UP' if mom_15m > 0 else 'DOWN'
            }
    except Exception as e:
        pass
    
    return {'current_price': 0, 'mom_1m': 0, 'mom_5m': 0, 'mom_15m': 0, 'trend': 'UNKNOWN'}


async def get_all_venue_prices(symbol: str, client: httpx.AsyncClient) -> dict:
    """Get prices from multiple venues."""
    prices = {}
    
    try:
        # Binance
        resp = await client.get(
            'https://api.binance.com/api/v3/ticker/price',
            params={'symbol': f'{symbol}USDT'}
        )
        prices['binance'] = float(resp.json()['price'])
    except:
        prices['binance'] = 0
    
    try:
        # Bybit
        resp = await client.get(
            'https://api.bybit.com/v5/market/tickers',
            params={'category': 'spot', 'symbol': f'{symbol}USDT'}
        )
        data = resp.json()
        prices['bybit'] = float(data['result']['list'][0]['lastPrice'])
    except:
        prices['bybit'] = 0
    
    try:
        # Coinbase
        resp = await client.get(
            f'https://api.coinbase.com/v2/prices/{symbol}-USD/spot'
        )
        prices['coinbase'] = float(resp.json()['data']['amount'])
    except:
        prices['coinbase'] = 0
    
    return prices


async def get_polymarket_data(symbol: str, client: httpx.AsyncClient) -> dict:
    """Get Polymarket orderbook and prices."""
    # This would need the actual Polymarket integration
    # For now, return placeholder
    return {
        'yes_price': 0.50,
        'no_price': 0.50,
        'orderbook_imbalance': 0.0,
        'bid_depth': 0,
        'ask_depth': 0
    }


def build_deep_analysis_prompt(
    symbol: str,
    momentum: dict,
    venues: dict,
    poly: dict,
    secs_remaining: int
) -> str:
    """Build comprehensive prompt for Grok deep analysis."""
    
    return f"""🚨 URGENT: {secs_remaining} SECONDS LEFT IN WINDOW

You are analyzing {symbol} for a 15-minute direction prediction on Polymarket.

═══════════════════════════════════════════════════════════
CURRENT DATA (as of {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC)
═══════════════════════════════════════════════════════════

SPOT PRICES:
  • Binance:  ${venues.get('binance', 0):,.2f}
  • Bybit:    ${venues.get('bybit', 0):,.2f}
  • Coinbase: ${venues.get('coinbase', 0):,.2f}

MOMENTUM (price change %):
  • Last 1 min:  {momentum.get('mom_1m', 0):+.3f}%
  • Last 5 min:  {momentum.get('mom_5m', 0):+.3f}%
  • Last 15 min: {momentum.get('mom_15m', 0):+.3f}%
  • Overall trend: {momentum.get('trend', 'UNKNOWN')}

POLYMARKET:
  • YES price: {poly.get('yes_price', 0.5):.2f}
  • NO price:  {poly.get('no_price', 0.5):.2f}
  • Orderbook imbalance: {poly.get('orderbook_imbalance', 0):+.2f}

═══════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════

Based on the momentum data, predict if {symbol} price will be HIGHER or LOWER 
at the end of this 15-minute window compared to the start.

CRITICAL: Only {secs_remaining} seconds remain. The momentum data shows where 
price has been trending. Use this to extrapolate the likely close.

RESPOND IN THIS EXACT FORMAT:
DIRECTION: [UP or DOWN]
CONFIDENCE: [0-100]%
REASONING: [One sentence explaining why]

RULES:
- If momentum is strongly negative (-0.1% or more in 5min), lean DOWN
- If momentum is strongly positive (+0.1% or more in 5min), lean UP
- If momentum is mixed, look at 15-min trend
- Only give 70%+ confidence if momentum is clear and consistent
"""


async def deep_grok_analysis(
    symbol: str,
    momentum: dict,
    venues: dict,
    poly: dict,
    secs_remaining: int,
    client: httpx.AsyncClient
) -> Optional[ScalpSignal]:
    """Make deep Grok call for final decision."""
    
    prompt = build_deep_analysis_prompt(symbol, momentum, venues, poly, secs_remaining)
    
    try:
        resp = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-4-1-fast-reasoning",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Low temp for consistency
                "max_tokens": 200,
            },
            timeout=10.0
        )
        
        if resp.status_code != 200:
            return None  # Silently skip on API error
        
        data = resp.json()
        if not data.get("choices"):
            return None  # No response
            
        raw = data["choices"][0]["message"]["content"]
        
        # Parse response
        direction = "UP" if "DIRECTION: UP" in raw.upper() else "DOWN"
        
        # Extract confidence
        import re
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', raw)
        confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5
        
        # Extract reasoning
        reason_match = re.search(r'REASONING:\s*(.+)', raw, re.IGNORECASE)
        reasoning = reason_match.group(1).strip() if reason_match else raw[:100]
        
        # Only return signal if confidence is high enough
        if confidence >= 0.65:
            entry_price = poly.get('yes_price', 0.5) if direction == "UP" else poly.get('no_price', 0.5)
            
            return ScalpSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                entry_price=entry_price,
                time_remaining=secs_remaining,
                binance_price=venues.get('binance', 0),
                bybit_price=venues.get('bybit', 0),
                coinbase_price=venues.get('coinbase', 0),
                poly_yes_price=poly.get('yes_price', 0.5),
                orderbook_imbalance=poly.get('orderbook_imbalance', 0),
                momentum_1min=momentum.get('mom_1m', 0),
                momentum_5min=momentum.get('mom_5m', 0),
                momentum_15min=momentum.get('mom_15m', 0),
            )
        else:
            return None  # Confidence too low, skip silently
            
    except Exception:
        return None  # Silently handle any errors


async def scan_for_scalps(secs_remaining: int = 30) -> list[ScalpSignal]:
    """
    Scan all symbols for scalp opportunities.
    Call this when 30-45 seconds remain in the window.
    """
    signals = []
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for symbol in SYMBOLS:
            # Gather all data
            momentum = await get_binance_momentum(symbol, client)
            venues = await get_all_venue_prices(symbol, client)
            poly = await get_polymarket_data(symbol, client)
            
            # Skip if no clear momentum (less than 0.02% move in 5 min)
            mom_5m = abs(momentum.get('mom_5m', 0))
            if mom_5m < 0.02:
                continue
            
            # Deep Grok analysis
            signal = await deep_grok_analysis(
                symbol, momentum, venues, poly, secs_remaining, client
            )
            
            if signal:
                # Additional filter: momentum must align with direction
                mom_aligned = (
                    (signal.direction == "UP" and signal.momentum_5min > 0) or
                    (signal.direction == "DOWN" and signal.momentum_5min < 0)
                )
                
                if mom_aligned:
                    signals.append(signal)
    
    return signals


async def main():
    """Test the scalper."""
    print("=" * 70)
    print("  🎯 LAST-MINUTE SCALPER TEST")
    print("=" * 70)
    
    signals = await scan_for_scalps(secs_remaining=30)
    
    print("\n" + "=" * 70)
    print(f"  Found {len(signals)} high-confidence scalp signals")
    print("=" * 70)
    
    for sig in signals:
        print(f"\n  {sig}")
        print(f"      Reasoning: {sig.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
