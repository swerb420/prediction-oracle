#!/usr/bin/env python3
"""
Smart Signal Trader - Real-time 15M trading with intelligent signals.

Combines:
1. Market Intelligence (orderbook, momentum, multi-venue)
2. ML Predictions (learning over time)
3. Fast Grok Analysis (tiered models)
4. Whale Intelligence (follow the big money)
5. Paper Trading with PnL tracking

Features:
- Scans every few seconds
- Detects entry opportunities
- Smart signal filtering
- Whale signal boosting
- Comprehensive logging
- Paper trading with real-time PnL

Usage:
    python smart_signal_trader.py --scan          # Scan for signals
    python smart_signal_trader.py --trade         # Paper trade on signals
    python smart_signal_trader.py --trade --fast  # Fast mode (every 5s)
"""

import argparse
import asyncio
import logging
import httpx
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, Literal

from market_intelligence import MarketIntelligence, CryptoMarketData
from real_data_store import get_store, RealDataStore, PaperTrade
from learning_ml_predictor import LearningMLPredictor
from grok_provider import GrokProvider
from trading_logger import get_logger, TradingLogger
from last_minute_scalper import scan_for_scalps, ScalpSignal

# Whale Intelligence integration
try:
    from whale_intelligence import WhaleIntelligence, DB_PATH as WHALE_DB_PATH
    WHALE_ENABLED = WHALE_DB_PATH.exists()
except ImportError:
    WHALE_ENABLED = False

# Market Open Strategy integration (Binance-verified patterns)
try:
    from market_open_strategy import get_market_open_signal
    MARKET_OPEN_ENABLED = True
except ImportError:
    MARKET_OPEN_ENABLED = False
    def get_market_open_signal(*args, **kwargs):
        return {'has_signal': False, 'direction': None, 'confidence': 0}

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]
SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
# OPTIMIZED: 15M has 78.5% WR vs 48.9% on 1H - focus on best performer!
TIMEFRAMES = ["15M"]  # Was ["15M", "1H", "4H"]

# Timeframe window durations in seconds
TIMEFRAME_WINDOWS = {
    "15M": 900,
    "1H": 3600,
    "4H": 14400,
}


async def get_binance_price_direction(symbol: str, window_start: datetime, window_end: datetime) -> tuple[str, float]:
    """
    Get the ACTUAL price direction from Binance klines.
    Returns (direction, change_percent) where direction is 'UP', 'DOWN', or 'UNKNOWN'.
    
    This is the SOURCE OF TRUTH for whether a prediction was correct!
    """
    start_ts = int(window_start.timestamp()) * 1000
    end_ts = int(window_end.timestamp()) * 1000
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                'https://api.binance.com/api/v3/klines',
                params={
                    'symbol': f'{symbol}USDT',
                    'interval': '1m',
                    'startTime': start_ts,
                    'endTime': end_ts,
                    'limit': 20
                }
            )
            klines = resp.json()
            
            if klines and len(klines) >= 2:
                open_price = float(klines[0][1])   # Open of first candle
                close_price = float(klines[-1][4]) # Close of last candle
                change_pct = (close_price - open_price) / open_price * 100
                direction = 'UP' if close_price > open_price else 'DOWN'
                return direction, change_pct
            
            return 'UNKNOWN', 0.0
    except Exception as e:
        logger.warning(f"Binance API error for {symbol}: {e}")
        return 'UNKNOWN', 0.0


def get_wall_street_bias() -> tuple[float, str, bool]:
    """
    Get bearish bias during Wall Street trading hours.
    
    Market opens at 9:30 AM EST and the first hour (9:30-10:30) typically sees
    institutional selling of crypto. Also applies during general trading hours
    on weekdays.
    
    Returns (bias, reason, is_dump_mode) where:
    - bias is a signal adjustment (-0.3 to 0)
    - reason is a string description
    - is_dump_mode is True during 9:30-11:00 AM EST (BOOST positions!)
    """
    from datetime import datetime
    import pytz
    
    try:
        est = pytz.timezone('America/New_York')
        now_est = datetime.now(est)
        
        # Skip weekends
        if now_est.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return 0.0, "", False
        
        hour = now_est.hour
        minute = now_est.minute
        current_time = hour + minute / 60
        
        # Pre-market (8:00-9:30 AM EST) - slight bearish
        if 8.0 <= current_time < 9.5:
            return -0.1, "pre-market_open", False
        
        # Market open first hour (9:30-10:30 AM EST) - STRONG bearish + DUMP MODE
        if 9.5 <= current_time < 10.5:
            return -0.3, "🔥WALL_STREET_DUMP🔥", True  # DUMP MODE = BOOST!
        
        # Extended dump window (10:30-11:00 AM EST) - still strong + DUMP MODE
        if 10.5 <= current_time < 11.0:
            return -0.2, "🔥EXTENDED_DUMP🔥", True  # Still DUMP MODE
        
        # First 2 hours (11:00-11:30 AM EST) - moderate bearish
        if 11.0 <= current_time < 11.5:
            return -0.15, "morning_selling", False
        
        # Lunch hour (11:30 AM - 1:00 PM EST) - slight bearish
        if 11.5 <= current_time < 13.0:
            return -0.05, "lunch_hour", False
        
        # Afternoon session (1:00-4:00 PM EST) - neutral to slight bearish
        if 13.0 <= current_time < 16.0:
            return -0.05, "afternoon_session", False
        
        return 0.0, "", False
        
    except Exception:
        return 0.0, "", False


def get_whale_signal_quality(direction: str, whale_signal: str, position_size: float = 0, 
                             entry_price: float = 0.5, whale_name: str = "") -> dict:
    """
    Filter whale signals based on VERIFIED patterns from 683 paper trades + whale buy/sell analysis.
    
    Data-driven patterns discovered:
    
    🔥 GOLDEN CONDITIONS (80%+ win rate from paper trades):
        - 15m-a4 UP + HIGH price + 21:00 UTC = 100% (21 trades!)
        - ExpressoMartini DOWN + VERY_LOW price + 23:00 = 97.5% (40 trades!)
        - ExpressoMartini general @ 23:00 = 89-97% WR
        - ExpressoMartini UP + VERY_LOW price + 22:00 = 86.4% (22 trades)
        - UP + VERY_LOW price overall = 80.9% WR, +$1084 profit
    
    🔥 GOLDEN from buy/sell price analysis (29,000+ trades):
        - 18:00 UTC (2pm EST) DOWN = +0.24 price gain (BEST HOUR)
        - 20:00 UTC (4pm EST) DOWN = +0.15 price gain 
        - 01:00 UTC (9pm EST) DOWN = +0.14 price gain
        - 00:00 UTC DOWN = +0.10 price gain (but paper trades show losses - skip)
        - 22:00 UTC UP = +0.07 price gain
    
    ❌ DEATH CONDITIONS (BLOCK COMPLETELY):
        - ANY UP @ 00:00 UTC = -0.15 price loss, paper trades 5.6% WR
        - ANY UP @ 20:00 UTC = -0.17 price loss (WORST)
        - ANY UP @ 01:00 UTC = -0.11 price loss
        - ExpressoMartini DOWN + HIGH price + 22:00 = 0% (37 trades!)
        - ExpressoMartini UP + MEDIUM price + 23:00 = 5% WR
        - 15m-a4 DOWN + HIGH price + 22:00 = 8.7% WR
        - DOWN + HIGH price overall = 25.4% WR, -$577 loss
    
    10am EST Market Open Pattern (14:00-15:00 UTC):
        - 93% of whale money bets DOWN at 0.81 avg price
        - Only 7% bets UP at 0.36 avg price
        - Strong DOWN bias at US market open!
    
    Price buckets (entry_price):
        very_low: < 0.40
        low: 0.40 - 0.49
        medium: 0.50 - 0.59
        high: >= 0.60
    
    Returns dict with:
        - quality: 'golden', 'high', 'medium', 'low', 'block'
        - weight_multiplier: 0.0 to 3.0
        - reason: explanation
    """
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Determine price bucket
    if entry_price < 0.40:
        price_bucket = 'very_low'
    elif entry_price < 0.50:
        price_bucket = 'low'
    elif entry_price < 0.60:
        price_bucket = 'medium'
    else:
        price_bucket = 'high'
    
    quality = "medium"
    weight_multiplier = 1.0
    reasons = []
    
    # Normalize whale name for comparison
    is_espresso = 'espresso' in whale_name.lower() if whale_name else True
    is_a4 = 'a4' in whale_name.lower() or '15m' in whale_name.lower() if whale_name else False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DEATH CONDITIONS - BLOCK COMPLETELY (verified from 683 trades + buy/sell analysis)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # 00:00 UTC - BLOCK UP ONLY (paper trades 5.6% WR, buy/sell shows -0.15 loss for UP)
    # DOWN actually profitable here (+0.10 price gain) - we'll handle in golden section
    if hour == 0 and whale_signal == 'UP':
        quality = "block"
        weight_multiplier = 0.0
        reasons.append("❌ BLOCK: UP @ 00:00 UTC = -0.15 loss, paper 5.6% WR!")
        return {"quality": quality, "weight_multiplier": weight_multiplier,
                "reasons": reasons, "reason": " | ".join(reasons)}
    
    # 20:00 UTC (4pm EST) - BLOCK UP (WORST HOUR: -0.17 price loss)
    if hour == 20 and whale_signal == 'UP':
        quality = "block"
        weight_multiplier = 0.0
        reasons.append("❌ BLOCK: UP @ 20:00 UTC = -0.17 price loss (WORST)!")
        return {"quality": quality, "weight_multiplier": weight_multiplier,
                "reasons": reasons, "reason": " | ".join(reasons)}
    
    # 01:00 UTC (9pm EST) - BLOCK UP (-0.11 price loss)
    if hour == 1 and whale_signal == 'UP':
        quality = "block"
        weight_multiplier = 0.0
        reasons.append("❌ BLOCK: UP @ 01:00 UTC = -0.11 price loss!")
        return {"quality": quality, "weight_multiplier": weight_multiplier,
                "reasons": reasons, "reason": " | ".join(reasons)}
    
    # ExpressoMartini DOWN + HIGH price + 22:00 = 0% WR (37 trades!)
    if is_espresso and whale_signal == 'DOWN' and price_bucket == 'high' and hour == 22:
        quality = "block"
        weight_multiplier = 0.0
        reasons.append("❌ BLOCK: Espresso DOWN high @ 22:00 = 0% WR!")
        return {"quality": quality, "weight_multiplier": weight_multiplier,
                "reasons": reasons, "reason": " | ".join(reasons)}
    
    # ExpressoMartini UP + MEDIUM price + 23:00 = 5% WR
    if is_espresso and whale_signal == 'UP' and price_bucket == 'medium' and hour == 23:
        quality = "block"
        weight_multiplier = 0.0
        reasons.append("❌ BLOCK: Espresso UP medium @ 23:00 = 5% WR!")
        return {"quality": quality, "weight_multiplier": weight_multiplier,
                "reasons": reasons, "reason": " | ".join(reasons)}
    
    # 15m-a4 DOWN + HIGH price + 22:00 = 8.7% WR
    if is_a4 and whale_signal == 'DOWN' and price_bucket == 'high' and hour == 22:
        quality = "block"
        weight_multiplier = 0.0
        reasons.append("❌ BLOCK: 15m-a4 DOWN high @ 22:00 = 8.7% WR!")
        return {"quality": quality, "weight_multiplier": weight_multiplier,
                "reasons": reasons, "reason": " | ".join(reasons)}
    
    # DOWN + HIGH price overall = 25.4% WR (avoid unless golden hour)
    if whale_signal == 'DOWN' and price_bucket == 'high' and hour not in [23]:
        quality = "low"
        weight_multiplier = 0.3
        reasons.append("⚠️ DOWN high price = 25.4% WR overall")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GOLDEN CONDITIONS - MAX WEIGHT (verified 80%+ WR from paper + buy/sell analysis)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # 18:00 UTC (2pm EST) DOWN = +0.24 price gain (BEST HOUR from buy/sell analysis!)
    elif whale_signal == 'DOWN' and hour == 18:
        quality = "golden"
        weight_multiplier = 2.8
        reasons.append("🌟 GOLDEN: DOWN @ 18:00 UTC = +0.24 gain (BEST)!")
    
    # 20:00 UTC (4pm EST) DOWN = +0.15 price gain 
    elif whale_signal == 'DOWN' and hour == 20:
        quality = "golden"
        weight_multiplier = 2.5
        reasons.append("🔥 GOLDEN: DOWN @ 20:00 UTC = +0.15 gain!")
    
    # 01:00 UTC (9pm EST) DOWN = +0.14 price gain
    elif whale_signal == 'DOWN' and hour == 1:
        quality = "golden"
        weight_multiplier = 2.4
        reasons.append("🔥 GOLDEN: DOWN @ 01:00 UTC = +0.14 gain!")
    
    # 00:00 UTC DOWN = +0.10 price gain (only DOWN is good here, UP blocked above)
    elif whale_signal == 'DOWN' and hour == 0:
        quality = "high"
        weight_multiplier = 1.8
        reasons.append("✅ DOWN @ 00:00 UTC = +0.10 gain!")
    
    # 15m-a4 UP + HIGH price + 21:00 = 100% WR (21 trades!)
    elif is_a4 and whale_signal == 'UP' and price_bucket == 'high' and hour == 21:
        quality = "golden"
        weight_multiplier = 3.0
        reasons.append("🌟 GOLDEN: 15m-a4 UP high @ 21:00 = 100% WR!")
    
    # ExpressoMartini DOWN + VERY_LOW price + 23:00 = 97.5% WR (40 trades!)
    elif is_espresso and whale_signal == 'DOWN' and price_bucket == 'very_low' and hour == 23:
        quality = "golden"
        weight_multiplier = 2.8
        reasons.append("🌟 GOLDEN: Espresso DOWN very_low @ 23:00 = 97.5% WR!")
    
    # ExpressoMartini DOWN @ 23:00 general = 89-97% WR
    elif is_espresso and whale_signal == 'DOWN' and hour == 23:
        quality = "golden"
        weight_multiplier = 2.5
        reasons.append("🔥 Espresso DOWN @ 23:00 = 89%+ WR!")
    
    # ExpressoMartini UP + HIGH @ 21:00 = 89.5% WR
    elif is_espresso and whale_signal == 'UP' and price_bucket == 'high' and hour == 21:
        quality = "golden"
        weight_multiplier = 2.5
        reasons.append("🔥 Espresso UP high @ 21:00 = 89.5% WR!")
    
    # ExpressoMartini UP + VERY_LOW price + 22:00 = 86.4% WR (22 trades)
    elif is_espresso and whale_signal == 'UP' and price_bucket == 'very_low' and hour == 22:
        quality = "golden"
        weight_multiplier = 2.3
        reasons.append("🔥 Espresso UP very_low @ 22:00 = 86.4% WR!")
    
    # 22:00 UTC UP = +0.07 price gain (from buy/sell analysis)
    elif whale_signal == 'UP' and hour == 22 and price_bucket != 'high':
        quality = "high"
        weight_multiplier = 1.6
        reasons.append("✅ UP @ 22:00 UTC = +0.07 gain!")
    
    # UP + VERY_LOW price overall = 80.9% WR (+$1084 profit)
    elif whale_signal == 'UP' and price_bucket == 'very_low':
        quality = "high"
        weight_multiplier = 2.0
        reasons.append("✅ UP very_low price = 80.9% WR!")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GOOD CONDITIONS (60-70% WR)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # 21:00 UTC general = good hour
    elif hour == 21:
        quality = "high"
        weight_multiplier = 1.7
        reasons.append("✅ 21:00 UTC prime hour (60%+ WR)")
    
    # 23:00 UTC general for ExpressoMartini
    elif is_espresso and hour == 23:
        quality = "high"
        weight_multiplier = 1.8
        reasons.append("✅ Espresso @ 23:00 = 61.8% WR hour")
    
    # ExpressoMartini UP + other @ 22:00 = 62.9% WR
    elif is_espresso and whale_signal == 'UP' and hour == 22:
        quality = "medium"
        weight_multiplier = 1.3
        reasons.append("ExpressoMartini UP @ 22:00 = 62.9% WR")
    
    # DOWN + very_low price = 55-56% WR (contrarian)
    elif whale_signal == 'DOWN' and price_bucket == 'very_low':
        quality = "medium"
        weight_multiplier = 1.2
        reasons.append("DOWN very_low = 55% WR (contrarian)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEAK CONDITIONS - REDUCE WEIGHT
    # ═══════════════════════════════════════════════════════════════════════════
    
    # 22:00 DOWN generally weak
    elif hour == 22 and whale_signal == 'DOWN':
        quality = "low"
        weight_multiplier = 0.4
        reasons.append("⚠️ 22:00 DOWN = weak overall")
    
    # UP + LOW price = 42.4% WR
    elif whale_signal == 'UP' and price_bucket == 'low':
        quality = "low"
        weight_multiplier = 0.5
        reasons.append("⚠️ UP low price = 42.4% WR")
    
    # UP + MEDIUM price = 34.1% WR
    elif whale_signal == 'UP' and price_bucket == 'medium':
        quality = "low"
        weight_multiplier = 0.4
        reasons.append("⚠️ UP medium price = 34.1% WR")
    
    # Default - neutral
    else:
        quality = "medium"
        weight_multiplier = 1.0
        reasons.append("Standard signal")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MARKET OPEN BOOST - Binance-verified day/hour patterns (60+ days backtest)
    # ═══════════════════════════════════════════════════════════════════════════
    if MARKET_OPEN_ENABLED:
        market_signal = get_market_open_signal(hour=hour)
        if market_signal['has_signal']:
            # If our direction matches the market open pattern, BOOST!
            if market_signal['direction'] == whale_signal:
                boost = market_signal['confidence'] - 0.50  # e.g., 0.667 -> +0.167
                weight_multiplier = min(3.0, weight_multiplier + boost)
                if market_signal['confidence'] >= 0.63:
                    quality = "golden"
                elif market_signal['confidence'] >= 0.58 and quality not in ["golden"]:
                    quality = "high"
                reasons.append(market_signal['reason'])
            # If our direction is OPPOSITE to the pattern, REDUCE!
            elif market_signal['direction'] and market_signal['direction'] != whale_signal:
                penalty = market_signal['confidence'] - 0.50
                weight_multiplier = max(0.2, weight_multiplier - penalty)
                if quality == "golden":
                    quality = "medium"
                reasons.append(f"⚠️ Against market open pattern ({market_signal['direction']})")
    
    # Ensure bounds
    weight_multiplier = max(0.0, min(3.0, weight_multiplier))
    
    return {
        "quality": quality,
        "weight_multiplier": weight_multiplier,
        "reasons": reasons,
        "reason": " | ".join(reasons) if reasons else "Standard signal"
    }


@dataclass
class EntrySignal:
    """A detected entry opportunity with full context for analysis."""
    symbol: str
    direction: str  # "UP" or "DOWN"
    confidence: float
    timeframe: str  # "15M", "1H", "4H"
    
    # Signal components
    orderbook_signal: float  # -1 to 1
    momentum_signal: float
    ml_signal: float
    
    # Entry details
    entry_price: float  # Polymarket price
    recommended_size_pct: float
    
    # Optional fields with defaults
    grok_signal: Optional[float] = None
    reasons: list[str] = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENHANCED CONTEXT - Full market data for detailed logging
    # ═══════════════════════════════════════════════════════════════════════════
    # Market data
    poly_yes_price: float = 0.0
    poly_no_price: float = 0.0
    spot_price: float = 0.0
    market_spread: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    
    # Multi-venue prices
    binance_price: float = 0.0
    bybit_price: float = 0.0
    coinbase_price: float = 0.0
    venue_agreement: float = 0.0
    
    # Orderbook details
    ob_bid_depth: float = 0.0
    ob_ask_depth: float = 0.0
    ob_top_bid: float = 0.0
    ob_top_ask: float = 0.0
    ob_weighted_mid: float = 0.0
    
    # Momentum details
    momentum_1min: float = 0.0
    momentum_5min: float = 0.0
    momentum_15min: float = 0.0
    momentum_1h: float = 0.0
    momentum_trend: str = ""
    
    # Signal analysis
    signal_raw_combined: float = 0.0
    signal_weights: str = ""
    
    # Wall Street dump mode - for position sizing boost
    is_dump_mode: bool = False
    ws_reason: str = ""
    
    # Grok analysis (filled after validation)
    grok_model_used: str = ""
    grok_confidence: float = 0.0
    grok_reasoning: str = ""
    grok_key_factors: str = ""
    grok_action: str = ""
    grok_urgency: str = ""
    grok_full_response: str = ""
    grok_cost: float = 0.0
    
    # Whale Intelligence (filled during signal generation)
    whale_signal: Optional[str] = None  # UP, DOWN, NEUTRAL
    whale_strength: int = 0  # 0-100
    whale_boost: float = 1.0  # Position size multiplier
    whale_confidence_add: float = 0.0  # Confidence adjustment
    whale_agree: Optional[bool] = None  # True if whales agree with our direction
    whale_reason: str = ""  # Explanation
    
    @property
    def total_signal(self) -> float:
        """Combined signal strength (absolute value for trade decisions)."""
        signals = [self.orderbook_signal, self.momentum_signal, self.ml_signal]
        if self.grok_signal is not None:
            signals.append(self.grok_signal)
        avg = sum(signals) / len(signals)
        return abs(avg)  # Use absolute value for signal strength


class SmartSignalTrader:
    """
    Fast signal-based trader for 15M crypto markets.
    
    Scanning loop:
    1. Fetch market intelligence (parallel)
    2. Check for entry signals
    3. Validate with ML + optional Grok
    4. Execute paper trades
    5. Monitor positions for early exit
    6. Track PnL
    
    Smart features:
    - Dynamic position sizing based on confidence
    - Multiple entries on very high confidence
    - Grok validation only for borderline cases
    - Better expiry tracking
    - EARLY EXIT: Stop-loss, take-profit, orderbook reversal detection
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL THRESHOLDS - Working configuration (70% win rate, +$1653.86)
    # ═══════════════════════════════════════════════════════════════════════════
    MIN_SIGNAL_TO_ALERT = 0.10  # Minimum signal strength to log
    MIN_SIGNAL_TO_TRADE = 0.20  # Minimum to paper trade (RAISED from 0.15)
    HIGH_CONFIDENCE_THRESHOLD = 0.62  # Auto-trade above this (RAISED from 0.58)
    ULTRA_HIGH_CONFIDENCE = 0.75  # Allow trading without Grok (RAISED from 0.70)
    MIN_CONFIDENCE = 0.58  # Minimum confidence to consider (RAISED from 0.52)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EDGE-BASED FILTERS - AGGRESSIVE! Only trade high-edge setups
    # ═══════════════════════════════════════════════════════════════════════════
    MIN_ORDERBOOK_STRENGTH = 0.90  # RAISED from 0.88 - demand stronger OB
    
    # Entry price ranges - TIGHTER for better edge
    ENTRY_PRICE_RANGES = {
        "15M": (0.45, 0.52),  # TIGHTENED for max edge
        "1H":  (0.30, 0.60),  # Tighter for hourly
        "4H":  (0.25, 0.65),  # Tighter for 4-hour
    }
    MIN_ENTRY_PRICE = 0.45  # Raised from 0.40
    MAX_ENTRY_PRICE = 0.52  # Tightened from 0.55
    
    # Position sizing - smaller for longer timeframes (more risk)
    POSITION_SIZE_BY_TF = {
        "15M": 1.0,   # Full size
        "1H":  0.75,  # 75% size
        "4H":  0.50,  # 50% size (longer exposure)
    }
    
    # Position sizing - WORKING CONFIG
    BASE_SIZE_PCT = 0.08  # 8% base position
    MAX_SIZE_PCT = 0.20   # 20% max single position
    ULTRA_SIZE_PCT = 0.30 # 30% for ultra high confidence
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WALL STREET DUMP MODE - BOOST during 9:30-11:00 AM EST
    # ═══════════════════════════════════════════════════════════════════════════
    WALL_STREET_SIZE_BOOST = 2.0  # 2x position size during dump window
    WALL_STREET_MAX_SIZE_PCT = 0.50  # Allow up to 50% position during dump
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RATE LIMITS - STRICT! Quality over quantity
    # ═══════════════════════════════════════════════════════════════════════════
    MAX_POSITIONS = 16             # Reduced from 24 - fewer but better trades
    MAX_POSITIONS_PER_SYMBOL = 4   # Reduced from 6 - max 4 per symbol
    MAX_TRADES_PER_HOUR = 24       # Reduced from 48 - quality over quantity
    MIN_SECONDS_BETWEEN_TRADES = 60  # RAISED from 30 - more time between trades
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EARLY EXIT SETTINGS - Smart risk management
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Stop-loss: Exit if price moves against us
    STOP_LOSS_PCT = 0.15  # Exit if entry price drops 15% (e.g., 0.50 -> 0.425)
    
    # Take-profit: Lock in gains
    TAKE_PROFIT_PCT = 0.30  # Exit if price gains 30% (e.g., 0.50 -> 0.65)
    
    # Orderbook reversal: Exit if market sentiment flips
    OB_REVERSAL_THRESHOLD = 0.50  # Exit if OB flips to opposite direction with 50%+ strength
    
    # Trailing stop: Once in profit, don't give it all back
    TRAILING_STOP_ACTIVATION = 0.15  # Activate trailing stop after 15% gain
    TRAILING_STOP_DISTANCE = 0.08   # Trail 8% behind peak price
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BTC CIRCUIT BREAKER - Exit losing positions on big BTC moves
    # ═══════════════════════════════════════════════════════════════════════════
    # Conservative settings - only exit positions already in the red
    BTC_ALERT_THRESHOLD = 0.005    # 0.5% move in 1 min = ALERT mode
    BTC_EXIT_THRESHOLD = 0.010     # 1.0% move in 2 min = EXIT losing positions
    BTC_HISTORY_SECONDS = 120      # Keep 2 minutes of BTC price history
    BTC_CHECK_INTERVAL = 15        # Check every 15 seconds (matches scan interval)
    
    # Grok timing strategy (seconds into 15-min window)
    GROK_EARLY_WINDOW = 60  # Call Grok in first 60 seconds
    GROK_LATE_WINDOW = 30   # Call Grok in last 30 seconds
    
    def __init__(
        self,
        use_grok: bool = True,
        starting_capital: float = 1000.0,
    ):
        self.use_grok = use_grok
        self.starting_capital = starting_capital
        
        # Components
        self.store = get_store()
        self.log = get_logger()
        self.intel = MarketIntelligence()
        self.predictor = LearningMLPredictor(store=self.store)
        self.grok: Optional[GrokProvider] = None
        
        # Portfolio tracking
        self.capital = starting_capital
        self.positions: dict[str, dict] = {}  # symbol -> position
        self.trade_history: list[dict] = []
        self.recent_trades: list[datetime] = []  # For rate limiting
        
        # Grok batch tracking
        self.last_grok_batch_time: Optional[datetime] = None
        self.current_15m_window: Optional[int] = None
        self.grok_early_done: bool = False
        self.grok_late_done: bool = False
        
        # Stats
        self.scan_count = 0
        self.signal_count = 0
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        
        # BTC Circuit Breaker - price history for detecting big moves
        self.btc_price_history: list[tuple[datetime, float]] = []  # (timestamp, price)
        self.btc_circuit_breaker_triggered = False
        self.btc_circuit_breaker_direction: Optional[str] = None  # "PUMP" or "DUMP"
        self.btc_circuit_breaker_cooldown: Optional[datetime] = None
    
    async def __aenter__(self):
        await self.intel.__aenter__()
        if self.use_grok:
            self.grok = GrokProvider(store=self.store)
            await self.grok.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        await self.intel.__aexit__(*args)
        if self.grok:
            await self.grok.__aexit__(*args)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Signal Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    async def detect_signals(self, timeframes: list[str] = None) -> list[EntrySignal]:
        """Detect entry signals for all symbols across all timeframes."""
        if timeframes is None:
            timeframes = TIMEFRAMES  # Default to all: 15M, 1H, 4H
        
        signals = []
        
        for tf in timeframes:
            # Fetch market data for this timeframe
            markets = await self.intel.get_all_markets(timeframe=tf)
            
            for symbol, data in markets.items():
                signal = await self._analyze_for_signal(symbol, data, tf)
                if signal:
                    signals.append(signal)
        
        return signals
    
    async def _analyze_for_signal(
        self, 
        symbol: str, 
        data: CryptoMarketData,
        timeframe: str = "15M",
    ) -> Optional[EntrySignal]:
        """Analyze market data for entry signal with FULL context logging."""
        import json as _json
        reasons = []
        
        # ═══════════════════════════════════════════════════════════════════════
        # 1. ORDERBOOK ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════
        ob_signal = 0.0
        ob_bid_depth = 0.0
        ob_ask_depth = 0.0
        ob_top_bid = 0.0
        ob_top_ask = 0.0
        ob_weighted_mid = 0.0
        
        if data.orderbook:
            ob_signal = data.orderbook.imbalance
            ob_bid_depth = getattr(data.orderbook, 'total_bid_size', 0) or 0
            ob_ask_depth = getattr(data.orderbook, 'total_ask_size', 0) or 0
            ob_top_bid = getattr(data.orderbook, 'best_bid', 0) or 0
            ob_top_ask = getattr(data.orderbook, 'best_ask', 0) or 0
            ob_weighted_mid = getattr(data.orderbook, 'mid_price', 0) or 0
            
            if abs(ob_signal) > 0.3:
                direction = "bullish" if ob_signal > 0 else "bearish"
                strength = "strong" if abs(ob_signal) >= 0.85 else "moderate"
                reasons.append(f"OB {strength} {direction} ({ob_signal:+.2f})")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. MOMENTUM ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════
        mom_signal = 0.0
        momentum_1min = 0.0
        momentum_5min = 0.0
        momentum_15min = 0.0
        momentum_1h = 0.0
        momentum_trend = "stable"
        
        if data.momentum:
            mom_signal = max(-1, min(1, data.momentum.change_5min_pct / 3))
            momentum_1min = getattr(data.momentum, 'change_1min_pct', 0) or 0
            momentum_5min = data.momentum.change_5min_pct or 0
            momentum_15min = getattr(data.momentum, 'change_15min_pct', 0) or 0
            momentum_1h = getattr(data.momentum, 'change_1h_pct', 0) or 0
            
            # Determine trend
            if momentum_1min * momentum_5min > 0 and abs(momentum_1min) > abs(momentum_5min):
                momentum_trend = "accelerating"
            elif momentum_1min * momentum_5min < 0:
                momentum_trend = "reversing"
            elif abs(momentum_5min) < 0.1:
                momentum_trend = "stable"
            else:
                momentum_trend = "decelerating"
            
            if abs(data.momentum.change_5min_pct) > 0.5:
                direction = "up" if mom_signal > 0 else "down"
                reasons.append(f"5m {direction} ({momentum_5min:+.2f}%), {momentum_trend}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. ML PREDICTION
        # ═══════════════════════════════════════════════════════════════════════
        ml_pred = self.predictor.predict(symbol, data.to_dict())
        ml_signal = (ml_pred.confidence - 0.5) * 2  # Normalize to -1 to 1
        if ml_pred.direction == "DOWN":
            ml_signal = -ml_signal
        
        if ml_pred.training_examples > 10:
            reasons.append(f"ML: {ml_pred.direction} ({ml_pred.confidence:.0%})")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4. MARKET SIGNAL
        # ═══════════════════════════════════════════════════════════════════════
        poly_signal = (data.poly_yes_price - 0.5) * 2
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4.5 WALL STREET FACTOR - Bearish bias during US market hours
        # ═══════════════════════════════════════════════════════════════════════
        ws_bias, ws_reason, is_dump_mode = get_wall_street_bias()
        if ws_bias != 0:
            reasons.append(f"WS: {ws_reason} ({ws_bias:+.2f})")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4.6 WHALE INTELLIGENCE - Get signal from top crypto whales (15m-a4, ExpressoMartini)
        # ═══════════════════════════════════════════════════════════════════════
        whale_signal_value = 0.0  # -1 to 1
        whale_signal = None
        whale_strength = 0
        whale_boost = 1.0
        whale_confidence_add = 0.0
        whale_agree = None
        whale_reason = ""
        whale_quality = "medium"
        whale_weight_mult = 1.0
        
        if WHALE_ENABLED:
            try:
                with WhaleIntelligence() as wi:
                    # Get 15M whale signals (from the elite traders like ExpressoMartini, 15m-a4)
                    whale_signals = wi.get_15m_whale_signals(minutes_back=15)
                    if symbol in whale_signals:
                        sig = whale_signals[symbol]
                        whale_signal = sig['signal']
                        whale_strength = sig['strength']
                        whale_name = sig.get('whale_name', sig.get('top_whales', ['Unknown'])[0] if 'top_whales' in sig else 'Unknown')
                        
                        # Get entry price for the whale signal direction
                        whale_entry_price = data.poly_yes_price if whale_signal == 'UP' else data.poly_no_price
                        
                        # 🔥 UPDATED: Apply whale signal quality filter with price and whale name
                        whale_filter = get_whale_signal_quality(
                            direction=whale_signal,  # Whale's direction
                            whale_signal=whale_signal,
                            position_size=sig.get('avg_size', 0),
                            entry_price=whale_entry_price,  # NEW: Pass Polymarket price
                            whale_name=whale_name if isinstance(whale_name, str) else str(whale_name)  # NEW: Pass whale name
                        )
                        whale_quality = whale_filter['quality']
                        whale_weight_mult = whale_filter['weight_multiplier']
                        
                        # Skip "block" quality signals entirely
                        if whale_quality == "block":
                            whale_signal_value = 0.0
                            reasons.append(f"🐋❌ Whale: BLOCKED ({whale_filter['reason']})")
                        else:
                            # Convert to -1 to 1 signal for weighted combination
                            # UP with high confidence = positive, DOWN = negative
                            if whale_signal == 'UP':
                                whale_signal_value = (sig['confidence'] - 0.5) * 2  # 0.7 conf -> 0.4 signal
                            elif whale_signal == 'DOWN':
                                whale_signal_value = -(sig['confidence'] - 0.5) * 2  # 0.7 conf -> -0.4 signal
                            
                            # Scale by strength (0-100 -> 0-1)
                            whale_signal_value = whale_signal_value * (whale_strength / 100)
                            
                            # 🔥 Apply quality-based weight multiplier (now up to 3.0x for golden)
                            whale_signal_value = whale_signal_value * whale_weight_mult
                            
                            if whale_strength >= 40:
                                quality_emoji = {"golden": "🌟", "high": "🔥", "medium": "🐋", "low": "⚠️"}.get(whale_quality, "🐋")
                                reasons.append(f"{quality_emoji} Whale: {whale_signal} ({whale_strength}%) [{whale_quality}]")
                        
            except Exception as e:
                logger.debug(f"Whale intelligence unavailable: {e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 5. COMBINE SIGNALS - Whales now have proper weight!
        # ═══════════════════════════════════════════════════════════════════════
        # Whale gets 15% weight - same as a human expert trader signal
        if is_dump_mode:
            # During dump mode, give WS bias more weight, reduce others proportionally
            signal_weights = {"ob": 0.15, "mom": 0.15, "ml": 0.20, "poly": 0.10, "ws": 0.25, "whale": 0.15}
        else:
            # Normal mode: Whale signal is weighted like a smart trader indicator
            signal_weights = {"ob": 0.22, "mom": 0.22, "ml": 0.26, "poly": 0.10, "ws": 0.05, "whale": 0.15}
        
        combined = (
            ob_signal * signal_weights["ob"] +
            mom_signal * signal_weights["mom"] +
            ml_signal * signal_weights["ml"] +
            poly_signal * signal_weights["poly"] +
            ws_bias * signal_weights["ws"] +
            whale_signal_value * signal_weights["whale"]  # 🐋 Whale signal!
        )
        signal_raw_combined = combined  # Store raw value before processing
        
        # Determine direction
        if combined > 0:
            direction = "UP"
            entry_price = data.poly_yes_price
        else:
            direction = "DOWN"
            entry_price = data.poly_no_price
            combined = abs(combined)
        
        # Check if signal is strong enough
        if combined < self.MIN_SIGNAL_TO_ALERT:
            return None
        
        # Calculate confidence
        confidence = 0.5 + (combined * 0.3)
        
        # Calculate position size
        size_pct = self.BASE_SIZE_PCT
        if combined > 0.4:
            size_pct = self.BASE_SIZE_PCT + (combined - 0.4) * 0.3
        size_pct = min(size_pct, self.MAX_SIZE_PCT)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 5.5 WHALE POSITION BOOST - Extra size when whales agree with direction
        # ═══════════════════════════════════════════════════════════════════════
        if WHALE_ENABLED and whale_signal is not None:
            try:
                with WhaleIntelligence() as wi:
                    boost_data = wi.get_whale_boost(symbol, direction, minutes_back=15)
                    whale_boost = boost_data['boost']
                    whale_confidence_add = boost_data['confidence_add']
                    whale_agree = boost_data['agree']
                    whale_reason = boost_data['reason']
                    
                    # Apply confidence adjustment (on top of weighted signal)
                    if whale_confidence_add != 0:
                        confidence = min(0.95, max(0.4, confidence + whale_confidence_add))
                    
                    # Apply whale boost to position size
                    size_pct = size_pct * whale_boost
                    size_pct = min(size_pct, self.MAX_SIZE_PCT * 1.5)  # Allow extra for strong whale agreement
                    
            except Exception as e:
                logger.debug(f"Whale boost unavailable: {e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 6. EXTRACT VENUE PRICES
        # ═══════════════════════════════════════════════════════════════════════
        binance_price = 0.0
        bybit_price = 0.0
        coinbase_price = 0.0
        
        if hasattr(data, 'spot_prices') and data.spot_prices:
            binance_price = data.spot_prices.get('binance', 0) or 0
            bybit_price = data.spot_prices.get('bybit', 0) or 0
            coinbase_price = data.spot_prices.get('coinbase', 0) or 0
        
        # Calculate venue agreement (how aligned are the prices)
        venue_prices = [p for p in [binance_price, bybit_price, coinbase_price] if p > 0]
        venue_agreement = 0.0
        if len(venue_prices) >= 2:
            avg_price = sum(venue_prices) / len(venue_prices)
            max_deviation = max(abs(p - avg_price) / avg_price for p in venue_prices) if avg_price > 0 else 0
            venue_agreement = max(0, 1 - max_deviation * 100)  # 1 = perfect agreement
        
        # ═══════════════════════════════════════════════════════════════════════
        # BUILD FULL SIGNAL WITH ALL CONTEXT
        # ═══════════════════════════════════════════════════════════════════════
        return EntrySignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            timeframe=timeframe,
            orderbook_signal=ob_signal,
            momentum_signal=mom_signal,
            ml_signal=ml_signal,
            entry_price=entry_price,
            recommended_size_pct=size_pct,
            reasons=reasons,
            # Market data
            poly_yes_price=data.poly_yes_price,
            poly_no_price=data.poly_no_price,
            spot_price=data.avg_spot_price,
            market_spread=abs(data.poly_yes_price - data.poly_no_price),
            volume_24h=getattr(data, 'volume', 0) or 0,
            liquidity=getattr(data, 'liquidity', 0) or 0,
            # Venue prices
            binance_price=binance_price,
            bybit_price=bybit_price,
            coinbase_price=coinbase_price,
            venue_agreement=venue_agreement,
            # Orderbook details
            ob_bid_depth=ob_bid_depth,
            ob_ask_depth=ob_ask_depth,
            ob_top_bid=ob_top_bid,
            ob_top_ask=ob_top_ask,
            ob_weighted_mid=ob_weighted_mid,
            # Momentum details
            momentum_1min=momentum_1min,
            momentum_5min=momentum_5min,
            momentum_15min=momentum_15min,
            momentum_1h=momentum_1h,
            momentum_trend=momentum_trend,
            # Signal analysis
            signal_raw_combined=signal_raw_combined,
            signal_weights=_json.dumps(signal_weights),
            # Wall Street dump mode
            is_dump_mode=is_dump_mode,
            ws_reason=ws_reason,
            # Whale Intelligence
            whale_signal=whale_signal,
            whale_strength=whale_strength,
            whale_boost=whale_boost,
            whale_confidence_add=whale_confidence_add,
            whale_agree=whale_agree,
            whale_reason=whale_reason,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Grok Validation
    # ─────────────────────────────────────────────────────────────────────────
    
    async def validate_with_grok(
        self, 
        signal: EntrySignal,
        market_data: CryptoMarketData,
    ) -> tuple[Optional[float], dict]:
        """
        Get Grok's opinion on the signal using quick_check (fast $0.20 model).
        Returns (grok_signal, grok_details) for detailed logging.
        """
        if not self.grok:
            return None, {}
        
        # Build orderbook and momentum data for enhanced prompt
        orderbook_data = None
        momentum_data = None
        
        if market_data.orderbook:
            orderbook_data = {
                'imbalance': market_data.orderbook.imbalance,
                'bid_depth': getattr(market_data.orderbook, 'total_bid_size', 0),
                'ask_depth': getattr(market_data.orderbook, 'total_ask_size', 0),
            }
        
        if market_data.momentum:
            momentum_data = {
                'change_5min_pct': market_data.momentum.change_5min_pct,
                'change_1h_pct': getattr(market_data.momentum, 'change_1h_pct', 0),
            }
        
        # Use enhanced quick_check
        try:
            result = await self.grok.quick_check(
                symbol=signal.symbol,
                yes_price=market_data.poly_yes_price,
                ml_direction=signal.direction,
                ml_confidence=signal.confidence,
                orderbook_data=orderbook_data,
                momentum_data=momentum_data,
            )
            
            if result:
                self.log.log_grok_call(
                    signal.symbol,
                    ["quick_check"],
                    result.direction,
                    result.confidence,
                    result.reasoning,
                )
                
                # Convert Grok's response to a signal value
                grok_agrees = (result.direction == signal.direction)
                grok_signal = result.confidence if grok_agrees else -result.confidence
                
                # Build detailed Grok info for logging
                import json as _json
                grok_details = {
                    'model_used': result.model_used,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'key_factors': _json.dumps(result.key_factors) if result.key_factors else '',
                    'action': result.action,
                    'urgency': result.urgency,
                    'full_response': result.raw_response[:500],  # Truncate for storage
                    'cost': result.cost_estimate,
                    'agrees': grok_agrees,
                }
                
                return grok_signal, grok_details
                
        except Exception as e:
            logger.warning(f"Grok validation failed: {e}")
        
        return None, {}
    
    async def batch_grok_analysis(
        self, 
        markets: dict[str, CryptoMarketData],
        signals: list[EntrySignal],
        phase: str = "early"
    ) -> dict[str, dict]:
        """
        Batch analyze all 4 coins with Grok in a single call.
        Returns dict of {symbol: {direction, confidence, reasoning}}
        """
        if not self.grok:
            return {}
        
        # Build a combined prompt for all coins
        coins_info = []
        for signal in signals:
            market = markets.get(signal.symbol)
            if not market:
                continue
            coins_info.append({
                "symbol": signal.symbol,
                "yes_price": market.poly_yes_price,
                "no_price": market.poly_no_price,
                "spot_price": market.avg_spot_price,
                "our_signal": signal.direction,
                "our_confidence": signal.confidence,
                "orderbook_bias": signal.orderbook_signal,
            })
        
        if not coins_info:
            return {}
        
        # Build batch prompt
        timing = "EARLY (first 60s - odds are still forming)" if phase == "early" else "LATE (last 30s - final decision)"
        
        prompt = f"""You are analyzing 4 crypto 15-minute UP/DOWN prediction markets. This is the {timing} phase.

Current market data:
"""
        for coin in coins_info:
            prompt += f"""
{coin['symbol']}:
  - Polymarket YES: {coin['yes_price']:.1%}, NO: {coin['no_price']:.1%}
  - Spot price: ${coin['spot_price']:,.2f}
  - Our signal: {coin['our_signal']} ({coin['our_confidence']:.0%} conf)
  - Orderbook bias: {coin['orderbook_bias']:+.2f} (-1=bearish, +1=bullish)
"""
        
        prompt += """
For each coin, provide your prediction in this EXACT format:
BTC: UP/DOWN, confidence%, brief reason
ETH: UP/DOWN, confidence%, brief reason  
SOL: UP/DOWN, confidence%, brief reason
XRP: UP/DOWN, confidence%, brief reason

Be decisive. Consider orderbook flow, market sentiment, and whether the current odds offer value."""

        try:
            # Use GrokProvider's HTTP client for API call
            await self.grok._ensure_client()
            
            resp = await self.grok.client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.grok.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-4-1-fast-reasoning",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                }
            )
            
            if resp.status_code != 200:
                logger.warning(f"Batch Grok HTTP error: {resp.status_code}")
                return {}
            
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            print(f"\n🤖 Grok Batch ({phase.upper()}): {raw[:200]}...")
            
            # Parse response
            results = {}
            for line in raw.split('\n'):
                line = line.strip()
                for sym in ["BTC", "ETH", "SOL", "XRP"]:
                    if line.upper().startswith(sym):
                        parts = line.split(',')
                        if len(parts) >= 2:
                            # Extract direction
                            direction = "UP" if "UP" in parts[0].upper() else "DOWN"
                            # Extract confidence
                            conf_match = [p for p in parts if '%' in p]
                            conf = 0.6
                            if conf_match:
                                try:
                                    conf = float(conf_match[0].replace('%', '').strip()) / 100
                                except:
                                    pass
                            reason = parts[-1].strip() if len(parts) > 2 else ""
                            results[sym] = {
                                "direction": direction,
                                "confidence": min(conf, 0.85),
                                "reasoning": reason,
                            }
                            print(f"   {sym}: {direction} ({conf:.0%}) - {reason[:50]}")
            
            self.log.system(f"Grok batch ({phase}): {len(results)} predictions")
            return results
            
        except Exception as e:
            logger.warning(f"Batch Grok failed: {e}")
            return {}
    
    def get_window_info(self, timeframe: str = "15M") -> tuple[int, int, int]:
        """
        Get current window info for any timeframe.
        Returns (window_number, seconds_into_window, window_duration)
        """
        now = datetime.now(timezone.utc)
        window_secs = TIMEFRAME_WINDOWS.get(timeframe, 900)
        
        if timeframe == "15M":
            # 15-minute windows start at :00, :15, :30, :45
            minutes_into_hour = now.minute
            window_in_hour = minutes_into_hour // 15
            window_start_minute = window_in_hour * 15
            seconds_into_window = (minutes_into_hour - window_start_minute) * 60 + now.second
            window_number = now.hour * 4 + window_in_hour
        elif timeframe == "1H":
            # Hourly windows start at :00
            seconds_into_window = now.minute * 60 + now.second
            window_number = now.hour
        elif timeframe == "4H":
            # 4-hour windows start at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
            window_in_day = now.hour // 4
            window_start_hour = window_in_day * 4
            seconds_into_window = (now.hour - window_start_hour) * 3600 + now.minute * 60 + now.second
            window_number = window_in_day
        else:
            # Default to 15M
            minutes_into_hour = now.minute
            window_in_hour = minutes_into_hour // 15
            window_start_minute = window_in_hour * 15
            seconds_into_window = (minutes_into_hour - window_start_minute) * 60 + now.second
            window_number = now.hour * 4 + window_in_hour
        
        return window_number, seconds_into_window, window_secs
    
    def get_15m_window_info(self) -> tuple[int, int]:
        """Legacy method - returns 15M window info for backward compatibility."""
        window_num, secs_in, _ = self.get_window_info("15M")
        return window_num, secs_in
    
    def should_call_grok_batch(self) -> tuple[bool, str]:
        """
        Determine if we should make a batch Grok call based on timing.
        Returns (should_call, phase) where phase is 'early' or 'late'
        """
        if not self.use_grok:
            return False, ""
        
        window_num, secs = self.get_15m_window_info()
        
        # New window started - reset flags
        if window_num != self.current_15m_window:
            self.current_15m_window = window_num
            self.grok_early_done = False
            self.grok_late_done = False
            print(f"\n📊 New 15M window #{window_num} started")
        
        # Early window: first 60 seconds
        if secs <= self.GROK_EARLY_WINDOW and not self.grok_early_done:
            self.grok_early_done = True
            return True, "early"
        
        # Late window: last 30 seconds (seconds 870-900)
        time_remaining = 900 - secs
        if time_remaining <= self.GROK_LATE_WINDOW and not self.grok_late_done:
            self.grok_late_done = True
            return True, "late"
        
        return False, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Trading
    # ─────────────────────────────────────────────────────────────────────────
    
    def can_trade(self, symbol: str, confidence: float = 0.5, timeframe: str = "15M", window_number: int = None) -> tuple[bool, str]:
        """Check if we can open a new trade. STRICT rate limiting to prevent runaway."""
        now = datetime.now(timezone.utc)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STRICT WINDOW TRACKING: Only 1 trade per symbol+timeframe+window!
        # Exception: ULTRA_HIGH_CONFIDENCE (75%+) can open 2nd trade
        # ═══════════════════════════════════════════════════════════════════════
        if window_number is not None:
            window_key = f"{symbol}/{timeframe}/{window_number}"
            if not hasattr(self, 'traded_windows'):
                self.traded_windows = {}  # dict: key -> count
            
            current_count = self.traded_windows.get(window_key, 0)
            
            # STRICT: Max 1 trade per window normally
            if current_count >= 1 and confidence < self.ULTRA_HIGH_CONFIDENCE:
                return False, f"Already traded {symbol}/{timeframe} this window"
            
            # Absolute max 2 even with ultra confidence
            if current_count >= 2:
                return False, f"Max 2 trades for {symbol}/{timeframe} in window {window_number}"
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Minimum time between trades (prevents runaway loops)
        # Per-timeframe tracking to allow concurrent trading across TFs
        # ═══════════════════════════════════════════════════════════════════════
        tf_recent = getattr(self, f'recent_trades_{timeframe}', [])
        if tf_recent:
            last_trade = max(tf_recent)
            seconds_since_last = (now - last_trade).total_seconds()
            min_seconds = getattr(self, 'MIN_SECONDS_BETWEEN_TRADES', 60)
            if seconds_since_last < min_seconds:
                return False, f"Too soon ({seconds_since_last:.0f}s < {min_seconds}s min for {timeframe})"
        
        # Rate limit per hour (global to prevent overall runaway)
        hour_ago = now - timedelta(hours=1)
        self.recent_trades = [t for t in self.recent_trades if t > hour_ago]
        
        if len(self.recent_trades) >= self.MAX_TRADES_PER_HOUR:
            return False, f"Rate limited ({len(self.recent_trades)}/{self.MAX_TRADES_PER_HOUR}/hr)"
        
        # Position limit
        if len(self.positions) >= self.MAX_POSITIONS:
            return False, f"Max positions ({self.MAX_POSITIONS})"
        
        # Count positions for this symbol+timeframe combo
        symbol_tf_key = f"{symbol}/{timeframe}"
        symbol_tf_positions = sum(1 for k in self.positions if k.startswith(symbol_tf_key))
        
        # Allow MAX_POSITIONS_PER_SYMBOL per symbol-timeframe (e.g., 3 BTC/15M + 3 BTC/1H OK)
        if symbol_tf_positions >= self.MAX_POSITIONS_PER_SYMBOL:
            return False, f"Max {symbol_tf_key} positions ({symbol_tf_positions})"
        
        # If already have one position for this symbol+timeframe, need higher confidence
        if symbol_tf_positions >= 1 and confidence < self.ULTRA_HIGH_CONFIDENCE:
            return False, f"Need {self.ULTRA_HIGH_CONFIDENCE:.0%} conf for 2nd {symbol_tf_key} position"
        
        # Capital check
        if self.capital < 10:
            return False, "Insufficient capital"
        
        return True, "OK"
    
    def open_position(
        self,
        signal: EntrySignal,
        grok_signal: Optional[float] = None,
        grok_details: dict = None,
    ) -> Optional[int]:
        """Open a paper position with FULL detailed logging."""
        import json as _json
        
        # Get timeframe and window for rate limiting
        timeframe = getattr(signal, 'timeframe', '15M')
        window_num, secs_in, window_duration = self.get_window_info(timeframe)
        
        can, reason = self.can_trade(signal.symbol, signal.confidence, timeframe, window_num)
        if not can:
            self.log.system(f"Cannot trade {signal.symbol}: {reason}")
            return None
        
        # Mark this window as traded IMMEDIATELY - increment counter
        if not hasattr(self, 'traded_windows'):
            self.traded_windows = {}  # dict: key -> count
        window_key = f"{signal.symbol}/{timeframe}/{window_num}"
        self.traded_windows[window_key] = self.traded_windows.get(window_key, 0) + 1
        
        # Calculate size DYNAMICALLY based on confidence
        if signal.confidence >= self.ULTRA_HIGH_CONFIDENCE:
            size_pct = self.ULTRA_SIZE_PCT
        elif signal.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            size_pct = self.MAX_SIZE_PCT
        else:
            # Scale between base and max
            scale = (signal.confidence - self.MIN_CONFIDENCE) / (self.HIGH_CONFIDENCE_THRESHOLD - self.MIN_CONFIDENCE)
            size_pct = self.BASE_SIZE_PCT + scale * (self.MAX_SIZE_PCT - self.BASE_SIZE_PCT)
        
        # Apply timeframe-specific size multiplier (smaller for longer TF = more risk)
        tf_multiplier = self.POSITION_SIZE_BY_TF.get(timeframe, 1.0)
        size_pct = size_pct * tf_multiplier
        
        # ═══════════════════════════════════════════════════════════════════════
        # WALL STREET DUMP MODE BOOST - 2x position size for DOWN trades!
        # ═══════════════════════════════════════════════════════════════════════
        dump_boost_applied = False
        if getattr(signal, 'is_dump_mode', False) and signal.direction == "DOWN":
            size_pct = size_pct * self.WALL_STREET_SIZE_BOOST
            dump_boost_applied = True
            self.log.system(f"🔥 DUMP MODE BOOST: {signal.symbol} DOWN - Position size 2x!")
        
        size_usd = self.capital * size_pct
        
        # Apply appropriate max size cap
        if dump_boost_applied:
            size_usd = min(size_usd, self.capital * self.WALL_STREET_MAX_SIZE_PCT)  # 50% max during dump
        else:
            size_usd = min(size_usd, self.capital * 0.40)  # 40% max normally
        
        if size_usd < 5:
            return None
        
        # Window info already calculated above
        time_until_end = window_duration - secs_in
        
        # Determine primary entry reason
        entry_reasons = []
        if abs(signal.orderbook_signal) >= 0.85:
            entry_reasons.append("strong_orderbook")
        if abs(signal.momentum_signal) >= 0.5:
            entry_reasons.append("momentum")
        if abs(signal.ml_signal) >= 0.3:
            entry_reasons.append("ml_signal")
        if grok_signal and grok_signal > 0:
            entry_reasons.append("grok_confirmed")
        entry_reason = "|".join(entry_reasons) if entry_reasons else "combined_signals"
        
        # Create trade record with FULL signal data for analysis
        trade = PaperTrade(
            symbol=signal.symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction=signal.direction,
            entry_price=signal.entry_price,
            size_usd=size_usd,
            confidence=signal.confidence,
            ml_confidence=signal.ml_signal or 0.0,
            grok_used=grok_signal is not None,
            grok_agreed=grok_signal is not None and grok_signal > 0,
            # Detailed signal data
            orderbook_signal=signal.orderbook_signal,
            momentum_signal=signal.momentum_signal,
            window_number=window_num,
            secs_into_window=secs_in,
            # Enhanced Polymarket data
            poly_yes_price=signal.poly_yes_price,
            poly_no_price=signal.poly_no_price,
            spot_price=signal.spot_price,
            timeframe=timeframe,
            # ═══════════════════════════════════════════════════════════════════
            # ENHANCED LOGGING - Full context
            # ═══════════════════════════════════════════════════════════════════
            market_spread=signal.market_spread,
            volume_24h=signal.volume_24h,
            liquidity=signal.liquidity,
            # Multi-venue prices
            binance_price=signal.binance_price,
            bybit_price=signal.bybit_price,
            coinbase_price=signal.coinbase_price,
            venue_agreement=signal.venue_agreement,
            # Orderbook details
            ob_bid_depth=signal.ob_bid_depth,
            ob_ask_depth=signal.ob_ask_depth,
            ob_top_bid=signal.ob_top_bid,
            ob_top_ask=signal.ob_top_ask,
            ob_weighted_mid=signal.ob_weighted_mid,
            # Momentum details
            momentum_1min=signal.momentum_1min,
            momentum_5min=signal.momentum_5min,
            momentum_15min=signal.momentum_15min,
            momentum_1h=signal.momentum_1h,
            momentum_trend=signal.momentum_trend,
            # Signal analysis
            signal_raw_combined=signal.signal_raw_combined,
            signal_weights=signal.signal_weights,
            signal_reasons=_json.dumps(signal.reasons) if signal.reasons else "",
            # Grok analysis (from grok_details dict)
            grok_model_used=grok_details.get('model_used', '') if grok_details else '',
            grok_confidence=grok_details.get('confidence', 0.0) if grok_details else abs(grok_signal) if grok_signal else 0.0,
            grok_reasoning=grok_details.get('reasoning', '') if grok_details else '',
            grok_key_factors=grok_details.get('key_factors', '') if grok_details else '',
            grok_action=grok_details.get('action', '') if grok_details else '',
            grok_urgency=grok_details.get('urgency', '') if grok_details else '',
            grok_full_response=grok_details.get('full_response', '') if grok_details else '',
            grok_cost=grok_details.get('cost', 0.0) if grok_details else 0.0,
            # Entry timing
            time_until_window_end=time_until_end,
            entry_reason=entry_reason,
        )
        
        trade_id = self.store.save_trade(trade)
        
        now = datetime.now(timezone.utc)
        secs_remaining = window_duration - secs_in
        window_end = now + timedelta(seconds=secs_remaining)
        
        # Track position with unique key (symbol_timeframe_tradeId for multiple positions)
        position_key = f"{signal.symbol}_{timeframe}_{trade_id}"
        self.positions[position_key] = {
            "trade_id": trade_id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "size_usd": size_usd,
            "confidence": signal.confidence,
            "opened_at": now,
            "timeframe": timeframe,
            "window_number": window_num,
            "window_duration": window_duration,
            "window_end": window_end,
            "secs_remaining": secs_remaining,
            # Track peak for trailing stop
            "peak_price": signal.entry_price,
            "entry_reason": entry_reason,
        }
        
        # Update capital
        self.capital -= size_usd
        self.recent_trades.append(datetime.now(timezone.utc))
        
        # Also track per-timeframe for rate limiting
        tf_attr = f'recent_trades_{timeframe}'
        if not hasattr(self, tf_attr):
            setattr(self, tf_attr, [])
        getattr(self, tf_attr).append(datetime.now(timezone.utc))
        
        self.trade_count += 1
        
        # Log with confidence level indicator and timeframe
        conf_emoji = "🔥" if signal.confidence >= self.ULTRA_HIGH_CONFIDENCE else "💰"
        grok_emoji = "🤖" if grok_signal and grok_signal > 0 else ""
        ob_emoji = "📊" if abs(signal.orderbook_signal) >= 0.85 else ""
        
        self.log.log_trade_entry(
            signal.symbol, signal.direction, 
            signal.entry_price, size_usd, trade_id
        )
        print(f"\n{conf_emoji}{grok_emoji}{ob_emoji} [{signal.symbol}/{timeframe}] ENTRY #{trade_id}: {signal.direction} @ {signal.entry_price:.3f} (${size_usd:.2f}, {signal.confidence:.0%} conf)")
        print(f"    📋 Reason: {entry_reason} | OB: {signal.orderbook_signal:+.2f} | Mom: {signal.momentum_5min:+.2f}% | Time left: {time_until_end}s")
        
        return trade_id
    
    def close_position(self, position_key: str, actual_outcome: str) -> float:
        """Close a position and calculate PnL."""
        if position_key not in self.positions:
            return 0.0
        
        pos = self.positions[position_key]
        symbol = pos["symbol"]
        
        # Calculate PnL
        was_correct = (pos["direction"] == actual_outcome)
        
        if was_correct:
            # Won: profit = size * (1.0 - entry_price) / entry_price
            pnl = pos["size_usd"] * (1.0 - pos["entry_price"]) / pos["entry_price"]
            self.wins += 1
        else:
            # Lost
            pnl = -pos["size_usd"]
            self.losses += 1
        
        # Update capital
        self.capital += pos["size_usd"] + pnl
        self.total_pnl += pnl
        
        # Track in history for stats
        self.trade_history.append({
            "trade_id": pos["trade_id"],
            "symbol": symbol,
            "direction": pos["direction"],
            "outcome": actual_outcome,
            "won": was_correct,
            "entry_price": pos["entry_price"],
            "size_usd": pos["size_usd"],
            "pnl": pnl,
            "confidence": pos.get("confidence", 0),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        
        # Close in database
        self.store.close_trade(pos["trade_id"], 1.0 if was_correct else 0.0, actual_outcome)
        
        # Remove position
        del self.positions[position_key]
        
        # Log with clear WIN/LOSS
        result_emoji = "✅ WIN" if was_correct else "❌ LOSS"
        print(f"\n{result_emoji} [{symbol}] #{pos['trade_id']}: Bet {pos['direction']}, Result {actual_outcome} | PnL: ${pnl:+.2f}")
        self.log.log_trade_exit(
            symbol, pos["trade_id"],
            1.0 if was_correct else 0.0,
            pnl, was_correct
        )
        
        return pnl
    
    def early_exit_position(self, position_key: str, current_price: float, exit_reason: str) -> float:
        """
        Exit a position early before window expiry.
        
        Args:
            position_key: The position to close
            current_price: Current Polymarket price for our side
            exit_reason: Why we're exiting (stop-loss, take-profit, ob-reversal, trailing-stop)
        
        Returns:
            PnL from the early exit
        """
        if position_key not in self.positions:
            return 0.0
        
        pos = self.positions[position_key]
        symbol = pos["symbol"]
        entry_price = pos["entry_price"]
        size_usd = pos["size_usd"]
        timeframe = pos.get("timeframe", "15M")
        
        # Calculate actual PnL based on price movement
        # If we bought at 0.50 and now it's 0.60, we're up 20%
        # If we bought at 0.50 and now it's 0.40, we're down 20%
        price_change_pct = (current_price - entry_price) / entry_price
        
        # PnL = size * price_change (simplified - not waiting for binary outcome)
        pnl = size_usd * price_change_pct
        
        # Update stats based on whether we made money
        if pnl > 0:
            self.wins += 1
            was_correct = True
        else:
            self.losses += 1
            was_correct = False
        
        # Update capital
        self.capital += size_usd + pnl
        self.total_pnl += pnl
        
        # Track in history
        self.trade_history.append({
            "trade_id": pos["trade_id"],
            "symbol": symbol,
            "direction": pos["direction"],
            "outcome": f"EARLY_EXIT:{exit_reason}",
            "won": was_correct,
            "entry_price": entry_price,
            "exit_price": current_price,
            "size_usd": size_usd,
            "pnl": pnl,
            "confidence": pos.get("confidence", 0),
            "exit_reason": exit_reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        
        # Close in database with early exit marker
        self.store.close_trade(pos["trade_id"], current_price, f"EARLY:{exit_reason}")
        
        # Remove position
        del self.positions[position_key]
        
        # Log with clear reason
        emoji = "🛑" if "stop" in exit_reason.lower() else "💰" if "profit" in exit_reason.lower() else "⚠️"
        print(f"\n{emoji} EARLY EXIT [{symbol}/{timeframe}] #{pos['trade_id']}: {exit_reason}")
        print(f"   Entry: {entry_price:.3f} → Exit: {current_price:.3f} ({price_change_pct:+.1%}) | PnL: ${pnl:+.2f}")
        
        self.log.system(f"EARLY EXIT #{pos['trade_id']} {symbol}/{timeframe}: {exit_reason} | PnL ${pnl:+.2f}")
        
        return pnl
    
    async def check_early_exits(self, markets: dict[str, 'CryptoMarketData']):
        """
        Check all open positions for early exit conditions:
        1. Stop-loss: Price moved against us too much
        2. Take-profit: Price moved in our favor enough
        3. Orderbook reversal: Market sentiment flipped
        4. Trailing stop: We were up but now giving back gains
        """
        if not self.positions:
            return
        
        exits_triggered = []
        
        for position_key, pos in list(self.positions.items()):
            symbol = pos["symbol"]
            direction = pos["direction"]
            entry_price = pos["entry_price"]
            timeframe = pos.get("timeframe", "15M")
            
            # Get current market data
            market = markets.get(symbol)
            if not market:
                continue
            
            # Current price for our side
            if direction == "UP":
                current_price = market.poly_yes_price
            else:
                current_price = market.poly_no_price
            
            # Calculate price change from entry
            price_change_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Track peak price for trailing stop
            if "peak_price" not in pos:
                pos["peak_price"] = current_price
            else:
                pos["peak_price"] = max(pos["peak_price"], current_price)
            
            peak_price = pos["peak_price"]
            drawdown_from_peak = (peak_price - current_price) / peak_price if peak_price > 0 else 0
            
            exit_reason = None
            
            # ═══════════════════════════════════════════════════════════════
            # CHECK 1: STOP-LOSS
            # ═══════════════════════════════════════════════════════════════
            if price_change_pct <= -self.STOP_LOSS_PCT:
                exit_reason = f"STOP-LOSS ({price_change_pct:+.1%})"
            
            # ═══════════════════════════════════════════════════════════════
            # CHECK 2: TAKE-PROFIT
            # ═══════════════════════════════════════════════════════════════
            elif price_change_pct >= self.TAKE_PROFIT_PCT:
                exit_reason = f"TAKE-PROFIT ({price_change_pct:+.1%})"
            
            # ═══════════════════════════════════════════════════════════════
            # CHECK 3: TRAILING STOP
            # ═══════════════════════════════════════════════════════════════
            elif price_change_pct >= self.TRAILING_STOP_ACTIVATION and drawdown_from_peak >= self.TRAILING_STOP_DISTANCE:
                exit_reason = f"TRAILING-STOP (peak {peak_price:.3f}, now {current_price:.3f})"
            
            # ═══════════════════════════════════════════════════════════════
            # CHECK 4: ORDERBOOK REVERSAL
            # ═══════════════════════════════════════════════════════════════
            if market.orderbook and not exit_reason:
                ob_imbalance = market.orderbook.imbalance
                
                # If we're UP and orderbook strongly negative, or DOWN and strongly positive
                if direction == "UP" and ob_imbalance <= -self.OB_REVERSAL_THRESHOLD:
                    exit_reason = f"OB-REVERSAL (imbalance {ob_imbalance:+.2f} against UP)"
                elif direction == "DOWN" and ob_imbalance >= self.OB_REVERSAL_THRESHOLD:
                    exit_reason = f"OB-REVERSAL (imbalance {ob_imbalance:+.2f} against DOWN)"
            
            # Execute early exit if triggered
            if exit_reason:
                exits_triggered.append((position_key, current_price, exit_reason))
        
        # Execute all triggered exits
        total_exit_pnl = 0.0
        for position_key, current_price, exit_reason in exits_triggered:
            pnl = self.early_exit_position(position_key, current_price, exit_reason)
            total_exit_pnl += pnl
        
        if exits_triggered:
            print(f"\n🔄 EARLY EXITS: {len(exits_triggered)} positions | PnL: ${total_exit_pnl:+.2f}")
    
    async def check_btc_circuit_breaker(self, markets: dict[str, 'CryptoMarketData']) -> int:
        """
        BTC Circuit Breaker - Exit losing positions when BTC makes a big move.
        
        Conservative approach:
        - Track BTC spot price every scan (15 seconds)
        - If BTC moves >0.5% in 1 min → ALERT mode
        - If BTC moves >1.0% in 2 min → EXIT losing positions in opposite direction
        - Only exits positions that are already in the red (eliminates false positives)
        
        Returns: Number of positions exited
        """
        # Get current BTC price from Binance
        btc_market = markets.get("BTC")
        if not btc_market:
            return 0
        
        # Get Binance price (most reliable)
        btc_price = 0.0
        if hasattr(btc_market, 'spot_prices') and btc_market.spot_prices:
            btc_price = btc_market.spot_prices.get('binance', 0) or 0
        if not btc_price:
            btc_price = btc_market.avg_spot_price or 0
        
        if btc_price <= 0:
            return 0
        
        now = datetime.now(timezone.utc)
        
        # Add current price to history
        self.btc_price_history.append((now, btc_price))
        
        # Trim history to keep only last 2 minutes
        cutoff = now - timedelta(seconds=self.BTC_HISTORY_SECONDS)
        self.btc_price_history = [(t, p) for t, p in self.btc_price_history if t >= cutoff]
        
        # Need at least 2 data points
        if len(self.btc_price_history) < 2:
            return 0
        
        # Check cooldown (don't trigger again within 5 minutes)
        if self.btc_circuit_breaker_cooldown:
            if now < self.btc_circuit_breaker_cooldown:
                return 0
        
        # Calculate price changes over different windows
        oldest_price = self.btc_price_history[0][1]
        oldest_time = self.btc_price_history[0][0]
        time_span = (now - oldest_time).total_seconds()
        
        if time_span < 30:  # Need at least 30 seconds of data
            return 0
        
        price_change_pct = (btc_price - oldest_price) / oldest_price
        
        # Check 1-minute change for alert
        one_min_ago = now - timedelta(seconds=60)
        one_min_prices = [(t, p) for t, p in self.btc_price_history if t >= one_min_ago]
        if one_min_prices:
            one_min_change = (btc_price - one_min_prices[0][1]) / one_min_prices[0][1]
        else:
            one_min_change = price_change_pct
        
        # Determine if circuit breaker should trigger
        is_alert = abs(one_min_change) >= self.BTC_ALERT_THRESHOLD
        is_exit = abs(price_change_pct) >= self.BTC_EXIT_THRESHOLD
        
        if is_alert and not self.btc_circuit_breaker_triggered:
            btc_direction = "PUMP" if price_change_pct > 0 else "DUMP"
            self.btc_circuit_breaker_triggered = True
            self.btc_circuit_breaker_direction = btc_direction
            print(f"\n⚠️ BTC CIRCUIT BREAKER ALERT: {btc_direction} detected! ({one_min_change:+.2%} in 1min)")
            self.log.system(f"BTC CIRCUIT BREAKER ALERT: {btc_direction} ({one_min_change:+.2%} in 1min, price ${btc_price:,.0f})")
        
        if not is_exit:
            return 0
        
        btc_direction = "PUMP" if price_change_pct > 0 else "DUMP"
        
        # Find losing positions to exit
        # If BTC PUMPS → exit DOWN positions that are in the red
        # If BTC DUMPS → exit UP positions that are in the red
        positions_to_exit = []
        
        for position_key, pos in list(self.positions.items()):
            symbol = pos["symbol"]
            direction = pos["direction"]
            entry_price = pos["entry_price"]
            
            # Get current price
            market = markets.get(symbol)
            if not market:
                continue
            
            if direction == "UP":
                current_price = market.poly_yes_price
            else:
                current_price = market.poly_no_price
            
            # Calculate if position is in the red
            price_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            is_losing = price_change < -0.02  # At least 2% in the red
            
            # Check if position should be exited
            should_exit = False
            if btc_direction == "PUMP" and direction == "DOWN" and is_losing:
                should_exit = True
            elif btc_direction == "DUMP" and direction == "UP" and is_losing:
                should_exit = True
            
            if should_exit:
                positions_to_exit.append((position_key, current_price, symbol, direction, price_change))
        
        if not positions_to_exit:
            return 0
        
        # Execute circuit breaker exits
        print(f"\n🚨🚨🚨 BTC CIRCUIT BREAKER TRIGGERED! 🚨🚨🚨")
        print(f"    BTC {btc_direction}: {price_change_pct:+.2%} in {time_span:.0f}s (${btc_price:,.0f})")
        print(f"    Exiting {len(positions_to_exit)} losing {('DOWN' if btc_direction == 'PUMP' else 'UP')} positions...")
        
        total_pnl = 0.0
        for position_key, current_price, symbol, direction, pos_change in positions_to_exit:
            exit_reason = f"BTC-CIRCUIT-BREAKER ({btc_direction} {price_change_pct:+.1%})"
            pnl = self.early_exit_position(position_key, current_price, exit_reason)
            total_pnl += pnl
            print(f"    💥 EXIT {symbol} {direction} @ {current_price:.3f} ({pos_change:+.1%}) → PnL: ${pnl:+.2f}")
        
        self.log.system(f"BTC CIRCUIT BREAKER EXIT: {len(positions_to_exit)} positions | BTC {btc_direction} {price_change_pct:+.2%} | PnL ${total_pnl:+.2f}")
        
        # Set cooldown
        self.btc_circuit_breaker_cooldown = now + timedelta(minutes=5)
        self.btc_circuit_breaker_triggered = False
        
        print(f"    TOTAL CIRCUIT BREAKER PnL: ${total_pnl:+.2f}")
        print(f"    Cooldown: 5 minutes")
        
        return len(positions_to_exit)
    
    async def check_last_minute_scalps(self, trade: bool = False):
        """
        Check for last-minute scalp opportunities in final 30-45 seconds of windows.
        Only executes trades for HIGH confidence scalps (80%+).
        """
        from last_minute_scalper import scan_for_scalps
        
        # Check timing - only scan when 30-45 seconds left in current 15M window
        window_num, secs_in, window_duration = self.get_window_info("15M")
        secs_remaining = window_duration - secs_in
        
        if not (25 <= secs_remaining <= 50):
            return  # Not in scalp window
        
        try:
            signals = await scan_for_scalps(secs_remaining=secs_remaining)
        except Exception:
            return  # Silently skip on API errors
        
        if not signals:
            return
        
        for sig in signals:
            # ONLY trade high-confidence scalps (80%+)
            if sig.confidence < 0.80:
                self.log.system(f"🎯 Scalp {sig.symbol} {sig.direction}: {sig.confidence:.0%} conf (need 80%+)")
                continue
            
            # Check if we can trade (respects max 2 per pair per window)
            can, reason = self.can_trade(sig.symbol, sig.confidence, "15M", window_num)
            if not can:
                self.log.system(f"🎯 Scalp {sig.symbol}: Cannot trade - {reason}")
                continue
            
            # Convert ScalpSignal to EntrySignal
            entry_signal = EntrySignal(
                symbol=sig.symbol,
                direction=sig.direction,
                confidence=sig.confidence,
                timeframe="15M",
                orderbook_signal=sig.orderbook_imbalance,
                momentum_signal=sig.momentum_5min / 100 if sig.momentum_5min else 0,  # Normalize
                ml_signal=0.0,
                entry_price=sig.entry_price,
                recommended_size_pct=0.15,  # Conservative for scalps
                reasons=["last_minute_scalp", f"grok_deep_analysis"],
                # Enhanced context
                poly_yes_price=sig.poly_yes_price,
                binance_price=sig.binance_price,
                bybit_price=sig.bybit_price,
                coinbase_price=sig.coinbase_price,
                momentum_1min=sig.momentum_1min,
                momentum_5min=sig.momentum_5min,
                momentum_15min=sig.momentum_15min,
            )
            
            if trade:
                trade_id = self.open_position(
                    entry_signal,
                    grok_signal=sig.confidence,  # Scalper already used Grok
                    grok_details={
                        'model_used': 'grok-4-1-fast-reasoning',
                        'reasoning': sig.reasoning,
                        'confidence': sig.confidence,
                    }
                )
                if trade_id:
                    print(f"\n🎯 SCALP ENTRY #{trade_id}: {sig.symbol} {sig.direction} @ {sig.entry_price:.3f} ({sig.confidence:.0%} conf, {secs_remaining}s left)")
    
    async def check_expired_positions(self):
        """
        Smart position resolution using ACTUAL Binance price data:
        1. Check if position's 15M window has ended (window_end passed)
        2. Fetch ACTUAL price from Binance to determine if price went UP or DOWN
        3. Compare window_start price with window_end price for TRUE outcome
        """
        now = datetime.now(timezone.utc)
        to_resolve = []
        
        for position_key, pos in self.positions.items():
            window_end = pos.get("window_end")
            if window_end and now >= window_end + timedelta(seconds=60):
                # Window has ended + 60s buffer for Binance data, time to resolve
                to_resolve.append(position_key)
            elif not window_end:
                # Old format - use age-based check
                age = (now - pos["opened_at"]).total_seconds()
                if age >= 960:  # 16 minutes
                    to_resolve.append(position_key)
        
        if not to_resolve:
            return
        
        print(f"\n📊 Resolving {len(to_resolve)} expired positions using Binance...")
        
        resolved_count = 0
        total_pnl = 0.0
        
        for position_key in to_resolve:
            pos = self.positions.get(position_key)
            if not pos:
                continue
            
            symbol = pos["symbol"]
            window_start = pos.get("window_start")
            window_end = pos.get("window_end")
            
            if not window_start or not window_end:
                # Old format without window times - force resolve after 20 mins
                age = (now - pos["opened_at"]).total_seconds()
                if age >= 1200:
                    actual = "UP" if pos["direction"] == "DOWN" else "DOWN"
                    pnl = self.close_position(position_key, actual)
                    total_pnl += pnl
                    resolved_count += 1
                    self.log.system(f"RESOLVED #{pos['trade_id']}: FORCED (no window times)")
                continue
            
            # FETCH ACTUAL PRICE DIRECTION FROM BINANCE - THE SOURCE OF TRUTH!
            actual_outcome, change_pct = await get_binance_price_direction(symbol, window_start, window_end)
            
            if actual_outcome == 'UNKNOWN':
                # Binance API failed - wait and retry later
                age = (now - pos["opened_at"]).total_seconds()
                if age >= 1200:  # 20 minutes - force resolve
                    # Last resort: use Polymarket as fallback
                    try:
                        markets = await self.intel.get_all_markets()
                        market = markets.get(symbol)
                        if market:
                            actual_outcome = "UP" if market.poly_yes_price >= 0.5 else "DOWN"
                            self.log.system(f"#{pos['trade_id']}: Binance failed, using Polymarket fallback")
                        else:
                            actual_outcome = "UP" if pos["direction"] == "DOWN" else "DOWN"
                            self.log.system(f"#{pos['trade_id']}: No data, forcing as LOSS")
                    except:
                        actual_outcome = "UP" if pos["direction"] == "DOWN" else "DOWN"
                else:
                    continue  # Wait for Binance data
            
            pnl = self.close_position(position_key, actual_outcome)
            total_pnl += pnl
            resolved_count += 1
            
            # Log resolution with REAL price data
            was_win = (pos["direction"] == actual_outcome)
            self.log.system(
                f"RESOLVED #{pos['trade_id']}: {pos['direction']} → {actual_outcome} "
                f"({'WIN' if was_win else 'LOSS'}) | Binance: {change_pct:+.3f}%"
            )
        
        if resolved_count > 0:
            # Show running totals
            wins = sum(1 for t in self.trade_history if t.get("won", False))
            losses = len(self.trade_history) - wins
            winrate = wins / len(self.trade_history) * 100 if self.trade_history else 0
            cumulative_pnl = sum(t.get("pnl", 0) for t in self.trade_history)
            
            print(f"\n{'='*60}")
            print(f"📈 BATCH RESOLUTION: {resolved_count} positions | Session PnL: ${total_pnl:+.2f}")
            print(f"📊 RUNNING STATS: {wins}W/{losses}L ({winrate:.1f}%) | Total PnL: ${cumulative_pnl:+.2f}")
            print(f"💰 CAPITAL: ${self.capital:.2f} (started: ${self.starting_capital:.2f})")
            print(f"{'='*60}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Last-Minute Scalper - High-conviction late-window entries
    # ─────────────────────────────────────────────────────────────────────────
    
    async def check_scalp_opportunities(self, trade: bool = False) -> int:
        """
        Check for last-minute scalp opportunities.
        Currently disabled - main trading loop handles entries.
        """
        # Disabled for now - main scan_once handles entries well
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────────────────
    
    async def scan_once(self, trade: bool = False) -> dict:
        """Run one scan cycle with smart batch Grok timing."""
        self.scan_count += 1
        
        # Detect signals
        signals = await self.detect_signals()
        
        results = {
            "scan": self.scan_count,
            "signals": len(signals),
            "trades": 0,
            "early_exits": 0,
        }
        
        # Check if we should do a batch Grok call
        should_batch, phase = self.should_call_grok_batch()
        grok_batch_results = {}
        markets = {}  # Initialize markets - will fetch if needed
        
        if should_batch and signals:
            markets = await self.intel.get_all_markets()
            grok_batch_results = await self.batch_grok_analysis(markets, signals, phase)
        elif signals:
            # Even without batch, fetch markets for individual Grok calls
            markets = await self.intel.get_all_markets()
        
        # ═══════════════════════════════════════════════════════════════════════
        # CHECK EARLY EXITS - Monitor open positions for stop-loss/take-profit
        # ═══════════════════════════════════════════════════════════════════════
        if self.positions and trade:
            # Fetch fresh market data for all symbols with open positions
            open_symbols = set(pos["symbol"] for pos in self.positions.values())
            open_timeframes = set(pos.get("timeframe", "15M") for pos in self.positions.values())
            
            # Get markets for each timeframe that has positions
            all_markets = {}
            for tf in open_timeframes:
                tf_markets = await self.intel.get_all_markets(timeframe=tf)
                all_markets.update(tf_markets)
            
            # Check for early exit conditions
            positions_before = len(self.positions)
            await self.check_early_exits(all_markets)
            results["early_exits"] = positions_before - len(self.positions)
            
            # Check BTC circuit breaker (exits losing positions on big BTC moves)
            circuit_breaker_exits = await self.check_btc_circuit_breaker(all_markets)
            results["circuit_breaker_exits"] = circuit_breaker_exits
        
        # Show timing info for each timeframe
        for signal in signals:
            self.signal_count += 1
            
            # Get timing for this signal's timeframe
            tf = getattr(signal, 'timeframe', '15M')
            window_num, secs, window_duration = self.get_window_info(tf)
            time_remaining = window_duration - secs
            
            # Log signal with timing and timeframe
            signal_str = "↑" if signal.direction == "UP" else "↓"
            timing_str = f"[{tf}: {secs}s in, {time_remaining}s left]"
            self.log.prediction(
                signal.symbol,
                f"SIGNAL {signal_str} ({signal.confidence:.0%}) "
                f"[OB={signal.orderbook_signal:+.2f} MOM={signal.momentum_signal:+.2f}] {timing_str}",
                {
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "orderbook": signal.orderbook_signal,
                    "momentum": signal.momentum_signal,
                    "ml": signal.ml_signal,
                    "timeframe": tf,
                    "reasons": signal.reasons,
                }
            )
            
            # Trade if enabled and signal strong enough
            if trade and signal.total_signal >= self.MIN_SIGNAL_TO_TRADE:
                grok_signal = None
                grok_details = None
                
                # ═══════════════════════════════════════════════════════════════
                # EDGE FILTERS - Only trade high-edge setups!
                # ═══════════════════════════════════════════════════════════════
                
                ob_strength = abs(signal.orderbook_signal)
                entry_price = signal.entry_price  # This is the poly YES price for direction we're betting
                
                # Filter 1: Orderbook strength check (100% win rate at >= 0.85!)
                if ob_strength < self.MIN_ORDERBOOK_STRENGTH:
                    self.log.system(f"⚙️ Cannot trade {signal.symbol}/{tf}: OB strength {ob_strength:.2f} < {self.MIN_ORDERBOOK_STRENGTH}")
                    continue
                
                # Filter 2: Entry price check (value-based entries) - TIMEFRAME SPECIFIC
                min_entry, max_entry = self.ENTRY_PRICE_RANGES.get(tf, (self.MIN_ENTRY_PRICE, self.MAX_ENTRY_PRICE))
                if entry_price and (entry_price < min_entry or entry_price > max_entry):
                    self.log.system(f"⚙️ Cannot trade {signal.symbol}/{tf}: Entry {entry_price:.2f} outside {min_entry}-{max_entry} range")
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # GROK VALIDATION - Ask Grok for EVERY trade!
                # Cost: ~$0.0002 per call = negligible
                # ═══════════════════════════════════════════════════════════════
                
                # First check if we have batch results
                if signal.symbol in grok_batch_results:
                    grok_pred = grok_batch_results[signal.symbol]
                    grok_agrees = (grok_pred["direction"] == signal.direction)
                    grok_signal = grok_pred["confidence"] if grok_agrees else -grok_pred["confidence"]
                    grok_details = grok_pred
                else:
                    # NO BATCH RESULT - Call Grok directly for this trade!
                    if self.grok and signal.symbol in markets:
                        try:
                            grok_signal, grok_details = await self.validate_with_grok(signal, markets[signal.symbol])
                            if grok_signal is not None:
                                grok_agrees = grok_signal > 0
                                print(f"   🤖 Grok quick_check: {'AGREES' if grok_agrees else 'DISAGREES'} ({abs(grok_signal):.0%})")
                        except Exception as e:
                            logger.warning(f"Grok validation failed: {e}")
                
                # Apply Grok veto/boost if we have a signal
                if grok_signal is not None:
                    grok_agrees = grok_signal > 0
                    
                    # Log Grok's opinion
                    emoji = "✅" if grok_agrees else "⚠️"
                    conf_str = f"{abs(grok_signal):.0%}" if isinstance(grok_signal, float) else str(grok_signal)
                    
                    # If Grok strongly disagrees, skip this coin
                    if grok_signal < -0.4:
                        print(f"   ❌ Grok VETO ({conf_str}) - skipping {signal.symbol}")
                        continue
                    
                    # If Grok agrees, boost confidence
                    if grok_agrees:
                        signal.confidence = min(signal.confidence + 0.08, 0.85)
                
                # ═══════════════════════════════════════════════════════════════
                # STRICT TRADING LOGIC - Only high-edge trades!
                # ═══════════════════════════════════════════════════════════════
                
                should_trade = False
                trade_reason = ""
                
                # Strong orderbook bias (-0.9 or stronger) = likely direction
                strong_orderbook = abs(signal.orderbook_signal) >= 0.90
                
                # Momentum must align with direction (critical filter!)
                momentum_aligned = (
                    (signal.direction == "UP" and signal.momentum_5min > 0.01) or
                    (signal.direction == "DOWN" and signal.momentum_5min < -0.01)
                )
                
                # 1. ULTRA HIGH CONFIDENCE (75%+): Trade without Grok
                if signal.confidence >= self.ULTRA_HIGH_CONFIDENCE and strong_orderbook:
                    should_trade = True
                    trade_reason = f"Ultra conf ({signal.confidence:.0%}) + strong OB"
                
                # 2. GROK AGREES (60%+): Require Grok confirmation for all other trades
                elif grok_signal and grok_signal >= 0.55 and signal.confidence >= self.MIN_CONFIDENCE:
                    if strong_orderbook or momentum_aligned:
                        should_trade = True
                        trade_reason = f"Grok({grok_signal:.0%}) + {'OB' if strong_orderbook else 'Mom'}"
                
                # 3. HIGH CONFIDENCE + ALL SIGNALS ALIGN
                elif signal.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                    if strong_orderbook and momentum_aligned:
                        # Still get Grok check for safety
                        if not grok_signal and self.use_grok:
                            markets = await self.intel.get_all_markets()
                            if signal.symbol in markets:
                                grok_signal = await self.validate_with_grok(signal, markets[signal.symbol])
                        
                        if grok_signal and grok_signal > 0.3:
                            should_trade = True
                            trade_reason = f"High conf + aligned + Grok({grok_signal:.0%})"
                
                # NO MORE WEAK ENTRIES! Every trade needs strong conviction
                
                if should_trade:
                    trade_id = self.open_position(signal, grok_signal)
                    if trade_id:
                        print(f"   💰 TRADE: {trade_reason}")
                        results["trades"] += 1
        
        return results
    
    async def run_scanner(
        self,
        interval: float = 10.0,
        trade: bool = False,
        duration_minutes: int = 60,
    ):
        """Run continuous scanning."""
        self.log.system(
            f"Starting scanner (interval={interval}s, trade={trade}, "
            f"duration={duration_minutes}m)"
        )
        
        end_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.now(timezone.utc) < end_time:
                start = datetime.now(timezone.utc)
                
                # Check for expired positions (15M = 900 seconds + buffer)
                await self.check_expired_positions()
                
                # Check for last-minute scalp opportunities (final 30-45 sec of windows)
                await self.check_last_minute_scalps(trade=trade)
                
                result = await self.scan_once(trade=trade)
                
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                
                # Status line with early exit info
                early_exits_str = f" | Exits: {result.get('early_exits', 0)}" if result.get('early_exits', 0) > 0 else ""
                
                # Get 6-hour rolling stats from DB
                stats_6h = self.store.get_stats_by_hours(6)
                wr_6h = f"{stats_6h['win_rate']:.0%}" if stats_6h['total'] > 0 else "N/A"
                pnl_6h = f"${stats_6h['pnl']:+.2f}" if stats_6h['total'] > 0 else "$0"
                stats_str = f" | 6h: {stats_6h['wins']}W/{stats_6h['losses']}L ({wr_6h}) {pnl_6h}"
                
                status = (
                    f"Scan #{self.scan_count} | "
                    f"Signals: {result['signals']} | "
                    f"Trades: {self.trade_count} | "
                    f"Open: {len(self.positions)}{early_exits_str}{stats_str} | "
                    f"Capital: ${self.capital:.2f}"
                )
                print(f"\r{status}", end="", flush=True)
                
                # Wait for next scan
                wait_time = max(0, interval - elapsed)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
        except KeyboardInterrupt:
            self.log.system("Scanner stopped by user")
        
        print()  # New line after status
        self.print_summary()
    
    def print_summary(self):
        """Print trading summary."""
        print("\n" + "="*60)
        print("TRADING SUMMARY")
        print("="*60)
        print(f"Scans: {self.scan_count}")
        print(f"Signals: {self.signal_count}")
        print(f"Trades: {self.trade_count}")
        
        if self.trade_count > 0:
            win_rate = self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0
            print(f"Win Rate: {win_rate:.1%} ({self.wins}W / {self.losses}L)")
        
        print(f"PnL: ${self.total_pnl:+.2f}")
        print(f"Capital: ${self.capital:.2f} / ${self.starting_capital:.2f}")
        print(f"Return: {(self.capital - self.starting_capital) / self.starting_capital:+.1%}")
        
        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for sym, pos in self.positions.items():
                print(f"  {sym}: {pos['direction']} @ {pos['entry_price']:.3f} "
                      f"(${pos['size_usd']:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Smart Signal Trader")
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--scan', action='store_true', help='Scan for signals only')
    mode.add_argument('--trade', action='store_true', help='Paper trade on signals')
    
    parser.add_argument('--fast', action='store_true', help='Fast mode (5s interval)')
    parser.add_argument('--no-grok', action='store_true', help='Disable Grok')
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    parser.add_argument('--capital', type=float, default=1000, help='Starting capital')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Interval
    interval = 5.0 if args.fast else 15.0
    
    # Run
    async with SmartSignalTrader(
        use_grok=not args.no_grok,
        starting_capital=args.capital,
    ) as trader:
        
        if args.scan:
            await trader.run_scanner(
                interval=interval,
                trade=False,
                duration_minutes=args.duration,
            )
        else:
            await trader.run_scanner(
                interval=interval,
                trade=True,
                duration_minutes=args.duration,
            )


if __name__ == "__main__":
    asyncio.run(main())
