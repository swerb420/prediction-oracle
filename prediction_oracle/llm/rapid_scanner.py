#!/usr/bin/env python3
"""
Rapid Market Scanner - Fast analysis loop for catching entries.

Scans all 4 crypto markets every few seconds:
1. Fetches real-time prices from Polymarket
2. Runs ML prediction
3. Quick Grok check (fast $0.20 model)
4. Identifies entry opportunities
5. Triggers deep analysis only for unusual situations

Usage:
    python rapid_scanner.py                    # Run scanner
    python rapid_scanner.py --interval 3       # 3 second intervals
    python rapid_scanner.py --no-grok          # Skip Grok calls
    python rapid_scanner.py --paper-trade      # Auto-enter positions
"""

import argparse
import asyncio
import logging
import re
import json
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass

import httpx

from real_data_store import get_store, MarketSnapshot
from learning_ml_predictor import LearningMLPredictor
from selective_bettor import SelectiveBettor
from grok_provider import GrokProvider
from paper_trading_engine import PaperTradingEngine
from trading_logger import get_logger

logger = logging.getLogger(__name__)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]
POLYMARKET_BASE = "https://polymarket.com"


@dataclass
class MarketState:
    """Current state of a market."""
    symbol: str
    slug: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    last_update: datetime
    price_change_1m: float = 0.0  # Price change in last minute
    
    @property
    def market_direction(self) -> str:
        return "UP" if self.yes_price > 0.5 else "DOWN"
    
    @property
    def market_confidence(self) -> float:
        return max(self.yes_price, 1 - self.yes_price)


@dataclass
class EntrySignal:
    """A potential entry opportunity."""
    symbol: str
    direction: str
    confidence: float
    ml_confidence: float
    grok_confidence: float
    grok_action: str
    urgency: str
    should_enter: bool
    reasoning: str


class RapidScanner:
    """
    Rapid market scanner for catching entry opportunities.
    
    - Scans every few seconds
    - Uses fast Grok model ($0.20/M)
    - Identifies high-confidence entries
    - Optional auto paper-trading
    """
    
    def __init__(
        self,
        use_grok: bool = True,
        paper_trade: bool = False,
        starting_capital: float = 1000.0,
    ):
        self.store = get_store()
        self.log = get_logger()
        
        # Components
        self.predictor = LearningMLPredictor(store=self.store)
        self.bettor = SelectiveBettor(store=self.store, predictor=self.predictor)
        
        # Grok (optional)
        self.use_grok = use_grok
        self.grok: Optional[GrokProvider] = None
        
        # Paper trading (optional)
        self.paper_trade = paper_trade
        self.engine: Optional[PaperTradingEngine] = None
        if paper_trade:
            self.engine = PaperTradingEngine(
                store=self.store,
                log=self.log,
                starting_capital=starting_capital,
            )
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Market state tracking
        self.market_states: dict[str, MarketState] = {}
        self.price_history: dict[str, list[tuple[datetime, float]]] = {
            s: [] for s in SYMBOLS
        }
        
        # Discovered slugs
        self.slugs: dict[str, str] = {}
        
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        if self.use_grok:
            self.grok = GrokProvider(store=self.store)
            await self.grok.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
        if self.grok:
            await self.grok.__aexit__(*args)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Market Discovery & Fetching
    # ─────────────────────────────────────────────────────────────────────────
    
    async def discover_markets(self) -> dict[str, str]:
        """Discover current 15M market slugs."""
        try:
            resp = await self.client.get(
                f"{POLYMARKET_BASE}/crypto/15M",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            
            patterns = {
                "BTC": r'btc-updown-15m-\d+',
                "ETH": r'eth-updown-15m-\d+',
                "SOL": r'sol-updown-15m-\d+',
                "XRP": r'xrp-updown-15m-\d+',
            }
            
            for symbol, pattern in patterns.items():
                match = re.search(pattern, resp.text.lower())
                if match:
                    self.slugs[symbol] = match.group(0)
            
            return self.slugs
            
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            return self.slugs
    
    async def fetch_market(self, symbol: str) -> Optional[MarketState]:
        """Fetch current market state for a symbol."""
        slug = self.slugs.get(symbol)
        if not slug:
            return None
        
        try:
            resp = await self.client.get(
                f"{POLYMARKET_BASE}/event/{slug}",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            
            if resp.status_code != 200:
                return None
            
            # Extract __NEXT_DATA__
            match = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                resp.text,
                re.DOTALL,
            )
            if not match:
                return None
            
            data = json.loads(match.group(1))
            queries = (
                data.get("props", {})
                .get("pageProps", {})
                .get("dehydratedState", {})
                .get("queries", [])
            )
            
            for q in queries:
                query_key = q.get("queryKey", [])
                if len(query_key) >= 2 and "updown-15m" in str(query_key[1]).lower():
                    event_data = q.get("state", {}).get("data", {})
                    markets = event_data.get("markets", [])
                    
                    if markets:
                        m = markets[0]
                        prices = m.get("outcomePrices", ["0.5", "0.5"])
                        
                        yes_price = float(prices[0]) if prices else 0.5
                        no_price = float(prices[1]) if len(prices) > 1 else 0.5
                        
                        now = datetime.now(timezone.utc)
                        
                        # Calculate price change
                        price_change = 0.0
                        history = self.price_history[symbol]
                        if history:
                            minute_ago = now - timedelta(minutes=1)
                            old_prices = [p for t, p in history if t > minute_ago]
                            if old_prices:
                                price_change = yes_price - old_prices[0]
                        
                        # Update history
                        history.append((now, yes_price))
                        # Keep last 5 minutes
                        cutoff = now - timedelta(minutes=5)
                        self.price_history[symbol] = [(t, p) for t, p in history if t > cutoff]
                        
                        state = MarketState(
                            symbol=symbol,
                            slug=slug,
                            yes_price=yes_price,
                            no_price=no_price,
                            volume=float(m.get("volume", 0)),
                            liquidity=float(m.get("liquidity", 0)),
                            last_update=now,
                            price_change_1m=price_change,
                        )
                        
                        self.market_states[symbol] = state
                        return state
            
            return None
            
        except Exception as e:
            logger.debug(f"Fetch error for {symbol}: {e}")
            return None
    
    async def fetch_all_markets(self) -> list[MarketState]:
        """Fetch all markets in parallel."""
        if not self.slugs:
            await self.discover_markets()
        
        tasks = [self.fetch_market(s) for s in SYMBOLS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        states = []
        for result in results:
            if isinstance(result, MarketState):
                states.append(result)
        
        return states
    
    # ─────────────────────────────────────────────────────────────────────────
    # Analysis & Entry Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    async def analyze_market(self, state: MarketState) -> EntrySignal:
        """Analyze a market and generate entry signal."""
        symbol = state.symbol
        
        # Build market data dict
        market_data = {
            "yes_price": state.yes_price,
            "no_price": state.no_price,
            "market_direction": state.market_direction,
            "volume": state.volume,
            "liquidity": state.liquidity,
            "spread": 0.02,  # Estimate
            "seconds_to_close": 600,  # Estimate
            "timestamp": state.last_update.isoformat(),
        }
        
        # ML prediction
        ml_pred = self.predictor.predict(symbol, market_data)
        
        # Grok quick check (if enabled)
        grok_confidence = 0.5
        grok_action = "WAIT"
        grok_urgency = "wait"
        
        if self.use_grok and self.grok:
            grok_resp = await self.grok.quick_check(
                symbol=symbol,
                yes_price=state.yes_price,
                ml_direction=ml_pred.direction,
                ml_confidence=ml_pred.confidence,
            )
            
            if grok_resp:
                grok_confidence = grok_resp.confidence
                grok_action = grok_resp.action
                grok_urgency = grok_resp.urgency
        
        # Combine signals
        combined_confidence = (ml_pred.confidence + grok_confidence) / 2
        
        # Entry criteria
        should_enter = (
            grok_action == "BUY" and
            grok_urgency in ["immediate", "soon"] and
            combined_confidence >= 0.60 and
            ml_pred.should_bet
        )
        
        # If no Grok, rely on ML + bettor
        if not self.use_grok:
            decision = self.bettor.should_bet(symbol, market_data)
            should_enter = decision.should_bet
            combined_confidence = decision.confidence
        
        # Build reasoning
        if should_enter:
            reasoning = f"ENTRY: ML={ml_pred.direction}({ml_pred.confidence:.0%}), Grok={grok_action}({grok_confidence:.0%})"
        else:
            reasoning = f"WAIT: ML={ml_pred.confidence:.0%}, Grok={grok_action}"
        
        return EntrySignal(
            symbol=symbol,
            direction=ml_pred.direction,
            confidence=combined_confidence,
            ml_confidence=ml_pred.confidence,
            grok_confidence=grok_confidence,
            grok_action=grok_action,
            urgency=grok_urgency,
            should_enter=should_enter,
            reasoning=reasoning,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────────────────
    
    async def scan_cycle(self) -> list[EntrySignal]:
        """Run one scan cycle."""
        # Fetch markets
        states = await self.fetch_all_markets()
        
        if not states:
            logger.warning("No markets available")
            return []
        
        # Analyze all markets in parallel
        tasks = [self.analyze_market(state) for state in states]
        signals = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_signals = []
        for sig in signals:
            if isinstance(sig, EntrySignal):
                valid_signals.append(sig)
                
                # Log entries
                if sig.should_enter:
                    self.log.trade(
                        sig.symbol,
                        f"🎯 ENTRY SIGNAL: {sig.direction} ({sig.confidence:.1%})",
                        {
                            "ml_confidence": sig.ml_confidence,
                            "grok_confidence": sig.grok_confidence,
                            "grok_action": sig.grok_action,
                            "urgency": sig.urgency,
                        }
                    )
                    
                    # Auto paper trade if enabled
                    if self.paper_trade and self.engine:
                        state = self.market_states.get(sig.symbol)
                        if state and not self.engine.has_position(sig.symbol):
                            size = self.engine.calculate_position_size(0.10)
                            entry_price = state.yes_price if sig.direction == "UP" else state.no_price
                            
                            self.engine.enter_position(
                                symbol=sig.symbol,
                                direction=sig.direction,
                                entry_price=entry_price,
                                size_usd=size,
                                confidence=sig.confidence,
                                ml_confidence=sig.ml_confidence,
                                grok_used=self.use_grok,
                                event_slug=state.slug,
                            )
        
        return valid_signals
    
    async def run(self, interval: float = 5.0, max_cycles: Optional[int] = None):
        """
        Run the rapid scanner.
        
        Args:
            interval: Seconds between scans
            max_cycles: Max cycles (None for infinite)
        """
        self.log.system(f"Starting rapid scanner (interval: {interval}s)")
        
        # Initial discovery
        await self.discover_markets()
        self.log.system(f"Discovered {len(self.slugs)} markets")
        
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                
                # Scan
                signals = await self.scan_cycle()
                
                # Display
                self._display_signals(signals, cycle)
                
                # Refresh slugs periodically (every 60 cycles)
                if cycle % 60 == 0:
                    await self.discover_markets()
                
                # Wait
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            self.log.system("Scanner stopped by user")
        
        # Final summary
        if self.paper_trade and self.engine:
            self.engine.print_summary()
    
    def _display_signals(self, signals: list[EntrySignal], cycle: int):
        """Display current signals."""
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        
        # Clear line and print header
        print(f"\r[{now}] Cycle {cycle}: ", end="")
        
        for sig in signals:
            state = self.market_states.get(sig.symbol)
            if not state:
                continue
            
            # Color code
            if sig.should_enter:
                indicator = "🟢"
            elif sig.grok_action == "WAIT":
                indicator = "🟡"
            else:
                indicator = "⚪"
            
            price_str = f"{state.yes_price:.1%}"
            change_str = f"{state.price_change_1m:+.1%}" if state.price_change_1m else ""
            
            print(f"{indicator}{sig.symbol}:{price_str}{change_str} ", end="")
        
        # Grok stats
        if self.use_grok and self.grok:
            stats = self.grok.get_stats()
            print(f"| Grok: {stats['fast_calls_last_min']}/{stats['fast_limit']}/min", end="")
        
        print("", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Rapid 15M Market Scanner")
    parser.add_argument('--interval', type=float, default=5.0, help='Scan interval (seconds)')
    parser.add_argument('--cycles', type=int, default=None, help='Max cycles (default: infinite)')
    parser.add_argument('--no-grok', action='store_true', help='Disable Grok')
    parser.add_argument('--paper-trade', action='store_true', help='Auto paper trade')
    parser.add_argument('--capital', type=float, default=1000.0, help='Starting capital')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    async with RapidScanner(
        use_grok=not args.no_grok,
        paper_trade=args.paper_trade,
        starting_capital=args.capital,
    ) as scanner:
        await scanner.run(
            interval=args.interval,
            max_cycles=args.cycles,
        )


if __name__ == "__main__":
    asyncio.run(main())
