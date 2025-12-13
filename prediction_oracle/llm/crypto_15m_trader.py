#!/usr/bin/env python3
"""
15M Crypto Trader - Master the Polymarket 15-minute crypto bets.

This is the main runner that combines:
1. Real-time 15M market tracking from Polymarket
2. Multi-venue CEX prices for BTC/ETH/SOL/XRP
3. ML predictions trained on whale trading patterns
4. Smart Grok 4.1 triggers (only on unusual signals)
5. Paper trading with entry/exit optimization

Usage:
    python crypto_15m_trader.py --once       # Single prediction cycle
    python crypto_15m_trader.py --monitor    # Continuous monitoring
    python crypto_15m_trader.py --trade      # Paper trade mode
    python crypto_15m_trader.py --stats      # Show trading stats
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

# Add current dir to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all our components
from polymarket_15m_tracker import Polymarket15MTracker, Market15MData
from multi_venue_client import MultiVenueClient
from smart_grok_trigger import SmartGrokTrigger, SignalContext, TriggerReason
from whale_pattern_trainer import WhalePatternTrainer, WhaleFeatures

# These use relative imports internally, so we need to handle them carefully
try:
    from enhanced_paper_trading import EnhancedPaperTradingEngine
    from enhanced_grok_provider import EnhancedGrokProvider
except ImportError:
    # Fallback - create simple stubs
    class EnhancedPaperTradingEngine:
        def __init__(self, **kwargs):
            self.capital = kwargs.get("initial_capital", 10000)
            self.trades = []
        def open_position(self, **kwargs):
            self.trades.append(kwargs)
        def get_stats(self):
            return {"total_trades": len(self.trades), "win_rate": 0, "total_pnl": 0, "current_capital": self.capital}
    
    EnhancedGrokProvider = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Crypto15MTrader:
    """
    Master 15M crypto trader that combines all components.

    Flow:
    1. Get real-time 15M market prices from Polymarket
    2. Fetch CEX prices for actual crypto values
    3. Get ML prediction based on whale patterns
    4. Check if Grok trigger conditions are met
    5. If triggered, call Grok 4.1 for final validation
    6. Execute paper trade if confidence is high enough
    """

    def __init__(
        self,
        use_grok: bool = True,
        min_confidence: float = 0.55,
        position_size: float = 100.0,
    ):
        """
        Initialize the 15M trader.

        Args:
            use_grok: Whether to use Grok 4.1 for validation
            min_confidence: Minimum confidence to place a trade
            position_size: Default position size in USD
        """
        self.use_grok = use_grok
        self.min_confidence = min_confidence
        self.position_size = position_size

        # Components
        self.tracker: Optional[Polymarket15MTracker] = None
        self.venue_client: Optional[MultiVenueClient] = None
        self.grok_trigger = SmartGrokTrigger(
            max_calls_per_hour=10,
            min_seconds_between_calls=60,
        )
        self.whale_trainer = WhalePatternTrainer()
        self.paper_engine = EnhancedPaperTradingEngine(
            initial_capital=10000.0,
            position_size=position_size,
        )

        # Grok provider (optional)
        self.grok_provider = None
        if use_grok and os.getenv("XAI_API_KEY") and EnhancedGrokProvider is not None:
            try:
                self.grok_provider = EnhancedGrokProvider(
                    api_key=os.getenv("XAI_API_KEY"),
                )
                logger.info("Grok 4.1 provider initialized")
            except Exception as e:
                logger.warning(f"Failed to init Grok provider: {e}")

        # State
        self._running = False

    async def __aenter__(self):
        """Initialize async components."""
        self.tracker = Polymarket15MTracker(poll_interval=2.0)
        await self.tracker.__aenter__()

        self.venue_client = MultiVenueClient()

        # Try to load pre-trained whale model
        if self.whale_trainer.load_models():
            logger.info("Loaded pre-trained whale ML models")
        else:
            logger.info("No pre-trained models found - will use raw predictions")

        return self

    async def __aexit__(self, *args):
        """Cleanup."""
        self._running = False
        if self.tracker:
            await self.tracker.__aexit__(*args)

    async def get_market_snapshot(self) -> dict:
        """
        Get current snapshot of all 15M markets and CEX prices.

        Returns:
            dict with market data for each symbol
        """
        snapshot = {}

        # Fetch 15M Polymarket prices
        markets = await self.tracker.fetch_all_markets()

        # Fetch CEX prices
        cex_prices = await self.venue_client.fetch_all_prices()

        for symbol in ["BTC", "ETH", "SOL", "XRP"]:
            market = markets.get(symbol)
            if not market:
                continue

            # Get CEX price
            cex_data = cex_prices.get(symbol, {})
            cex_price = cex_data.get("mid_price", 0) if cex_data else 0

            snapshot[symbol] = {
                "polymarket": {
                    "yes_price": market.yes_price,
                    "no_price": market.no_price,
                    "direction": market.market_direction,
                    "confidence": market.confidence,
                    "question": market.question,
                },
                "cex_price": cex_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return snapshot

    async def analyze_symbol(self, symbol: str, market: Market15MData) -> dict:
        """
        Full analysis for a single symbol.

        Returns prediction with confidence and Grok trigger decision.
        """
        # Build signal context for smart trigger
        ctx = SignalContext(
            asset=symbol,
            ml_direction=market.market_direction,
            ml_confidence=market.confidence,
            whale_direction=market.market_direction,  # Using market as proxy
            whale_confidence=market.confidence,
            polymarket_yes_price=market.yes_price,
            polymarket_spread_bps=market.spread * 10000,
            polymarket_volume_24h=market.volume,
            momentum=self.tracker.get_momentum(symbol, 30),
            volatility=self.tracker.get_volatility(symbol, 60),
            rsi=50.0,  # Placeholder
        )

        # Check if we should trigger Grok
        trigger_decision = self.grok_trigger.evaluate(ctx)

        # Base prediction from market
        prediction = {
            "symbol": symbol,
            "direction": market.market_direction,
            "confidence": market.confidence,
            "source": "polymarket",
            "grok_triggered": False,
            "trigger_reasons": [],
        }

        # If triggered and Grok is available, get validation
        if trigger_decision.should_trigger and self.grok_provider:
            logger.info(f"🎯 GROK TRIGGERED for {symbol}: {trigger_decision.explanation}")
            prediction["grok_triggered"] = True
            prediction["trigger_reasons"] = [r.value for r in trigger_decision.reasons]

            # Call Grok for validation
            try:
                grok_result = await self._call_grok_validation(symbol, market, ctx)
                if grok_result:
                    # Blend Grok's opinion
                    prediction["grok_direction"] = grok_result.get("direction")
                    prediction["grok_confidence"] = grok_result.get("confidence")
                    prediction["grok_reasoning"] = grok_result.get("reasoning")

                    # If Grok disagrees, adjust confidence
                    if grok_result.get("direction") != prediction["direction"]:
                        prediction["confidence"] *= 0.8  # Reduce confidence on disagreement
                        prediction["note"] = "Grok disagrees with market"
            except Exception as e:
                logger.error(f"Grok validation failed: {e}")

        return prediction

    async def _call_grok_validation(
        self,
        symbol: str,
        market: Market15MData,
        ctx: SignalContext,
    ) -> Optional[dict]:
        """Call Grok 4.1 for validation on unusual signals."""
        if not self.grok_provider:
            return None

        prompt = f"""
Crypto 15M Direction Analysis for {symbol}

Current Polymarket 15M Market:
- Question: {market.question}
- UP probability: {market.yes_price:.1%}
- DOWN probability: {market.no_price:.1%}
- Market confidence: {market.confidence:.1%}

Context:
- Momentum (30s): {ctx.momentum:.2f}%
- Volatility: {ctx.volatility:.2f}%

This is an unusual signal because: {', '.join(r.value for r in self.grok_trigger._trigger_stats.keys() if self.grok_trigger._trigger_stats.get(r, 0) > 0)}

Should we trust the market direction ({market.market_direction}) or fade it?

Respond in JSON format:
{{"direction": "UP" or "DOWN", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

        try:
            response = await self.grok_provider.validate_prediction(
                symbol=symbol,
                direction=market.market_direction,
                confidence=market.confidence,
                context={"prompt": prompt},
            )
            return {
                "direction": response.direction,
                "confidence": response.confidence,
                "reasoning": response.reasoning,
            }
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return None

    async def run_once(self) -> dict:
        """
        Run a single prediction cycle.

        Returns predictions for all symbols.
        """
        logger.info("=" * 60)
        logger.info("15M CRYPTO ANALYSIS")
        logger.info("=" * 60)

        # Discover current markets
        await self.tracker.discover_current_markets()

        # Fetch all market data
        markets = await self.tracker.fetch_all_markets()

        predictions = {}
        for symbol, market in markets.items():
            pred = await self.analyze_symbol(symbol, market)
            predictions[symbol] = pred

            # Log result
            grok_flag = "🎯" if pred.get("grok_triggered") else ""
            logger.info(
                f"{symbol}: {pred['direction']} ({pred['confidence']:.1%}) {grok_flag}"
            )
            if pred.get("grok_reasoning"):
                logger.info(f"  Grok: {pred['grok_reasoning'][:50]}...")

        return predictions

    async def run_monitor(self, duration_minutes: int = 60):
        """
        Run continuous monitoring mode.

        Args:
            duration_minutes: How long to run
        """
        logger.info(f"Starting {duration_minutes}-minute monitoring session...")
        self._running = True

        # Discover markets
        await self.tracker.discover_current_markets()

        interval = 15  # seconds between updates
        iterations = (duration_minutes * 60) // interval

        for i in range(iterations):
            if not self._running:
                break

            try:
                # Fetch current prices
                markets = await self.tracker.fetch_all_markets()

                # Log current state
                logger.info("-" * 40)
                for symbol, market in markets.items():
                    momentum = self.tracker.get_momentum(symbol, 30)
                    logger.info(
                        f"{symbol}: {market.yes_price:.1%} UP | "
                        f"{market.no_price:.1%} DOWN | "
                        f"mom={momentum:+.2f}%"
                    )

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)

        logger.info("Monitoring session ended")

    async def run_paper_trade(self, cycles: int = 10, cycle_interval: int = 60):
        """
        Run paper trading mode.

        Places paper trades based on predictions.
        """
        logger.info(f"Starting paper trading ({cycles} cycles, {cycle_interval}s interval)...")
        self._running = True

        # Discover markets
        await self.tracker.discover_current_markets()

        for cycle in range(cycles):
            if not self._running:
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"CYCLE {cycle + 1}/{cycles}")
            logger.info("=" * 60)

            try:
                # Get predictions
                markets = await self.tracker.fetch_all_markets()

                for symbol, market in markets.items():
                    pred = await self.analyze_symbol(symbol, market)

                    # Check if we should trade
                    if pred["confidence"] >= self.min_confidence:
                        # Place paper trade
                        entry_price = market.yes_price if pred["direction"] == "UP" else market.no_price

                        self.paper_engine.open_position(
                            symbol=symbol,
                            direction=pred["direction"],
                            confidence=pred["confidence"],
                            entry_price=entry_price,
                            size=self.position_size,
                            whale_consensus=pred.get("grok_direction"),
                        )

                        logger.info(
                            f"📈 TRADE: {symbol} {pred['direction']} @ {entry_price:.3f} "
                            f"(conf={pred['confidence']:.1%})"
                        )

                # Wait for next cycle
                if cycle < cycles - 1:
                    await asyncio.sleep(cycle_interval)

            except Exception as e:
                logger.error(f"Trade cycle error: {e}")
                await asyncio.sleep(10)

        # Show final stats
        stats = self.paper_engine.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("PAPER TRADING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total trades: {stats.get('total_trades', 0)}")
        logger.info(f"Win rate: {stats.get('win_rate', 0):.1%}")
        logger.info(f"Total PnL: ${stats.get('total_pnl', 0):.2f}")

    def show_stats(self):
        """Show current trading statistics."""
        stats = self.paper_engine.get_stats()
        trigger_stats = self.grok_trigger.get_stats()

        print("\n" + "=" * 60)
        print("TRADING STATISTICS")
        print("=" * 60)

        print("\nPaper Trading:")
        print(f"  Total trades: {stats.get('total_trades', 0)}")
        print(f"  Win rate: {stats.get('win_rate', 0):.1%}")
        print(f"  Total PnL: ${stats.get('total_pnl', 0):.2f}")
        print(f"  Capital: ${stats.get('current_capital', 10000):.2f}")

        print("\nGrok Triggers:")
        print(f"  Total triggers: {trigger_stats.get('total_triggers', 0)}")
        print(f"  Calls last hour: {trigger_stats.get('calls_last_hour', 0)}")
        print(f"  By reason: {trigger_stats.get('triggers_by_reason', {})}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="15M Crypto Trader")
    parser.add_argument("--once", action="store_true", help="Single prediction cycle")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring")
    parser.add_argument("--trade", action="store_true", help="Paper trade mode")
    parser.add_argument("--stats", action="store_true", help="Show trading stats")
    parser.add_argument("--no-grok", action="store_true", help="Disable Grok validation")
    parser.add_argument("--duration", type=int, default=60, help="Monitor duration (minutes)")
    parser.add_argument("--cycles", type=int, default=10, help="Trade cycles")

    args = parser.parse_args()

    async with Crypto15MTrader(use_grok=not args.no_grok) as trader:
        if args.stats:
            trader.show_stats()
        elif args.monitor:
            await trader.run_monitor(args.duration)
        elif args.trade:
            await trader.run_paper_trade(args.cycles)
        else:
            # Default: single prediction cycle
            await trader.run_once()


if __name__ == "__main__":
    asyncio.run(main())
