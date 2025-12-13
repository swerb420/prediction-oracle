#!/usr/bin/env python3
"""
Polymarket 15M Trader - Main Runner Script

NO FAKE DATA. Everything is real.

Modes:
1. --collect: Collect market data and outcomes (builds training set)
2. --predict: Make predictions (uses ML model)
3. --trade: Paper trade with selective betting
4. --monitor: Watch markets in real-time
5. --status: Show system status

Usage:
    python poly_15m_trader.py --collect           # Collect data for ML training
    python poly_15m_trader.py --predict           # Show predictions (no trading)
    python poly_15m_trader.py --trade             # Paper trade
    python poly_15m_trader.py --trade --no-grok   # Trade without Grok
    python poly_15m_trader.py --status            # Show system status
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

# Local imports
from real_data_store import get_store, MarketSnapshot
from outcome_collector import OutcomeCollector
from learning_ml_predictor import LearningMLPredictor
from selective_bettor import SelectiveBettor
from grok_provider import GrokProvider
from paper_trading_engine import PaperTradingEngine
from trading_logger import get_logger

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]


class Poly15MTrader:
    """
    Main orchestrator for 15M crypto trading on Polymarket.
    
    Components:
    - OutcomeCollector: Fetches real market data
    - LearningMLPredictor: Trains on real outcomes
    - SelectiveBettor: Only bets when conditions are right
    - GrokProvider: Optional LLM validation
    - PaperTradingEngine: Tracks positions and PnL
    - TradingLogger: Logs everything
    """
    
    def __init__(
        self,
        use_grok: bool = True,
        starting_capital: float = 1000.0,
    ):
        # Core components
        self.store = get_store()
        self.log = get_logger()
        
        # Collector for market data
        self.collector = OutcomeCollector(store=self.store)
        
        # ML predictor
        self.predictor = LearningMLPredictor(store=self.store)
        
        # Selective bettor
        self.bettor = SelectiveBettor(
            store=self.store,
            predictor=self.predictor,
        )
        
        # Paper trading
        self.engine = PaperTradingEngine(
            store=self.store,
            log=self.log,
            starting_capital=starting_capital,
        )
        
        # Grok provider (optional)
        self.use_grok = use_grok
        self.grok: Optional[GrokProvider] = None
        
    async def __aenter__(self):
        await self.collector.__aenter__()
        if self.use_grok:
            self.grok = GrokProvider(store=self.store)
            await self.grok.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        await self.collector.__aexit__(*args)
        if self.grok:
            await self.grok.__aexit__(*args)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Data Collection Mode
    # ─────────────────────────────────────────────────────────────────────────
    
    async def run_collection(self, cycles: int = 1, interval: int = 60):
        """
        Run data collection mode.
        
        Collects market snapshots and outcomes to build training dataset.
        """
        self.log.system(f"Starting data collection ({cycles} cycles, {interval}s interval)")
        
        for i in range(cycles):
            if cycles > 1:
                self.log.system(f"Collection cycle {i+1}/{cycles}")
            
            result = await self.collector.collect_cycle()
            
            # Summary
            summary = self.store.get_summary()
            self.log.system(
                f"Collection complete: {summary['snapshots']} snapshots, "
                f"{summary['outcomes']} outcomes, "
                f"{summary['labeled_examples']} labeled examples"
            )
            
            if i < cycles - 1:
                await asyncio.sleep(interval)
        
        # Retrain models with new data
        self.log.system("Retraining ML models...")
        self.predictor.train()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Prediction Mode
    # ─────────────────────────────────────────────────────────────────────────
    
    async def run_predictions(self):
        """
        Run prediction mode.
        
        Shows predictions for all symbols without trading.
        """
        self.log.system("Running predictions (no trading)")
        
        # Collect current snapshots
        snapshots = await self.collector.collect_all_snapshots()
        
        if not snapshots:
            self.log.warning("No market snapshots available")
            return
        
        print("\n" + "="*70)
        print("PREDICTIONS")
        print("="*70)
        
        for snapshot in snapshots:
            symbol = snapshot.symbol
            
            # Get market data
            market_data = {
                "yes_price": snapshot.yes_price,
                "no_price": snapshot.no_price,
                "market_direction": snapshot.market_direction,
                "volume": snapshot.volume,
                "liquidity": snapshot.liquidity,
                "timestamp": snapshot.timestamp,
            }
            
            # Get ML prediction
            prediction = self.predictor.predict(symbol, market_data)
            
            # Get betting decision
            decision = self.bettor.should_bet(symbol, market_data)
            
            # Display
            bet_str = "✓ WOULD BET" if decision.should_bet else "✗ WOULD SKIP"
            print(f"\n{symbol}:")
            print(f"  Market: YES={snapshot.yes_price:.1%} → {snapshot.market_direction}")
            print(f"  ML: {prediction.direction} ({prediction.confidence:.1%})")
            print(f"  Training: {prediction.training_examples} examples")
            print(f"  Decision: {bet_str}")
            
            if decision.failed_filters:
                print(f"  Failed: {', '.join(decision.failed_filters)}")
            
            # Log
            self.log.log_snapshot(
                symbol, snapshot.yes_price, snapshot.no_price, 
                snapshot.market_direction
            )
            self.log.log_ml_prediction(
                symbol, prediction.direction, prediction.confidence,
                prediction.should_bet, prediction.reasoning
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Trading Mode
    # ─────────────────────────────────────────────────────────────────────────
    
    async def run_trading(self, cycles: int = 1, interval: int = 300):
        """
        Run trading mode.
        
        Makes predictions, applies filters, and paper trades.
        """
        self.log.system(f"Starting paper trading ({cycles} cycles)")
        
        for i in range(cycles):
            if cycles > 1:
                self.log.system(f"Trading cycle {i+1}/{cycles}")
            
            await self._trading_cycle()
            
            if i < cycles - 1:
                # Wait for next cycle
                self.log.system(f"Waiting {interval}s for next cycle...")
                await asyncio.sleep(interval)
        
        # Final summary
        self.engine.print_summary()
    
    async def _trading_cycle(self):
        """Run one trading cycle."""
        # 1. Collect current market data
        snapshots = await self.collector.collect_all_snapshots()
        
        if not snapshots:
            self.log.warning("No markets available")
            return
        
        # 2. Check for resolved markets (close positions)
        outcomes = await self.collector.check_resolved_markets()
        for outcome in outcomes:
            self.engine.close_by_event_slug(outcome.event_slug, outcome.actual_outcome)
        
        # 3. Process each market
        for snapshot in snapshots:
            await self._process_market(snapshot)
    
    async def _process_market(self, snapshot: MarketSnapshot):
        """Process a single market for trading."""
        symbol = snapshot.symbol
        
        # Skip if we already have a position
        if self.engine.has_position(symbol):
            self.log.system(f"Already have position in {symbol}, skipping")
            return
        
        # Market data
        market_data = {
            "yes_price": snapshot.yes_price,
            "no_price": snapshot.no_price,
            "market_direction": snapshot.market_direction,
            "volume": snapshot.volume,
            "liquidity": snapshot.liquidity,
            "spread": abs(snapshot.yes_price - (1 - snapshot.no_price)),
            "seconds_to_close": 600,  # Assume 10 min
            "timestamp": snapshot.timestamp,
        }
        
        # Log snapshot
        self.log.log_snapshot(
            symbol, snapshot.yes_price, snapshot.no_price,
            snapshot.market_direction, volume=snapshot.volume
        )
        
        # Get ML prediction
        prediction = self.predictor.predict(symbol, market_data)
        
        # Log prediction
        self.log.log_ml_prediction(
            symbol, prediction.direction, prediction.confidence,
            prediction.should_bet, prediction.reasoning,
            training_examples=prediction.training_examples
        )
        
        # Maybe call Grok
        grok_response = None
        if self.use_grok and self.grok:
            ml_pred_dict = {
                "direction": prediction.direction,
                "confidence": prediction.confidence,
                "training_examples": prediction.training_examples,
                "reasoning": prediction.reasoning,
            }
            
            result, reasons = await self.grok.maybe_analyze(
                symbol, market_data, ml_pred_dict
            )
            
            if result:
                grok_response = {
                    "direction": result.direction,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                }
                
                self.log.log_grok_call(
                    symbol, reasons, result.direction,
                    result.confidence, result.reasoning
                )
        
        # Get betting decision
        decision = self.bettor.should_bet(symbol, market_data, grok_response)
        
        # Log decision
        self.log.log_bet_decision(
            symbol, decision.should_bet, decision.direction,
            decision.confidence, decision.position_size_pct,
            decision.passed_filters, decision.failed_filters
        )
        
        # Execute trade if approved
        if decision.should_bet:
            # Calculate position size
            size_usd = self.engine.calculate_position_size(decision.position_size_pct)
            
            if size_usd < 1.0:
                self.log.warning(f"Position too small: ${size_usd:.2f}", symbol=symbol)
                return
            
            # Determine entry price
            if decision.direction == "UP":
                entry_price = snapshot.yes_price
            else:
                entry_price = snapshot.no_price
            
            # Enter position
            trade_id = self.engine.enter_position(
                symbol=symbol,
                direction=decision.direction,
                entry_price=entry_price,
                size_usd=size_usd,
                confidence=decision.confidence,
                ml_confidence=prediction.confidence,
                grok_used=grok_response is not None,
                grok_agreed=decision.grok_agreed,
                event_slug=snapshot.event_slug,
            )
            
            # Save prediction for later evaluation
            self.store.save_prediction(
                symbol=symbol,
                event_slug=snapshot.event_slug,
                direction=decision.direction,
                confidence=decision.confidence,
                features=market_data,
                grok_response=grok_response,
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Monitor Mode
    # ─────────────────────────────────────────────────────────────────────────
    
    async def run_monitor(self, interval: int = 30):
        """
        Run monitoring mode.
        
        Continuously watches markets and logs without trading.
        """
        self.log.system(f"Starting monitor mode (interval: {interval}s)")
        
        try:
            while True:
                await self.run_predictions()
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            self.log.system("Monitor stopped by user")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────
    
    def show_status(self):
        """Show system status."""
        print("\n" + "="*70)
        print("POLYMARKET 15M TRADER STATUS")
        print("="*70)
        
        # Data store
        summary = self.store.get_summary()
        print("\n📊 DATA STORE")
        print(f"  Snapshots: {summary['snapshots']}")
        print(f"  Outcomes: {summary['outcomes']}")
        print(f"  Labeled Examples: {summary['labeled_examples']}")
        
        # ML Models
        print("\n🧠 ML MODELS")
        model_status = self.predictor.get_model_status()
        for symbol, status in model_status.items():
            ready = "✓" if status['ready_to_bet'] else "✗"
            print(f"  {symbol}: {status['training_examples']} examples, "
                  f"acc={status['validation_accuracy']:.1%} [{ready}]")
        
        # Trading
        print("\n💰 PAPER TRADING")
        stats = self.engine.get_stats()
        print(f"  Capital: ${stats['capital']:.2f} / ${stats['starting_capital']:.2f}")
        print(f"  PnL: ${stats['realized_pnl']:+.2f} ({stats['return_pct']:+.1%})")
        print(f"  Trades: {stats['total_trades']} ({stats['win_rate']:.1%} win rate)")
        print(f"  Open Positions: {stats['open_positions']}")
        
        # Grok
        if self.grok:
            print("\n🤖 GROK")
            grok_stats = self.grok.get_stats()
            print(f"  Calls (7d): {grok_stats['total_calls']}")
            print(f"  Cost (7d): ${grok_stats['total_cost']:.2f}")
            print(f"  Rate Limit: {grok_stats['remaining_calls']}/{grok_stats['rate_limit']} remaining")
        
        # Prediction accuracy
        print("\n📈 PREDICTION ACCURACY")
        pred_acc = summary['prediction_accuracy']
        print(f"  Total: {pred_acc['total']}")
        print(f"  Accuracy: {pred_acc['accuracy']:.1%}")
        
        print("\n" + "="*70)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Polymarket 15M Crypto Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python poly_15m_trader.py --collect           # Collect data (builds training set)
  python poly_15m_trader.py --collect --cycles 10 --interval 120
  python poly_15m_trader.py --predict           # Show predictions
  python poly_15m_trader.py --trade             # Paper trade
  python poly_15m_trader.py --trade --no-grok   # Trade without Grok
  python poly_15m_trader.py --monitor           # Watch markets
  python poly_15m_trader.py --status            # Show status
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--collect', action='store_true', help='Collect market data')
    mode_group.add_argument('--predict', action='store_true', help='Show predictions')
    mode_group.add_argument('--trade', action='store_true', help='Paper trade')
    mode_group.add_argument('--monitor', action='store_true', help='Monitor markets')
    mode_group.add_argument('--status', action='store_true', help='Show status')
    
    # Options
    parser.add_argument('--no-grok', action='store_true', help='Disable Grok API')
    parser.add_argument('--cycles', type=int, default=1, help='Number of cycles')
    parser.add_argument('--interval', type=int, default=60, help='Interval between cycles (seconds)')
    parser.add_argument('--capital', type=float, default=1000.0, help='Starting capital')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Create trader
    async with Poly15MTrader(
        use_grok=not args.no_grok,
        starting_capital=args.capital,
    ) as trader:
        
        if args.collect:
            await trader.run_collection(cycles=args.cycles, interval=args.interval)
        
        elif args.predict:
            await trader.run_predictions()
        
        elif args.trade:
            await trader.run_trading(cycles=args.cycles, interval=args.interval)
        
        elif args.monitor:
            await trader.run_monitor(interval=args.interval)
        
        elif args.status:
            trader.show_status()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
        sys.exit(0)
