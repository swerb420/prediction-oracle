#!/usr/bin/env python3
"""
Crypto 15-Minute Prediction Trading Bot
======================================

ML + Grok 4.1 Fast hybrid prediction system for BTC, ETH, SOL.
Paper trades on 15-minute intervals with full position tracking.

Usage:
    python -m prediction_oracle.llm.crypto_trader --help
    
    # Run once (single prediction cycle)
    python -m prediction_oracle.llm.crypto_trader --once
    
    # Run continuously (every 15 min)
    python -m prediction_oracle.llm.crypto_trader --continuous
    
    # Show current stats
    python -m prediction_oracle.llm.crypto_trader --stats
    
    # Retrain ML models
    python -m prediction_oracle.llm.crypto_trader --retrain

Environment Variables:
    XAI_API_KEY: Your xAI API key for Grok 4.1 Fast validation
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


async def run_single_cycle(
    oracle,
    engine,
    verbose: bool = True
) -> dict:
    """
    Run a single prediction + trading cycle.
    
    Returns:
        Summary of what happened this cycle
    """
    from .hybrid_oracle import HybridPrediction
    from .paper_trading import ClosedTrade
    
    cycle_start = datetime.utcnow()
    results = {
        "timestamp": cycle_start.isoformat(),
        "predictions": {},
        "positions_opened": [],
        "positions_closed": [],
    }
    
    if verbose:
        logger.info("=" * 60)
        logger.info(f"🔮 Starting prediction cycle at {cycle_start.strftime('%H:%M:%S')}")
        logger.info("=" * 60)
    
    # Step 1: Close any positions that hit their target
    closed = await engine.update_positions()
    results["positions_closed"] = [t.id for t in closed]
    
    if closed and verbose:
        for trade in closed:
            emoji = "✅" if trade.is_win else "❌"
            logger.info(f"{emoji} Closed: {trade.symbol} {trade.direction} | PnL: {trade.pnl_pct:+.2%}")
    
    # Step 2: Get predictions for all symbols
    try:
        predictions = await oracle.predict_all()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return results
    
    # Step 3: Process each prediction
    for symbol, pred in predictions.items():
        results["predictions"][symbol] = {
            "signal": pred.final_signal,
            "confidence": pred.final_confidence,
            "should_trade": pred.should_trade,
            "ml_direction": pred.ml_direction,
            "grok_agrees": pred.grok_agrees,
            "price": pred.current_price
        }
        
        if verbose:
            emoji_map = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "⚪"}
            emoji = emoji_map.get(pred.final_signal, "⚪")
            
            logger.info(
                f"{emoji} {symbol}: {pred.final_signal} @ ${pred.current_price:,.2f} | "
                f"Conf: {pred.final_confidence:.1%} | "
                f"ML: {pred.ml_direction} ({pred.ml_confidence:.1%}) | "
                f"Grok: {'✓' if pred.grok_agrees else '✗'} ({pred.grok_recommendation})"
            )
            
            if pred.grok_reasoning:
                logger.info(f"   └─ {pred.grok_reasoning[:80]}...")
        
        # Step 4: Open position if signal is tradeable
        if pred.should_trade:
            position = await engine.open_position(pred)
            if position:
                results["positions_opened"].append(position.id)
    
    # Step 5: Log current state
    if verbose:
        logger.info("-" * 60)
        logger.info(f"Open positions: {len(engine.open_positions)}")
        logger.info(f"Capital: ${engine.capital:,.2f}")
        logger.info(f"Win rate: {engine.stats.win_rate:.1%} ({engine.stats.total_trades} trades)")
        logger.info("-" * 60)
    
    return results


async def run_continuous(
    oracle,
    engine,
    interval_minutes: int = 15
):
    """
    Run continuous trading loop.
    Executes at the start of each 15-minute interval.
    """
    logger.info("🚀 Starting continuous trading mode")
    logger.info(f"   Interval: {interval_minutes} minutes")
    logger.info("   Press Ctrl+C to stop")
    logger.info("")
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            
            # Run prediction cycle
            await run_single_cycle(oracle, engine, verbose=True)
            
            # Calculate sleep until next interval
            now = datetime.utcnow()
            minutes_past = now.minute % interval_minutes
            seconds_past = now.second
            
            # Sleep until next interval boundary
            sleep_seconds = (interval_minutes - minutes_past) * 60 - seconds_past
            
            if sleep_seconds < 10:
                # Too close to next interval, skip to the one after
                sleep_seconds += interval_minutes * 60
            
            next_run = now + timedelta(seconds=sleep_seconds)
            logger.info(f"💤 Sleeping until {next_run.strftime('%H:%M:%S')} ({sleep_seconds // 60}m {sleep_seconds % 60}s)")
            
            await asyncio.sleep(sleep_seconds)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping continuous mode...")
        
        # Close all positions before exiting
        if engine.open_positions:
            logger.info("Closing all open positions...")
            await engine.force_close_all()
        
        print(engine.get_summary())


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto 15-Min ML + Grok Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--once", "-1",
        action="store_true",
        help="Run single prediction cycle"
    )
    
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuously every 15 minutes"
    )
    
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show current trading statistics"
    )
    
    parser.add_argument(
        "--retrain", "-r",
        action="store_true",
        help="Retrain ML models on fresh data"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all paper trading state"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial paper trading capital (default: 10000)"
    )
    
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.05,
        help="Position size as fraction of capital (default: 0.05 = 5%%)"
    )
    
    parser.add_argument(
        "--hold-candles",
        type=int,
        default=1,
        help="Number of 15m candles to hold position (default: 1)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum confidence to trade (default: 0.60)"
    )
    
    parser.add_argument(
        "--no-grok",
        action="store_true",
        help="Disable Grok validation (ML-only mode)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check for XAI API key
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key and not args.no_grok:
        logger.warning("XAI_API_KEY not set. Running in ML-only mode.")
        args.no_grok = True
    
    # Import components
    from .hybrid_oracle import HybridCryptoOracle
    from .paper_trading import PaperTradingEngine, get_trading_engine
    
    # Initialize engine
    engine = get_trading_engine(
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        hold_candles=args.hold_candles
    )
    
    # Handle stats-only mode
    if args.stats:
        print(engine.get_summary())
        return
    
    # Handle reset
    if args.reset:
        confirm = input("Are you sure you want to reset all trading state? (y/N): ")
        if confirm.lower() == 'y':
            engine.reset()
            print("Trading state reset.")
        return
    
    # Initialize oracle
    oracle = HybridCryptoOracle(
        xai_api_key=xai_key,
        trade_confidence_threshold=args.confidence_threshold,
        use_grok_validation=not args.no_grok
    )
    
    logger.info("🔧 Initializing ML models...")
    await oracle.initialize(retrain_ml=args.retrain)
    logger.info("✓ Models initialized")
    
    # Handle retrain-only mode
    if args.retrain and not (args.once or args.continuous):
        logger.info("ML models retrained. Use --once or --continuous to start trading.")
        return
    
    # Run mode
    if args.continuous:
        await run_continuous(oracle, engine)
    else:
        # Default to single cycle
        await run_single_cycle(oracle, engine, verbose=True)
        print()
        print(engine.get_summary())
    
    # Cleanup
    await oracle.close()


if __name__ == "__main__":
    asyncio.run(main())
