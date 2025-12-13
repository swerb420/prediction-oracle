#!/usr/bin/env python3
"""
Enhanced Crypto Trading Bot with Whale/Venue Signals
=====================================================

Multi-venue ML + Whale Consensus + Grok 4.1 hybrid system.
Targets 63%+ win rate through additional alpha sources.

Usage:
    python -m prediction_oracle.llm.enhanced_crypto_trader --help
    
    # Run once (single prediction cycle with whale data)
    python -m prediction_oracle.llm.enhanced_crypto_trader --once
    
    # Run continuously (every 15 min)
    python -m prediction_oracle.llm.enhanced_crypto_trader --continuous
    
    # Show enhanced stats (includes whale alignment analysis)
    python -m prediction_oracle.llm.enhanced_crypto_trader --stats
    
    # Retrain enhanced ML models
    python -m prediction_oracle.llm.enhanced_crypto_trader --retrain
    
    # Quick test without whale/venue data (faster)
    python -m prediction_oracle.llm.enhanced_crypto_trader --once --fast

Environment Variables:
    XAI_API_KEY: Your xAI API key for Grok 4.1 Fast validation
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


async def run_enhanced_cycle(
    oracle,
    engine,
    verbose: bool = True,
    fast_mode: bool = False,
) -> dict:
    """
    Run a single enhanced prediction + trading cycle.
    
    Args:
        oracle: Enhanced hybrid oracle
        engine: Enhanced paper trading engine
        verbose: Print details
        fast_mode: Skip whale/venue data for speed
        
    Returns:
        Summary of cycle results
    """
    from .enhanced_hybrid_oracle import EnhancedHybridPrediction
    from .enhanced_paper_trading import EnhancedClosedTrade
    
    cycle_start = datetime.now(timezone.utc)
    results = {
        "timestamp": cycle_start.isoformat(),
        "predictions": {},
        "positions_opened": [],
        "positions_closed": [],
    }
    
    if verbose:
        logger.info("=" * 60)
        logger.info(f"🔮 Starting enhanced prediction cycle at {cycle_start.strftime('%H:%M:%S')}")
        if not fast_mode:
            logger.info("   Including whale consensus and multi-venue data...")
        logger.info("=" * 60)
    
    # Step 1: Close any expired positions
    closed = await engine.update_positions()
    results["positions_closed"] = [t.id for t in closed]
    
    if closed and verbose:
        for trade in closed:
            emoji = "✅" if trade.is_win else "❌"
            whale_info = f"whale:{'+' if trade.whale_aligned else '-'}"
            logger.info(
                f"{emoji} Closed: {trade.symbol} {trade.direction} | "
                f"PnL: {trade.pnl_pct:+.2%} | {whale_info}"
            )
    
    # Step 2: Get enhanced predictions
    try:
        # Temporarily adjust oracle settings for fast mode
        original_whale = oracle.use_whale
        original_venue = oracle.use_venue
        
        if fast_mode:
            oracle.use_whale = False
            oracle.use_venue = False
        
        predictions = await oracle.predict_all()
        
        # Restore settings
        oracle.use_whale = original_whale
        oracle.use_venue = original_venue
        
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
            "whale_consensus": pred.whale_consensus,
            "venue_consensus": pred.venue_consensus,
            "clean_score": pred.clean_score,
            "price": pred.current_price,
        }
        
        if verbose:
            emoji_map = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "⚪"}
            emoji = emoji_map.get(pred.final_signal, "⚪")
            
            whale_emoji = "🐋" if abs(pred.whale_consensus) > 0.3 else ""
            venue_emoji = "🏛️" if abs(pred.venue_consensus) > 0.3 else ""
            
            logger.info(
                f"{emoji} {symbol}: {pred.final_signal} @ ${pred.current_price:,.2f} | "
                f"Conf: {pred.final_confidence:.1%} | "
                f"Whale: {pred.whale_consensus:+.2f} {whale_emoji} | "
                f"Venue: {pred.venue_consensus:+.2f} {venue_emoji}"
            )
            
            if pred.should_trade:
                logger.info(f"   📝 {pred.trade_rationale}")
        
        # Step 4: Open positions for tradeable signals
        if pred.should_trade:
            position = await engine.open_position(pred)
            if position:
                results["positions_opened"].append(position.id)
    
    # Save state
    engine._save_state()
    
    if verbose:
        logger.info("-" * 60)
        logger.info(
            f"Cycle complete: {len(results['positions_opened'])} opened, "
            f"{len(results['positions_closed'])} closed"
        )
        logger.info(f"Capital: ${engine.capital:,.2f} | Win Rate: {engine.stats.win_rate:.1%}")
    
    return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Crypto Trading Bot with Whale/Venue Signals"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run single prediction cycle and exit"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run continuously every 15 minutes"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show enhanced trading statistics"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain enhanced ML models"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip whale/venue data for faster predictions"
    )
    parser.add_argument(
        "--capital", type=float, default=10000.0,
        help="Initial paper trading capital (default: 10000)"
    )
    
    args = parser.parse_args()
    
    # Import components
    from .enhanced_hybrid_oracle import EnhancedHybridOracle
    from .enhanced_paper_trading import EnhancedPaperTradingEngine
    
    # Check API key
    if not os.getenv("XAI_API_KEY"):
        logger.warning(
            "XAI_API_KEY not set. Grok validation will be disabled. "
            "Set it for better predictions."
        )
    
    # Initialize components
    oracle = EnhancedHybridOracle(
        use_whale_signals=not args.fast,
        use_venue_data=not args.fast,
    )
    
    engine = EnhancedPaperTradingEngine(
        initial_capital=args.capital,
        state_file="./enhanced_paper_trading_state.json",
    )
    
    if args.stats:
        engine.print_stats()
        return
    
    if args.retrain:
        logger.info("Retraining enhanced ML models...")
        await oracle.initialize(retrain_ml=True)
        logger.info("Retrain complete!")
        return
    
    # Initialize oracle
    await oracle.initialize()
    
    if args.once:
        await run_enhanced_cycle(oracle, engine, verbose=True, fast_mode=args.fast)
        engine.print_stats()
        
    elif args.continuous:
        logger.info("Starting continuous enhanced trading mode...")
        logger.info("Press Ctrl+C to stop")
        
        while True:
            try:
                await run_enhanced_cycle(oracle, engine, verbose=True, fast_mode=args.fast)
                
                # Calculate wait time until next 15-min mark
                now = datetime.now(timezone.utc)
                minutes_to_wait = 15 - (now.minute % 15)
                seconds_to_wait = minutes_to_wait * 60 - now.second
                
                if seconds_to_wait < 30:
                    seconds_to_wait += 15 * 60
                
                logger.info(f"⏰ Next cycle in {seconds_to_wait // 60}m {seconds_to_wait % 60}s")
                await asyncio.sleep(seconds_to_wait)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Stopping enhanced trading bot...")
                engine.print_stats()
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(60)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
