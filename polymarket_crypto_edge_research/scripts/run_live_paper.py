#!/usr/bin/env python3
"""
Run live paper trading.
Starts the orchestrator for real-time prediction and paper trading.

Usage:
    python -m scripts.run_live_paper
    python -m scripts.run_live_paper --symbols BTC ETH SOL
    python -m scripts.run_live_paper --prediction-interval 15
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from core.config import get_settings
from core.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


# Global reference for signal handler
_orchestrator = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    if _orchestrator:
        asyncio.create_task(_orchestrator.stop())


async def run_paper_trading(
    symbols: list[str],
    prediction_interval: int = 15,
    model_dir: Path = Path("models"),
    trades_dir: Path = Path("trades"),
    poll_polymarket: bool = True,
    status_interval: int = 300
) -> None:
    """
    Run live paper trading.
    
    Args:
        symbols: Crypto symbols to trade
        prediction_interval: Prediction interval in minutes
        model_dir: Path to model registry
        trades_dir: Path to save trades
        poll_polymarket: Whether to poll Polymarket
        status_interval: Status print interval in seconds
    """
    global _orchestrator
    
    from exec.live_orchestrator import LiveOrchestrator, OrchestratorConfig
    from core.time_utils import now_utc
    
    logger.info("=" * 60)
    logger.info("Starting Live Paper Trading")
    logger.info(f"Time: {now_utc().isoformat()}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Prediction Interval: {prediction_interval}m")
    logger.info("=" * 60)
    
    # Create config
    config = OrchestratorConfig(
        mode="paper",
        crypto_symbols=symbols,
        prediction_interval_minutes=prediction_interval,
        model_registry_path=model_dir,
        poll_polymarket=poll_polymarket,
        log_trades=True,
        trades_log_path=trades_dir
    )
    
    # Create orchestrator
    _orchestrator = LiveOrchestrator(config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Status printing task
    async def print_status_loop():
        while _orchestrator._running:
            await asyncio.sleep(status_interval)
            if _orchestrator._running:
                _orchestrator.print_status()
    
    # Start orchestrator with status printing
    try:
        status_task = asyncio.create_task(print_status_loop())
        
        logger.info("Paper trading started. Press Ctrl+C to stop.")
        await _orchestrator.start()
        
    except asyncio.CancelledError:
        logger.info("Paper trading cancelled")
    finally:
        await _orchestrator.stop()
        status_task.cancel()
        
        # Print final stats
        logger.info("\n" + "=" * 60)
        logger.info("Final Paper Trading Stats:")
        _orchestrator.print_status()
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run live paper trading")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH", "SOL"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--prediction-interval",
        type=int,
        default=15,
        help="Prediction interval in minutes"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Model registry directory"
    )
    parser.add_argument(
        "--trades-dir",
        type=Path,
        default=Path("trades"),
        help="Trades log directory"
    )
    parser.add_argument(
        "--no-polymarket",
        action="store_true",
        help="Disable Polymarket polling"
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=300,
        help="Status print interval in seconds"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Create directories
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.trades_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for models
    if not any(args.model_dir.glob("*.pkl")):
        logger.warning(
            f"No models found in {args.model_dir}. "
            "Run daily_retrain.py first or predictions will use defaults."
        )
    
    # Run
    try:
        asyncio.run(run_paper_trading(
            symbols=args.symbols,
            prediction_interval=args.prediction_interval,
            model_dir=args.model_dir,
            trades_dir=args.trades_dir,
            poll_polymarket=not args.no_polymarket,
            status_interval=args.status_interval
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
