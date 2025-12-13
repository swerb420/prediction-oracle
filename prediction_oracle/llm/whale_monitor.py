#!/usr/bin/env python3
"""
Whale Monitor - Background process that keeps whale data fresh.

Runs continuously in background, refreshing whale activity every 5 minutes.
The smart_signal_trader.py reads from the same SQLite database.

Usage:
    python whale_monitor.py            # Start monitoring
    python whale_monitor.py --daemon   # Run as background daemon
"""

import argparse
import time
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from whale_intelligence import WhaleIntelligence, init_database, DB_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class WhaleMonitor:
    """Background whale data refresher."""
    
    def __init__(
        self,
        refresh_interval: int = 300,  # 5 minutes
        activity_hours: float = 0.5,  # 30 minutes of activity
        leaderboard_interval: int = 3600,  # Refresh leaderboards hourly
    ):
        self.refresh_interval = refresh_interval
        self.activity_hours = activity_hours
        self.leaderboard_interval = leaderboard_interval
        self.last_leaderboard_refresh = 0
        
    def run(self):
        """Main monitoring loop."""
        logger.info("="*60)
        logger.info("🐋 WHALE MONITOR STARTING")
        logger.info("="*60)
        logger.info(f"  Refresh interval: {self.refresh_interval}s")
        logger.info(f"  Activity window:  {self.activity_hours * 60:.0f} minutes")
        logger.info(f"  Leaderboard refresh: {self.leaderboard_interval / 60:.0f} minutes")
        logger.info("-"*60)
        
        # Initialize database if needed
        if not DB_PATH.exists():
            logger.info("📦 Initializing whale database...")
            init_database()
            
            # Do initial full scan
            logger.info("🔄 Running initial full scan (this may take a few minutes)...")
            with WhaleIntelligence() as wi:
                wi.scrape_all_leaderboards(limit=100)
                wi.scrape_all_whale_activity(hours_back=24)
            self.last_leaderboard_refresh = time.time()
        
        cycle = 0
        while True:
            try:
                cycle += 1
                now = datetime.now(UTC)
                
                # Check if we need to refresh leaderboards
                if time.time() - self.last_leaderboard_refresh > self.leaderboard_interval:
                    logger.info("📊 Refreshing leaderboards...")
                    with WhaleIntelligence() as wi:
                        wi.scrape_all_leaderboards(limit=100)
                    self.last_leaderboard_refresh = time.time()
                
                # Refresh recent activity
                logger.info(f"[{cycle}] 🔄 Refreshing whale activity (last {self.activity_hours * 60:.0f}min)...")
                with WhaleIntelligence() as wi:
                    total = wi.scrape_all_whale_activity(
                        hours_back=self.activity_hours,
                        min_rank=50  # Only top 50 whales for speed
                    )
                
                # Show current signals
                with WhaleIntelligence() as wi:
                    signals = wi.get_15m_whale_signals(minutes_back=15)
                    
                    active_signals = []
                    for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
                        sig = signals[symbol]
                        if sig['signal'] != 'NEUTRAL' and sig['strength'] >= 40:
                            emoji = '🟢' if sig['signal'] == 'UP' else '🔴'
                            active_signals.append(f"{emoji}{symbol}:{sig['signal']}({sig['strength']})")
                    
                    if active_signals:
                        logger.info(f"  ⚡ Active signals: {' | '.join(active_signals)}")
                    else:
                        logger.info(f"  ⚪ No strong signals currently")
                
                # Sleep until next refresh
                logger.info(f"  💤 Next refresh in {self.refresh_interval}s...")
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Whale monitor stopped.")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)  # Wait a minute before retrying


def main():
    parser = argparse.ArgumentParser(description='Whale Monitor - Keep whale data fresh')
    parser.add_argument('--interval', type=int, default=300, help='Refresh interval in seconds (default: 300)')
    parser.add_argument('--daemon', action='store_true', help='Run as background daemon')
    
    args = parser.parse_args()
    
    if args.daemon:
        # Fork to background
        import os
        if os.fork() > 0:
            sys.exit(0)
        os.setsid()
        if os.fork() > 0:
            sys.exit(0)
        
        # Redirect output to log file
        log_file = Path(__file__).parent / 'data' / 'whale_monitor.log'
        log_file.parent.mkdir(exist_ok=True)
        
        sys.stdout = open(log_file, 'a')
        sys.stderr = sys.stdout
        
        logger.info("🐋 Whale monitor started as daemon")
    
    monitor = WhaleMonitor(refresh_interval=args.interval)
    monitor.run()


if __name__ == '__main__':
    main()
