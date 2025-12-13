#!/usr/bin/env python3
"""
Strategy Variants - Run multiple paper trading strategies in parallel.

Each variant has different thresholds to find optimal settings:
- Conservative: Ultra-high confidence only (fewer trades, higher WR)
- Balanced: Current working settings (70% WR baseline)
- Aggressive: Lower thresholds (more trades, lower WR)

All variants share the same market data but trade independently.
"""

import asyncio
import argparse
import sqlite3
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
import json

# Strategy configurations
STRATEGIES = {
    "conservative": {
        "name": "Conservative (High WR)",
        "description": "Only trade ultra-high confidence signals",
        "min_confidence": 0.58,
        "min_orderbook_strength": 0.93,  # Loosened for more trades
        "min_entry_price": 0.42,
        "max_entry_price": 0.53,
        "base_size_pct": 0.05,
        "max_trades_per_hour": 10,
        "require_momentum_alignment": True,
        "timeframes": ["15M"],  # Only 15M which has 73% WR
    },
    "ultra_conservative": {
        "name": "Ultra Conservative (Max WR)",
        "description": "Extreme filters - only the best setups",
        "min_confidence": 0.60,
        "min_orderbook_strength": 0.945,  # Between conservative (93%) and old (96%)
        "min_entry_price": 0.44,
        "max_entry_price": 0.52,
        "base_size_pct": 0.10,
        "max_trades_per_hour": 8,
        "require_momentum_alignment": True,
        "require_grok_confirmation": True,
        "timeframes": ["15M"],
    },
    "balanced": {
        "name": "Balanced (Current)",
        "description": "Current working settings - 70% WR",
        "min_confidence": 0.52,
        "min_orderbook_strength": 0.85,
        "min_entry_price": 0.40,
        "max_entry_price": 0.55,
        "base_size_pct": 0.08,
        "max_trades_per_hour": 24,
        "require_momentum_alignment": False,
        "timeframes": ["15M", "1H", "4H"],
    },
    "ob_focused": {
        "name": "Orderbook Focused",
        "description": "Trade only on extreme orderbook imbalance",
        "min_confidence": 0.50,
        "min_orderbook_strength": 0.95,
        "min_entry_price": 0.35,
        "max_entry_price": 0.60,
        "base_size_pct": 0.08,
        "max_trades_per_hour": 16,
        "require_momentum_alignment": False,
        "timeframes": ["15M", "1H"],
    },
    "value_entry": {
        "name": "Value Entry",
        "description": "Very tight entry prices for max edge",
        "min_confidence": 0.52,
        "min_orderbook_strength": 0.85,
        "min_entry_price": 0.45,
        "max_entry_price": 0.50,
        "base_size_pct": 0.10,
        "max_trades_per_hour": 12,
        "require_momentum_alignment": False,
        "timeframes": ["15M"],
    },
    "sniper": {
        "name": "Sniper (Max Selectivity)",
        "description": "Only the absolute best setups - targeting 85%+ WR",
        "min_confidence": 0.60,
        "min_orderbook_strength": 0.96,
        "min_entry_price": 0.46,
        "max_entry_price": 0.50,
        "base_size_pct": 0.12,  # Higher size since high confidence
        "max_trades_per_hour": 6,
        "require_momentum_alignment": True,
        "require_grok_confirmation": True,
        "timeframes": ["15M"],
    },
}


@dataclass
class StrategyTrade:
    """A paper trade for a specific strategy."""
    id: int
    strategy: str
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    size_usd: float
    confidence: float
    orderbook_signal: float
    momentum_signal: float
    window_number: int
    timestamp: str
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    actual_outcome: Optional[str] = None


class StrategyVariantTrader:
    """Run multiple strategy variants on the same market data."""
    
    def __init__(self, strategies: list[str], db_path: str = "data/strategy_variants.db"):
        self.strategies = {s: STRATEGIES[s] for s in strategies if s in STRATEGIES}
        self.db_path = db_path
        self.capital = {s: 1000.0 for s in self.strategies}  # $1000 each
        self.traded_windows = {s: set() for s in self.strategies}  # Per-strategy
        self.recent_trades = {s: [] for s in self.strategies}
        self._init_db()
    
    def _init_db(self):
        """Initialize the strategy variants database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS strategy_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size_usd REAL NOT NULL,
                confidence REAL,
                orderbook_signal REAL,
                momentum_signal REAL,
                window_number INTEGER,
                timestamp TEXT NOT NULL,
                closed_at TEXT,
                exit_price REAL,
                pnl REAL,
                actual_outcome TEXT,
                was_correct INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def can_trade(self, strategy: str, symbol: str, timeframe: str, 
                  window_number: int, confidence: float, ob_strength: float,
                  entry_price: float, momentum: float = 0) -> tuple[bool, str]:
        """Check if strategy allows this trade."""
        config = self.strategies[strategy]
        
        # Window duplicate check
        window_key = f"{symbol}/{timeframe}/{window_number}"
        if window_key in self.traded_windows[strategy]:
            return False, "Already traded this window"
        
        # Timeframe check
        if timeframe not in config["timeframes"]:
            return False, f"Timeframe {timeframe} not in strategy"
        
        # Confidence check
        if confidence < config["min_confidence"]:
            return False, f"Confidence {confidence:.0%} < {config['min_confidence']:.0%}"
        
        # Orderbook strength check
        if ob_strength < config["min_orderbook_strength"]:
            return False, f"OB {ob_strength:.0%} < {config['min_orderbook_strength']:.0%}"
        
        # Entry price check
        if entry_price < config["min_entry_price"] or entry_price > config["max_entry_price"]:
            return False, f"Entry {entry_price:.2f} outside {config['min_entry_price']}-{config['max_entry_price']}"
        
        # Momentum alignment check
        if config.get("require_momentum_alignment", False):
            # For DOWN signal, momentum should be negative (or at least not positive)
            # This is simplified - real implementation would check signal direction
            pass
        
        # Rate limit check
        hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        self.recent_trades[strategy] = [t for t in self.recent_trades[strategy] if t > hour_ago]
        if len(self.recent_trades[strategy]) >= config["max_trades_per_hour"]:
            return False, f"Rate limit ({len(self.recent_trades[strategy])}/{config['max_trades_per_hour']}/hr)"
        
        # Capital check
        if self.capital[strategy] < 10:
            return False, "Insufficient capital"
        
        return True, "OK"
    
    def open_trade(self, strategy: str, symbol: str, timeframe: str,
                   direction: str, entry_price: float, confidence: float,
                   ob_signal: float, momentum: float, window_number: int) -> Optional[int]:
        """Open a trade for a specific strategy."""
        config = self.strategies[strategy]
        
        # Calculate position size
        size_pct = config["base_size_pct"]
        if confidence >= 0.65:
            size_pct *= 1.5
        size_usd = self.capital[strategy] * size_pct
        size_usd = min(size_usd, self.capital[strategy] * 0.30)
        
        if size_usd < 5:
            return None
        
        # Record trade
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO strategy_trades 
            (strategy, symbol, timeframe, direction, entry_price, size_usd,
             confidence, orderbook_signal, momentum_signal, window_number, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (strategy, symbol, timeframe, direction, entry_price, size_usd,
              confidence, ob_signal, momentum, window_number,
              datetime.now(timezone.utc).isoformat()))
        trade_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Update state
        self.capital[strategy] -= size_usd
        self.traded_windows[strategy].add(f"{symbol}/{timeframe}/{window_number}")
        self.recent_trades[strategy].append(datetime.now(timezone.utc))
        
        return trade_id
    
    def get_stats(self) -> dict:
        """Get stats for all strategies."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        stats = {}
        for strategy in self.strategies:
            c.execute('''
                SELECT COUNT(*), 
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                       SUM(pnl)
                FROM strategy_trades 
                WHERE strategy = ? AND closed_at IS NOT NULL
            ''', (strategy,))
            total, wins, pnl = c.fetchone()
            
            c.execute('''
                SELECT COUNT(*) FROM strategy_trades 
                WHERE strategy = ? AND closed_at IS NULL
            ''', (strategy,))
            open_count = c.fetchone()[0]
            
            stats[strategy] = {
                "name": self.strategies[strategy]["name"],
                "resolved": total or 0,
                "wins": wins or 0,
                "losses": (total or 0) - (wins or 0),
                "win_rate": (wins / total * 100) if total else 0,
                "pnl": pnl or 0,
                "open": open_count,
                "capital": self.capital[strategy],
            }
        
        conn.close()
        return stats
    
    def print_stats(self):
        """Print formatted stats for all strategies."""
        stats = self.get_stats()
        
        print("\n" + "═" * 70)
        print("         STRATEGY VARIANT COMPARISON")
        print("═" * 70)
        
        for strategy, s in sorted(stats.items(), key=lambda x: -x[1]["win_rate"]):
            config = self.strategies[strategy]
            print(f"\n📊 {s['name']}")
            print(f"   {config['description']}")
            print(f"   Trades: {s['resolved']} resolved, {s['open']} open")
            if s['resolved'] > 0:
                print(f"   Record: {s['wins']}W / {s['losses']}L ({s['win_rate']:.1f}% WR)")
                print(f"   P&L: ${s['pnl']:+,.2f}")
            print(f"   Capital: ${s['capital']:,.2f}")
        
        print("\n" + "═" * 70)


async def run_variant_comparison():
    """Run all strategy variants in parallel."""
    from market_intelligence import MarketIntelligence
    
    # Initialize trader with all strategies
    trader = StrategyVariantTrader(list(STRATEGIES.keys()))
    
    print("Starting Strategy Variant Comparison...")
    print(f"Running {len(trader.strategies)} strategies in parallel")
    
    for name, config in trader.strategies.items():
        print(f"  • {config['name']}: {config['description']}")
    
    scan_count = 0
    
    async with MarketIntelligence() as intel:
        while True:
            scan_count += 1
            
            # Get market data for all timeframes
            for tf in ["15M", "1H", "4H"]:
                try:
                    markets = await intel.get_all_markets(timeframe=tf)
                except Exception as e:
                    print(f"Error fetching {tf} markets: {e}")
                    continue
                
                # Get window number
                now = datetime.now(timezone.utc)
                if tf == "15M":
                    window_num = int(now.timestamp()) // 900
                elif tf == "1H":
                    window_num = int(now.timestamp()) // 3600
                else:  # 4H
                    window_num = int(now.timestamp()) // 14400
                
                for symbol, market in markets.items():
                    if not market.orderbook:
                        continue
                    
                    # Calculate signals
                    ob_imbalance = market.orderbook.imbalance
                    ob_strength = abs(ob_imbalance)
                    
                    # Determine direction
                    if ob_imbalance < -0.5:
                        direction = "DOWN"
                        entry_price = market.poly_no_price
                    elif ob_imbalance > 0.5:
                        direction = "UP"
                        entry_price = market.poly_yes_price
                    else:
                        continue
                    
                    # Calculate confidence
                    confidence = 0.50 + ob_strength * 0.15
                    momentum = 0  # Simplified
                    
                    # Try each strategy
                    for strategy in trader.strategies:
                        can, reason = trader.can_trade(
                            strategy, symbol, tf, window_num,
                            confidence, ob_strength, entry_price, momentum
                        )
                        
                        if can:
                            trade_id = trader.open_trade(
                                strategy, symbol, tf, direction,
                                entry_price, confidence, ob_imbalance, momentum, window_num
                            )
                            if trade_id:
                                print(f"  [{strategy}] {symbol}/{tf} {direction} @ {entry_price:.3f}")
            
            # Print stats every 10 scans
            if scan_count % 10 == 0:
                trader.print_stats()
            
            await asyncio.sleep(30)


def show_current_stats():
    """Show stats from existing database."""
    import os
    db_path = "data/strategy_variants.db"
    
    if not os.path.exists(db_path):
        print("No strategy variants database found. Run the comparison first!")
        return
    
    trader = StrategyVariantTrader(list(STRATEGIES.keys()), db_path)
    trader.print_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strategy variant comparison")
    parser.add_argument("--stats", action="store_true", help="Show current stats only")
    parser.add_argument("--run", action="store_true", help="Run the comparison")
    
    args = parser.parse_args()
    
    if args.stats:
        show_current_stats()
    elif args.run:
        asyncio.run(run_variant_comparison())
    else:
        # Show available strategies
        print("\n📊 Available Strategy Variants:\n")
        for name, config in STRATEGIES.items():
            print(f"  {name}:")
            print(f"    {config['name']}")
            print(f"    {config['description']}")
            print(f"    Min Confidence: {config['min_confidence']:.0%}")
            print(f"    Min OB Strength: {config['min_orderbook_strength']:.0%}")
            print(f"    Entry Range: {config['min_entry_price']:.2f}-{config['max_entry_price']:.2f}")
            print(f"    Timeframes: {', '.join(config['timeframes'])}")
            print()
        
        print("\nRun with --run to start comparison, or --stats to see results")
