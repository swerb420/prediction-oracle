#!/usr/bin/env python3
"""
Whale Paper Trader - Track top whale performance with paper trades

Watches ExpressoMartini, 15m-a4, and other elite 15M traders.
Paper trades their bets to see how they perform.

Usage:
    python whale_paper_trader.py            # Start tracking
    python whale_paper_trader.py --status   # Show current whale performance
"""

import sqlite3
import time
import argparse
import logging
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

from whale_intelligence import WhaleIntelligence, DB_PATH as WHALE_DB_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database for whale paper trades
PAPER_DB_PATH = Path(__file__).parent / 'data' / 'whale_paper_trades.db'

# Elite whales to track - ONLY the top 15M performers
ELITE_WHALES = {
    '15m-a4': {'capital': 100.0, 'wallet': None},  # Will be resolved
    'ExpressoMartini': {'capital': 100.0, 'wallet': None},
}

# Starting capital per whale
DEFAULT_CAPITAL = 100.0


@dataclass
class WhalePaperTrade:
    """A paper trade copying a whale's bet."""
    id: int
    whale_name: str
    symbol: str
    direction: str  # UP or DOWN
    entry_time: datetime
    entry_price: float
    size_usd: float
    market_title: str
    tx_hash: str
    
    # Filled on resolution
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[str] = None  # WIN, LOSS, PENDING


def init_paper_db():
    """Initialize whale paper trades database."""
    PAPER_DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(PAPER_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            whale_name TEXT NOT NULL,
            whale_wallet TEXT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            entry_price REAL NOT NULL,
            size_usd REAL NOT NULL,
            market_title TEXT,
            tx_hash TEXT UNIQUE,
            
            exit_time TIMESTAMP,
            exit_price REAL,
            pnl REAL,
            outcome TEXT DEFAULT 'PENDING',
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Performance summary by whale
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            whale_name TEXT UNIQUE NOT NULL,
            starting_capital REAL DEFAULT 100.0,
            current_capital REAL DEFAULT 100.0,
            total_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            avg_pnl_per_trade REAL DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"✅ Whale paper trades DB initialized at {PAPER_DB_PATH}")


class WhalePaperTrader:
    """Paper trade whale bets to track their performance."""
    
    def __init__(self):
        if not PAPER_DB_PATH.exists():
            init_paper_db()
        self.conn = sqlite3.connect(PAPER_DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.whale_conn = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.conn:
            self.conn.close()
    
    def scan_whale_trades(self, minutes_back: int = 15) -> List[Dict]:
        """Scan for new trades from elite whales."""
        new_trades = []
        cursor = self.conn.cursor()
        
        # Get trades from whale intelligence DB
        whale_conn = sqlite3.connect(WHALE_DB_PATH)
        whale_conn.row_factory = sqlite3.Row
        whale_cursor = whale_conn.cursor()
        
        cutoff = (datetime.now(UTC) - timedelta(minutes=minutes_back)).isoformat()
        
        # Get whale names
        whale_names = list(ELITE_WHALES.keys())
        
        # Find 15M trades from our tracked whales only
        whale_cursor.execute(f'''
            SELECT 
                ct.whale_wallet,
                wp.name,
                wp.pseudonym,
                ct.symbol,
                ct.direction,
                ct.side,
                ct.price,
                ct.usdc_value,
                ct.timestamp,
                ct.market_title,
                ct.tx_hash,
                wp.pnl_30d
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            WHERE ct.timestamp >= ?
              AND ct.market_type = '15MIN'
              AND (wp.name IN ({','.join(['?']*len(whale_names))}))
              AND ct.side = 'BUY'
            ORDER BY ct.timestamp DESC
        ''', (cutoff, *whale_names))
        
        trades = whale_cursor.fetchall()
        
        for trade in trades:
            # Check if we already have this trade
            cursor.execute('SELECT id FROM whale_paper_trades WHERE tx_hash = ?', (trade['tx_hash'],))
            if cursor.fetchone():
                continue  # Already tracked
            
            whale_name = trade['name'] or trade['pseudonym'] or trade['whale_wallet'][:12]
            direction = trade['direction']
            
            if direction not in ['UP', 'DOWN']:
                continue
            
            # Get whale's capital and calculate scaled position
            whale_capital = ELITE_WHALES.get(whale_name, {}).get('capital', DEFAULT_CAPITAL)
            
            # Scale position: if whale bets $100 and we have $100 capital, we bet proportionally
            # Assume whale's avg position is ~$80 based on data, so scale to that
            whale_size = trade['usdc_value'] or 0
            if whale_size > 0:
                # Cap at 20% of capital per trade for safety
                scaled_size = min(whale_capital * 0.20, whale_size * (whale_capital / 1000))
                scaled_size = max(1.0, scaled_size)  # Minimum $1
            else:
                scaled_size = whale_capital * 0.10  # Default 10% if no size
            
            # Create paper trade
            cursor.execute('''
                INSERT INTO whale_paper_trades (
                    whale_name, whale_wallet, symbol, direction,
                    entry_time, entry_price, size_usd, market_title, tx_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                whale_name,
                trade['whale_wallet'],
                trade['symbol'],
                direction,
                trade['timestamp'],
                trade['price'],
                scaled_size,  # Use scaled size, not whale's actual size
                trade['market_title'],
                trade['tx_hash']
            ))
            
            new_trades.append({
                'whale': whale_name,
                'symbol': trade['symbol'],
                'direction': direction,
                'size': trade['usdc_value'] or 0,
                'pnl_30d': trade['pnl_30d'] or 0
            })
        
        self.conn.commit()
        whale_conn.close()
        
        return new_trades
    
    def resolve_trades(self) -> List[Dict]:
        """Check pending trades and resolve based on actual Binance price."""
        import httpx
        
        cursor = self.conn.cursor()
        
        # Get trades older than 15 minutes that are still pending
        cutoff = (datetime.now(UTC) - timedelta(minutes=16)).isoformat()
        
        cursor.execute('''
            SELECT * FROM whale_paper_trades 
            WHERE outcome = 'PENDING' AND entry_time < ?
        ''', (cutoff,))
        
        resolved = []
        for row in cursor.fetchall():
            trade_id = row['id']
            symbol = row['symbol']
            direction = row['direction']
            entry_time_str = row['entry_time']
            size = row['size_usd']
            entry_price = row['entry_price']
            
            try:
                # Parse entry time
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                exit_time = entry_time + timedelta(minutes=15)
                
                # Get Binance klines
                start_ts = int(entry_time.timestamp()) * 1000
                end_ts = int(exit_time.timestamp()) * 1000
                
                resp = httpx.get(
                    'https://api.binance.com/api/v3/klines',
                    params={
                        'symbol': f'{symbol}USDT',
                        'interval': '1m',
                        'startTime': start_ts,
                        'endTime': end_ts,
                        'limit': 20
                    },
                    timeout=10
                )
                
                if resp.status_code == 200:
                    klines = resp.json()
                    if klines and len(klines) >= 2:
                        open_price = float(klines[0][1])
                        close_price = float(klines[-1][4])
                        
                        actual_direction = 'UP' if close_price > open_price else 'DOWN'
                        won = actual_direction == direction
                        
                        # Calculate PnL
                        if won:
                            # Win: price goes from ~0.50 to ~1.00, profit = size * (1/entry_price - 1)
                            pnl = size * ((1.0 / entry_price) - 1) if entry_price > 0 else size
                        else:
                            # Loss: lose the full position
                            pnl = -size
                        
                        outcome = 'WIN' if won else 'LOSS'
                        
                        # Update trade
                        cursor.execute('''
                            UPDATE whale_paper_trades
                            SET exit_time = ?, exit_price = ?, pnl = ?, outcome = ?
                            WHERE id = ?
                        ''', (exit_time.isoformat(), close_price, pnl, outcome, trade_id))
                        
                        resolved.append({
                            'id': trade_id,
                            'whale': row['whale_name'],
                            'symbol': symbol,
                            'direction': direction,
                            'outcome': outcome,
                            'pnl': pnl
                        })
                        
                        logger.info(f"{'✅' if won else '❌'} {row['whale_name']} {symbol} {direction}: ${pnl:+.2f}")
                        
            except Exception as e:
                logger.warning(f"Error resolving trade {trade_id}: {e}")
        
        self.conn.commit()
        return resolved
    
    def get_whale_stats(self) -> Dict[str, Dict]:
        """Get performance stats for each tracked whale."""
        cursor = self.conn.cursor()
        
        stats = {}
        for whale_name in ELITE_WHALES.keys():
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome = 'PENDING' THEN 1 ELSE 0 END) as pending,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    AVG(size_usd) as avg_size
                FROM whale_paper_trades
                WHERE whale_name = ?
            ''', (whale_name,))
            
            row = cursor.fetchone()
            if row and row['total'] > 0:
                total = row['total']
                wins = row['wins'] or 0
                losses = row['losses'] or 0
                resolved = wins + losses
                
                # Calculate current capital
                starting_capital = ELITE_WHALES[whale_name]['capital']
                current_capital = starting_capital + (row['total_pnl'] or 0)
                
                stats[whale_name] = {
                    'total': total,
                    'wins': wins,
                    'losses': losses,
                    'pending': row['pending'] or 0,
                    'win_rate': (wins / resolved * 100) if resolved > 0 else 0,
                    'total_pnl': row['total_pnl'] or 0,
                    'avg_size': row['avg_size'] or 0,
                    'starting_capital': starting_capital,
                    'current_capital': current_capital,
                    'roi': ((current_capital - starting_capital) / starting_capital * 100) if starting_capital > 0 else 0
                }
            else:
                # No trades yet for this whale
                starting_capital = ELITE_WHALES[whale_name]['capital']
                stats[whale_name] = {
                    'total': 0, 'wins': 0, 'losses': 0, 'pending': 0,
                    'win_rate': 0, 'total_pnl': 0, 'avg_size': 0,
                    'starting_capital': starting_capital, 
                    'current_capital': starting_capital,
                    'roi': 0
                }
        
        return stats
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get most recent whale paper trades."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM whale_paper_trades
            ORDER BY entry_time DESC
            LIMIT ?
        ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def print_status(self):
        """Print current whale tracking status."""
        print("\n" + "="*80)
        print("🐋 WHALE PAPER TRADER - ExpressoMartini & 15m-a4")
        print("="*80)
        
        # Overall stats
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) as total FROM whale_paper_trades')
        total = cursor.fetchone()['total']
        
        # Per-whale stats
        stats = self.get_whale_stats()
        
        print("\n💰 WHALE PORTFOLIO ($100 starting capital each)")
        print("-"*80)
        print(f"{'Whale':<18} | {'Capital':>10} | {'PnL':>10} | {'ROI':>8} | {'W/L':>7} | {'WR':>6} | {'Pending':>7}")
        print("-"*80)
        
        total_capital = 0
        total_pnl = 0
        
        for whale, data in stats.items():
            emoji = '🟢' if data['total_pnl'] > 0 else ('🔴' if data['total_pnl'] < 0 else '⚪')
            wl = f"{data['wins']}/{data['losses']}"
            print(f"  {emoji} {whale:<15} | ${data['current_capital']:>9.2f} | ${data['total_pnl']:>+9.2f} | "
                  f"{data['roi']:>+7.1f}% | {wl:>7} | {data['win_rate']:>5.1f}% | {data['pending']:>7}")
            
            total_capital += data['current_capital']
            total_pnl += data['total_pnl']
        
        print("-"*80)
        starting = sum(ELITE_WHALES[w]['capital'] for w in ELITE_WHALES)
        total_roi = ((total_capital - starting) / starting * 100) if starting > 0 else 0
        print(f"  📊 TOTAL          | ${total_capital:>9.2f} | ${total_pnl:>+9.2f} | {total_roi:>+7.1f}%")
        
        # Recent trades
        recent = self.get_recent_trades(15)
        
        if recent:
            print("\n⚡ RECENT WHALE TRADES")
            print("-"*80)
            for trade in recent:
                dir_emoji = '🟢' if trade['direction'] == 'UP' else '🔴'
                outcome = trade['outcome']
                if outcome == 'WIN':
                    out_emoji = '✅'
                elif outcome == 'LOSS':
                    out_emoji = '❌'
                else:
                    out_emoji = '⏳'
                
                ts = trade['entry_time'][11:19] if trade['entry_time'] else '??:??:??'
                pnl_str = f"${trade['pnl']:>+7.2f}" if trade['pnl'] else "  pending"
                print(f"  {ts} | {dir_emoji} {trade['symbol']:4} {trade['direction']:4} | "
                      f"${trade['size_usd']:>6.2f} | {trade['whale_name'][:15]:15} | {out_emoji} {pnl_str}")


def main():
    parser = argparse.ArgumentParser(description='Whale Paper Trader - Track 15m-a4 & ExpressoMartini')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--scan', action='store_true', help='Scan for new whale trades')
    parser.add_argument('--resolve', action='store_true', help='Resolve pending trades')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Monitor interval in seconds')
    parser.add_argument('--reset', action='store_true', help='Reset all paper trades (start fresh)')
    
    args = parser.parse_args()
    
    if not PAPER_DB_PATH.exists():
        init_paper_db()
    
    # Reset if requested
    if args.reset:
        import os
        if PAPER_DB_PATH.exists():
            os.remove(PAPER_DB_PATH)
            print("🗑️  Deleted old paper trades database")
        init_paper_db()
        print("✅ Fresh start with $100 capital per whale")
        return
    
    with WhalePaperTrader() as trader:
        if args.status:
            # Resolve any pending trades first
            trader.resolve_trades()
            trader.print_status()
        
        elif args.scan:
            print("🔍 Scanning for whale trades from 15m-a4 & ExpressoMartini...")
            new_trades = trader.scan_whale_trades(minutes_back=60)
            
            if new_trades:
                print(f"\n✅ Found {len(new_trades)} new whale trades:")
                for t in new_trades:
                    emoji = '🟢' if t['direction'] == 'UP' else '🔴'
                    print(f"  {emoji} {t['whale']}: {t['symbol']} {t['direction']} (${t['size']:.2f})")
            else:
                print("  No new BUY trades from our whales")
            
            # Resolve pending trades
            resolved = trader.resolve_trades()
            if resolved:
                print(f"\n📊 Resolved {len(resolved)} trades")
            
            trader.print_status()
        
        elif args.resolve:
            print("🔄 Resolving pending trades...")
            resolved = trader.resolve_trades()
            if resolved:
                print(f"✅ Resolved {len(resolved)} trades")
                for r in resolved:
                    emoji = '✅' if r['outcome'] == 'WIN' else '❌'
                    print(f"  {emoji} {r['whale']}: {r['symbol']} {r['direction']} -> ${r['pnl']:+.2f}")
            else:
                print("  No trades to resolve")
            
            trader.print_status()
        
        elif args.monitor:
            print("🐋 Starting whale paper trader...")
            print(f"  Tracking: 15m-a4 ($100), ExpressoMartini ($100)")
            print(f"  Scan interval: {args.interval}s")
            print("-"*60)
            
            try:
                cycle = 0
                while True:
                    cycle += 1
                    
                    # Scan for new trades
                    new_trades = trader.scan_whale_trades(minutes_back=5)
                    
                    # Resolve pending trades
                    resolved = trader.resolve_trades()
                    
                    now = datetime.now(UTC).strftime("%H:%M:%S")
                    
                    if new_trades:
                        for t in new_trades:
                            emoji = '🟢' if t['direction'] == 'UP' else '🔴'
                            print(f"  [{now}] 🆕 {emoji} {t['whale']}: {t['symbol']} {t['direction']} (${t['size']:.2f})")
                    
                    if resolved:
                        for r in resolved:
                            emoji = '✅' if r['outcome'] == 'WIN' else '❌'
                            print(f"  [{now}] {emoji} {r['whale']}: {r['symbol']} {r['direction']} -> ${r['pnl']:+.2f}")
                    
                    # Show periodic status every 5 cycles
                    if cycle % 5 == 0:
                        stats = trader.get_whale_stats()
                        total_pnl = sum(s['total_pnl'] for s in stats.values())
                        print(f"  [{now}] 📊 Total PnL: ${total_pnl:+.2f} | "
                              f"15m-a4: ${stats.get('15m-a4', {}).get('total_pnl', 0):+.2f} | "
                              f"Espresso: ${stats.get('ExpressoMartini', {}).get('total_pnl', 0):+.2f}")
                    
                    time.sleep(args.interval)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Stopped")
                trader.print_status()
        
        else:
            # Default: scan and show status
            trader.scan_whale_trades(minutes_back=60)
            trader.resolve_trades()
            trader.print_status()


if __name__ == '__main__':
    main()
