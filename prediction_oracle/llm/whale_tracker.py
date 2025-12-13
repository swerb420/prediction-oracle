#!/usr/bin/env python3
"""
Whale Tracker - Track top Polymarket traders on crypto markets
Scrapes leaderboard, tracks activity, stores in SQLite for analysis
"""

import sqlite3
import time
import argparse
from datetime import datetime, timedelta, UTC
from typing import Optional
from pathlib import Path

from polymarket_apis import PolymarketDataClient, PolymarketGammaClient

# Crypto market keywords to filter
CRYPTO_KEYWORDS = [
    'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 
    'xrp', 'ripple', 'crypto', 'doge', 'dogecoin',
    'will bitcoin', 'will ethereum', 'will solana', 'will xrp',
    'btc price', 'eth price', 'sol price', 'xrp price',
    '15-minute', '15 minute', 'hourly', 'daily',
]

# 15-minute crypto market condition IDs (from our trader)
CRYPTO_15MIN_MARKETS = {
    'BTC': [
        '0xbd31dc8a20211944f6b70f31a54d24f6dd4e3a6e',  # Example - will need to update
    ],
    'ETH': [],
    'SOL': [],
    'XRP': [],
}

DB_PATH = Path(__file__).parent / 'data' / 'whale_tracker.db'


def init_database():
    """Initialize the whale tracker database."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Whales table - top traders we're tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proxy_wallet TEXT UNIQUE NOT NULL,
            name TEXT,
            pseudonym TEXT,
            monthly_pnl REAL,
            all_time_pnl REAL,
            rank_monthly INTEGER,
            rank_all_time INTEGER,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_crypto_trader BOOLEAN DEFAULT 0,
            crypto_trade_count INTEGER DEFAULT 0
        )
    ''')
    
    # Whale trades table - all their trades
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            whale_id INTEGER NOT NULL,
            proxy_wallet TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            market_title TEXT,
            market_slug TEXT,
            condition_id TEXT,
            side TEXT,
            size REAL,
            price REAL,
            usdc_size REAL,
            outcome TEXT,
            is_crypto BOOLEAN DEFAULT 0,
            crypto_symbol TEXT,
            transaction_hash TEXT UNIQUE,
            FOREIGN KEY (whale_id) REFERENCES whales(id)
        )
    ''')
    
    # Whale positions table - current open positions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            whale_id INTEGER NOT NULL,
            proxy_wallet TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            token_id TEXT,
            market_title TEXT,
            size REAL,
            avg_price REAL,
            current_value REAL,
            pnl REAL,
            is_crypto BOOLEAN DEFAULT 0,
            crypto_symbol TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (whale_id) REFERENCES whales(id),
            UNIQUE(whale_id, condition_id)
        )
    ''')
    
    # Whale signals - derived trading signals
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            signal_type TEXT,  -- 'BUY', 'SELL', 'ACCUMULATION', 'DISTRIBUTION'
            crypto_symbol TEXT,
            direction TEXT,  -- 'UP', 'DOWN'
            confidence REAL,
            whale_count INTEGER,
            total_size REAL,
            avg_price REAL,
            whale_wallets TEXT,  -- JSON list
            notes TEXT
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_whale_trades_timestamp ON whale_trades(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_whale_trades_crypto ON whale_trades(is_crypto, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_whale_positions_crypto ON whale_positions(is_crypto)')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")


def is_crypto_market(title: str, slug: str = "") -> tuple[bool, Optional[str]]:
    """Check if a market is crypto-related and return the symbol."""
    text = (title + " " + slug).lower()
    
    # Check for specific crypto symbols
    if 'bitcoin' in text or 'btc' in text:
        return True, 'BTC'
    elif 'ethereum' in text or ' eth ' in text or 'eth ' in text:
        return True, 'ETH'
    elif 'solana' in text or ' sol ' in text or 'sol ' in text:
        return True, 'SOL'
    elif 'xrp' in text or 'ripple' in text:
        return True, 'XRP'
    elif 'dogecoin' in text or 'doge' in text:
        return True, 'DOGE'
    elif any(kw in text for kw in ['crypto', 'cryptocurrency']):
        return True, 'CRYPTO'
    
    return False, None


def scrape_leaderboard(limit: int = 100):
    """Scrape top traders from leaderboard."""
    print(f"\n📊 Scraping top {limit} traders from leaderboard...")
    
    client = PolymarketDataClient()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get monthly top traders
    monthly_users = client.get_leaderboard_top_users(
        metric='profit', 
        window='30d', 
        limit=limit
    )
    
    # Get all-time for comparison
    alltime_users = client.get_leaderboard_top_users(
        metric='profit', 
        window='all', 
        limit=limit
    )
    
    # Build lookup for all-time ranks
    alltime_lookup = {u.proxy_wallet: (i+1, u.amount) for i, u in enumerate(alltime_users)}
    
    inserted = 0
    updated = 0
    
    for rank, user in enumerate(monthly_users, 1):
        wallet = user.proxy_wallet
        name = user.name or ""
        pseudonym = getattr(user, 'pseudonym', '') or ""
        monthly_pnl = user.amount
        
        # Get all-time stats
        alltime_rank, alltime_pnl = alltime_lookup.get(wallet, (None, None))
        
        # Check if exists
        cursor.execute('SELECT id FROM whales WHERE proxy_wallet = ?', (wallet,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE whales SET 
                    name = ?, pseudonym = ?, monthly_pnl = ?, 
                    rank_monthly = ?, all_time_pnl = ?, rank_all_time = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE proxy_wallet = ?
            ''', (name, pseudonym, monthly_pnl, rank, alltime_pnl, alltime_rank, wallet))
            updated += 1
        else:
            cursor.execute('''
                INSERT INTO whales (proxy_wallet, name, pseudonym, monthly_pnl, 
                                   rank_monthly, all_time_pnl, rank_all_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (wallet, name, pseudonym, monthly_pnl, rank, alltime_pnl, alltime_rank))
            inserted += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Leaderboard scraped: {inserted} new, {updated} updated")
    return monthly_users


def scrape_whale_activity(whale_wallet: str, hours_back: int = 24):
    """Scrape recent trading activity for a whale."""
    client = PolymarketDataClient()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get whale ID
    cursor.execute('SELECT id, name, pseudonym FROM whales WHERE proxy_wallet = ?', (whale_wallet,))
    whale_row = cursor.fetchone()
    if not whale_row:
        print(f"⚠️ Whale not found: {whale_wallet}")
        return 0
    
    whale_id, name, pseudonym = whale_row
    display_name = name or pseudonym or whale_wallet[:10]
    
    # Calculate time window
    start_time = datetime.now(UTC) - timedelta(hours=hours_back)
    
    try:
        activity = client.get_activity(
            user=whale_wallet,
            type='TRADE',
            limit=500,
            start=start_time
        )
    except Exception as e:
        print(f"⚠️ Error fetching activity for {display_name}: {e}")
        return 0
    
    crypto_count = 0
    inserted = 0
    
    for trade in activity:
        # Check if crypto market
        is_crypto, symbol = is_crypto_market(trade.title, trade.slug)
        
        if is_crypto:
            crypto_count += 1
        
        # Try to insert (skip if duplicate tx hash)
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO whale_trades 
                (whale_id, proxy_wallet, timestamp, market_title, market_slug,
                 condition_id, side, size, price, usdc_size, outcome, 
                 is_crypto, crypto_symbol, transaction_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                whale_id, whale_wallet, trade.timestamp, trade.title, trade.slug,
                trade.condition_id, trade.side, trade.size, trade.price, 
                trade.usdc_size, trade.outcome, is_crypto, symbol, trade.transaction_hash
            ))
            if cursor.rowcount > 0:
                inserted += 1
        except Exception as e:
            pass  # Skip duplicates silently
    
    # Update whale crypto trader status
    if crypto_count > 0:
        cursor.execute('''
            UPDATE whales SET 
                is_crypto_trader = 1,
                crypto_trade_count = crypto_trade_count + ?
            WHERE id = ?
        ''', (crypto_count, whale_id))
    
    conn.commit()
    conn.close()
    
    if inserted > 0:
        print(f"  📈 {display_name}: {inserted} new trades ({crypto_count} crypto)")
    
    return inserted


def scrape_all_whale_activity(top_n: int = 50, hours_back: int = 24):
    """Scrape activity for top N whales."""
    print(f"\n🐋 Scraping activity for top {top_n} whales (last {hours_back}h)...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT proxy_wallet, name, pseudonym, monthly_pnl 
        FROM whales 
        ORDER BY monthly_pnl DESC 
        LIMIT ?
    ''', (top_n,))
    
    whales = cursor.fetchall()
    conn.close()
    
    total_trades = 0
    for wallet, name, pseudo, pnl in whales:
        trades = scrape_whale_activity(wallet, hours_back)
        total_trades += trades
        time.sleep(0.5)  # Rate limiting
    
    print(f"✅ Total new trades scraped: {total_trades}")
    return total_trades


def get_crypto_whale_trades(hours_back: int = 24, symbol: Optional[str] = None):
    """Get recent crypto trades from whales."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
    
    query = '''
        SELECT 
            wt.timestamp, w.name, w.pseudonym, w.monthly_pnl,
            wt.market_title, wt.side, wt.size, wt.price, wt.crypto_symbol
        FROM whale_trades wt
        JOIN whales w ON wt.whale_id = w.id
        WHERE wt.is_crypto = 1 
        AND wt.timestamp >= ?
    '''
    params = [cutoff.isoformat()]
    
    if symbol:
        query += ' AND wt.crypto_symbol = ?'
        params.append(symbol)
    
    query += ' ORDER BY wt.timestamp DESC'
    
    cursor.execute(query, params)
    trades = cursor.fetchall()
    conn.close()
    
    return trades


def analyze_whale_signals(hours_back: int = 4):
    """Analyze recent whale activity for trading signals."""
    print(f"\n🔍 Analyzing whale signals (last {hours_back}h)...")
    
    trades = get_crypto_whale_trades(hours_back)
    
    if not trades:
        print("No crypto trades found in time window")
        return []
    
    # Group by crypto symbol and direction
    signals = {}
    
    for ts, name, pseudo, pnl, title, side, size, price, symbol in trades:
        if not symbol:
            continue
            
        # Determine direction from market title
        title_lower = title.lower()
        if 'up' in title_lower or 'above' in title_lower or 'rise' in title_lower:
            direction = 'UP' if side == 'BUY' else 'DOWN'
        elif 'down' in title_lower or 'below' in title_lower or 'fall' in title_lower:
            direction = 'DOWN' if side == 'BUY' else 'UP'
        else:
            direction = 'UNKNOWN'
        
        key = (symbol, direction)
        if key not in signals:
            signals[key] = {
                'whale_count': 0,
                'total_size': 0,
                'total_usdc': 0,
                'whales': set(),
                'trades': []
            }
        
        display_name = name or pseudo or 'Unknown'
        signals[key]['whale_count'] = len(signals[key]['whales'])
        signals[key]['total_size'] += size
        signals[key]['total_usdc'] += size * price
        signals[key]['whales'].add(display_name)
        signals[key]['trades'].append({
            'whale': display_name,
            'pnl': pnl,
            'size': size,
            'price': price,
            'time': ts
        })
    
    # Print signals
    print("\n" + "="*60)
    print("🐋 WHALE SIGNALS")
    print("="*60)
    
    for (symbol, direction), data in sorted(signals.items(), key=lambda x: -x[1]['total_usdc']):
        if direction == 'UNKNOWN':
            continue
            
        whale_names = list(data['whales'])[:5]
        print(f"\n{symbol} {direction}:")
        print(f"  Whales: {len(data['whales'])}")
        print(f"  Total Size: {data['total_size']:,.0f} shares")
        print(f"  Est. USDC: ${data['total_usdc']:,.2f}")
        print(f"  Top Whales: {', '.join(whale_names)}")
    
    return signals


def show_crypto_whales():
    """Show whales who trade crypto markets."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            name, pseudonym, proxy_wallet, monthly_pnl, 
            rank_monthly, crypto_trade_count
        FROM whales
        WHERE is_crypto_trader = 1
        ORDER BY crypto_trade_count DESC
        LIMIT 20
    ''')
    
    whales = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*60)
    print("🐋 TOP CRYPTO WHALES")
    print("="*60)
    
    for name, pseudo, wallet, pnl, rank, crypto_count in whales:
        display = name or pseudo or wallet[:12]
        print(f"#{rank or '?'} {display}: ${pnl:,.0f} PnL, {crypto_count} crypto trades")
    
    return whales


def show_recent_crypto_trades(hours: int = 4, limit: int = 20):
    """Show recent crypto trades from whales."""
    trades = get_crypto_whale_trades(hours)[:limit]
    
    print("\n" + "="*60)
    print(f"🐋 RECENT CRYPTO WHALE TRADES (last {hours}h)")
    print("="*60)
    
    for ts, name, pseudo, pnl, title, side, size, price, symbol in trades:
        display = name or pseudo or 'Unknown'
        ts_str = ts.strftime('%H:%M') if hasattr(ts, 'strftime') else str(ts)[11:16]
        print(f"{ts_str} | {symbol or '?':4} | {side:4} {size:>8.1f} @ {price:.2f} | {display[:15]}")
        print(f"       └─ {title[:50]}...")
    
    if not trades:
        print("No crypto trades found. Run --scrape first.")


def get_live_whale_positions_on_crypto():
    """Check what positions whales currently hold on crypto markets."""
    print("\n🔍 Checking live whale positions on crypto markets...")
    
    client = PolymarketDataClient()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get top crypto whales
    cursor.execute('''
        SELECT proxy_wallet, name, pseudonym, monthly_pnl 
        FROM whales 
        WHERE is_crypto_trader = 1
        ORDER BY monthly_pnl DESC 
        LIMIT 20
    ''')
    whales = cursor.fetchall()
    
    crypto_positions = []
    
    for wallet, name, pseudo, pnl in whales:
        display = name or pseudo or wallet[:10]
        try:
            positions = client.get_positions(user=wallet, size_threshold=10)
            
            for pos in positions:
                title = getattr(pos, 'title', '') or ''
                is_crypto, symbol = is_crypto_market(title)
                
                if is_crypto:
                    crypto_positions.append({
                        'whale': display,
                        'pnl': pnl,
                        'title': title,
                        'symbol': symbol,
                        'size': getattr(pos, 'size', 0),
                        'value': getattr(pos, 'current_value', 0),
                    })
        except Exception as e:
            pass
        
        time.sleep(0.3)
    
    conn.close()
    
    print("\n" + "="*60)
    print("🐋 LIVE WHALE CRYPTO POSITIONS")
    print("="*60)
    
    for p in sorted(crypto_positions, key=lambda x: -x.get('value', 0))[:20]:
        print(f"{p['symbol']:4} | ${p.get('value', 0):>10,.2f} | {p['whale'][:15]}")
        print(f"     └─ {p['title'][:50]}...")
    
    return crypto_positions


def main():
    parser = argparse.ArgumentParser(description='Whale Tracker - Track top Polymarket crypto traders')
    parser.add_argument('--init', action='store_true', help='Initialize database')
    parser.add_argument('--scrape', action='store_true', help='Scrape leaderboard and activity')
    parser.add_argument('--top', type=int, default=50, help='Number of top whales to track')
    parser.add_argument('--hours', type=int, default=24, help='Hours of history to scrape')
    parser.add_argument('--signals', action='store_true', help='Analyze and show trading signals')
    parser.add_argument('--whales', action='store_true', help='Show top crypto whales')
    parser.add_argument('--trades', action='store_true', help='Show recent crypto trades')
    parser.add_argument('--positions', action='store_true', help='Check live crypto positions')
    parser.add_argument('--all', action='store_true', help='Run full scrape and analysis')
    
    args = parser.parse_args()
    
    # Always ensure DB exists
    if not DB_PATH.exists() or args.init:
        init_database()
    
    if args.scrape or args.all:
        scrape_leaderboard(args.top)
        scrape_all_whale_activity(args.top, args.hours)
    
    if args.signals or args.all:
        analyze_whale_signals(args.hours)
    
    if args.whales or args.all:
        show_crypto_whales()
    
    if args.trades or args.all:
        show_recent_crypto_trades(args.hours)
    
    if args.positions:
        get_live_whale_positions_on_crypto()
    
    if not any([args.scrape, args.signals, args.whales, args.trades, args.positions, args.all, args.init]):
        parser.print_help()
        print("\n💡 Quick start: python whale_tracker.py --all --hours 24")


if __name__ == '__main__':
    main()
