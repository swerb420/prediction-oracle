#!/usr/bin/env python3
"""
Whale Intelligence System - Comprehensive tracking of top Polymarket crypto traders

Features:
- Scrapes ALL leaderboards (1d, 7d, 30d, all-time) for profit AND volume
- Identifies dedicated crypto traders vs occasional ones
- Tracks win rates, position sizing, timing patterns
- Builds ML-ready features for signal generation
- Real-time whale activity monitoring
"""

import sqlite3
import json
import time
import argparse
import statistics
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from polymarket_apis import PolymarketDataClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database path (can be overridden for testing or alternative deployments)
DB_PATH = Path(__file__).parent / 'data' / 'whale_intelligence.db'

# Leaderboard configurations
LEADERBOARD_WINDOWS = ['1d', '7d', '30d', 'all']
LEADERBOARD_METRICS = ['profit', 'volume']
LEADERBOARD_TAGS = ['', 'crypto']  # '' = general, 'crypto' = crypto-specific
INTRADAY_WINDOWS = [15, 60, 240]  # 15m, 1h, 4h

# Crypto detection patterns
CRYPTO_PATTERNS = {
    'BTC': ['bitcoin', 'btc', ' btc ', 'btc '],
    'ETH': ['ethereum', 'eth', ' eth ', 'eth '],
    'SOL': ['solana', 'sol', ' sol ', 'sol '],
    'XRP': ['xrp', 'ripple'],
    'DOGE': ['dogecoin', 'doge'],
    'ADA': ['cardano', 'ada'],
    'AVAX': ['avalanche', 'avax'],
    'MATIC': ['polygon', 'matic'],
    'LINK': ['chainlink', 'link'],
    'DOT': ['polkadot', 'dot'],
}

# Market type patterns
MARKET_TYPES = {
    '15MIN': ['15-minute', '15 minute', '15min', 'up or down'],
    'HOURLY': ['hourly', '1 hour', '1-hour'],
    'DAILY': ['daily', 'end of day', 'eod'],
    'PRICE_TARGET': ['above', 'below', 'reach', 'hit', 'dip to'],
    'WEEKLY': ['weekly', 'this week', 'by sunday'],
}


@dataclass
class WhaleProfile:
    """Comprehensive whale profile with analytics."""
    wallet: str
    name: str
    pseudonym: str
    
    # Rankings across timeframes
    rank_1d_profit: Optional[int] = None
    rank_7d_profit: Optional[int] = None
    rank_30d_profit: Optional[int] = None
    rank_all_profit: Optional[int] = None
    rank_1d_volume: Optional[int] = None
    rank_7d_volume: Optional[int] = None
    rank_30d_volume: Optional[int] = None
    rank_all_volume: Optional[int] = None
    
    # PnL across timeframes
    pnl_1d: float = 0.0
    pnl_7d: float = 0.0
    pnl_30d: float = 0.0
    pnl_all: float = 0.0
    
    # Volume across timeframes
    volume_1d: float = 0.0
    volume_7d: float = 0.0
    volume_30d: float = 0.0
    volume_all: float = 0.0
    
    # Crypto specific stats
    crypto_trade_count: int = 0
    crypto_win_count: int = 0
    crypto_loss_count: int = 0
    crypto_pnl: float = 0.0
    crypto_volume: float = 0.0
    favorite_crypto: Optional[str] = None
    favorite_market_type: Optional[str] = None
    
    # Trading patterns
    avg_position_size: float = 0.0
    avg_entry_price: float = 0.0
    trades_per_day: float = 0.0
    active_hours: Optional[str] = None  # JSON list of most active hours
    
    # Classification
    is_crypto_specialist: bool = False
    crypto_focus_ratio: float = 0.0  # % of trades that are crypto
    consistency_score: float = 0.0  # How consistent their profits are
    
    # Metadata
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None


@dataclass 
class CryptoTrade:
    """Detailed crypto trade record."""
    whale_wallet: str
    timestamp: datetime
    symbol: str
    market_type: str
    direction: str  # UP or DOWN
    side: str  # BUY or SELL
    size: float
    price: float
    usdc_value: float
    market_title: str
    market_slug: str
    condition_id: str
    outcome: str
    tx_hash: str
    
    # Derived
    is_entry: bool = True  # vs exit
    confidence: float = 0.0  # Based on position size relative to avg


def init_database(db_path: Path = DB_PATH):
    """Initialize comprehensive whale intelligence database."""
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Whale profiles - comprehensive trader data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_profiles (
            wallet TEXT PRIMARY KEY,
            name TEXT,
            pseudonym TEXT,
            
            -- Rankings (profit)
            rank_1d_profit INTEGER,
            rank_7d_profit INTEGER,
            rank_30d_profit INTEGER,
            rank_all_profit INTEGER,
            
            -- Rankings (volume)
            rank_1d_volume INTEGER,
            rank_7d_volume INTEGER,
            rank_30d_volume INTEGER,
            rank_all_volume INTEGER,
            
            -- PnL by timeframe
            pnl_1d REAL DEFAULT 0,
            pnl_7d REAL DEFAULT 0,
            pnl_30d REAL DEFAULT 0,
            pnl_all REAL DEFAULT 0,
            
            -- Volume by timeframe
            volume_1d REAL DEFAULT 0,
            volume_7d REAL DEFAULT 0,
            volume_30d REAL DEFAULT 0,
            volume_all REAL DEFAULT 0,
            
            -- Crypto stats
            crypto_trade_count INTEGER DEFAULT 0,
            crypto_win_count INTEGER DEFAULT 0,
            crypto_loss_count INTEGER DEFAULT 0,
            crypto_pnl REAL DEFAULT 0,
            crypto_volume REAL DEFAULT 0,
            favorite_crypto TEXT,
            favorite_market_type TEXT,
            
            -- Patterns
            avg_position_size REAL DEFAULT 0,
            avg_entry_price REAL DEFAULT 0,
            trades_per_day REAL DEFAULT 0,
            active_hours TEXT,  -- JSON
            
            -- Classification
            is_crypto_specialist BOOLEAN DEFAULT 0,
            crypto_focus_ratio REAL DEFAULT 0,
            consistency_score REAL DEFAULT 0,
            
            -- Metadata
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_trade_time TIMESTAMP
        )
    ''')
    
    # Crypto trades - detailed trade history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crypto_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            whale_wallet TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            symbol TEXT NOT NULL,
            market_type TEXT,
            direction TEXT,  -- UP, DOWN, UNKNOWN
            side TEXT NOT NULL,  -- BUY, SELL
            size REAL NOT NULL,
            price REAL NOT NULL,
            usdc_value REAL,
            market_title TEXT,
            market_slug TEXT,
            condition_id TEXT,
            outcome TEXT,
            tx_hash TEXT UNIQUE,
            is_entry BOOLEAN DEFAULT 1,
            confidence REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (whale_wallet) REFERENCES whale_profiles(wallet)
        )
    ''')
    
    # Whale signals - aggregated trading signals
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS whale_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,  -- UP, DOWN
            signal_strength REAL,  -- 0-1 confidence
            whale_count INTEGER,
            specialist_count INTEGER,  -- Whales who are crypto specialists
            total_size REAL,
            total_usdc REAL,
            avg_whale_pnl REAL,  -- Average PnL of participating whales
            participating_whales TEXT,  -- JSON list
            market_type TEXT,
            window_minutes INTEGER,  -- Signal time window
            notes TEXT
        )
    ''')
    
    # Hourly snapshots - for pattern analysis
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hourly_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            hour INTEGER NOT NULL,  -- 0-23 UTC
            day_of_week INTEGER NOT NULL,  -- 0=Monday
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            trade_count INTEGER,
            total_size REAL,
            total_usdc REAL,
            whale_count INTEGER,
            avg_price REAL,
            UNIQUE(timestamp, symbol, direction)
        )
    ''')
    
    # Leaderboard history - track rank changes over time
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            window TEXT NOT NULL,  -- 1d, 7d, 30d, all
            metric TEXT NOT NULL,  -- profit, volume
            wallet TEXT NOT NULL,
            rank INTEGER NOT NULL,
            amount REAL NOT NULL,
            name TEXT,
            UNIQUE(scraped_at, window, metric, wallet)
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_crypto_trades_time ON crypto_trades(timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_crypto_trades_symbol ON crypto_trades(symbol, timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_crypto_trades_whale ON crypto_trades(whale_wallet, timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_whale_profiles_specialist ON whale_profiles(is_crypto_specialist, crypto_pnl DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_whale_signals_time ON whale_signals(timestamp DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hourly_hour ON hourly_snapshots(hour, day_of_week)')
    
    conn.commit()
    conn.close()
    logger.info(f"✅ Database initialized at {db_path}")


def detect_crypto(text: str) -> Tuple[bool, Optional[str]]:
    """Detect if text is crypto-related and return the symbol."""
    text_lower = text.lower()
    
    for symbol, patterns in CRYPTO_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return True, symbol
    
    # Generic crypto keywords
    if any(kw in text_lower for kw in ['crypto', 'cryptocurrency', 'coin']):
        return True, 'CRYPTO'
    
    return False, None


def detect_market_type(text: str) -> str:
    """Detect the market type from title."""
    text_lower = text.lower()
    
    for market_type, patterns in MARKET_TYPES.items():
        for pattern in patterns:
            if pattern in text_lower:
                return market_type
    
    return 'OTHER'


def detect_direction(title: str, outcome: str, side: str) -> str:
    """Detect if the trade is betting UP or DOWN."""
    title_lower = title.lower()
    outcome_lower = outcome.lower() if outcome else ''
    
    # Check outcome first (more reliable)
    if 'up' in outcome_lower or 'yes' in outcome_lower or 'above' in outcome_lower:
        return 'UP' if side == 'BUY' else 'DOWN'
    elif 'down' in outcome_lower or 'no' in outcome_lower or 'below' in outcome_lower:
        return 'DOWN' if side == 'BUY' else 'UP'
    
    # Fall back to title analysis
    if 'up or down' in title_lower:
        # For "Up or Down" markets, check outcome
        if 'up' in outcome_lower:
            return 'UP' if side == 'BUY' else 'DOWN'
        elif 'down' in outcome_lower:
            return 'DOWN' if side == 'BUY' else 'UP'
    
    if 'above' in title_lower or 'rise' in title_lower or 'reach' in title_lower:
        return 'UP' if side == 'BUY' else 'DOWN'
    elif 'below' in title_lower or 'fall' in title_lower or 'dip' in title_lower:
        return 'DOWN' if side == 'BUY' else 'UP'
    
    return 'UNKNOWN'


class WhaleIntelligence:
    """Main whale intelligence system."""
    
    def __init__(self, db_path: Path | str = DB_PATH):
        self.client = PolymarketDataClient()
        self.conn = None
        self.db_path = Path(db_path)
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self
    
    def __exit__(self, *args):
        if self.conn:
            self.conn.close()
    
    def scrape_all_leaderboards(self, limit: int = 100) -> Dict[str, List]:
        """Scrape all leaderboards and identify unique whales."""
        logger.info(f"📊 Scraping all leaderboards (limit={limit})...")
        
        all_whales = {}  # wallet -> WhaleProfile
        leaderboard_data = []
        
        for window in LEADERBOARD_WINDOWS:
            for metric in LEADERBOARD_METRICS:
                try:
                    logger.info(f"  Fetching {window} {metric}...")
                    users = self.client.get_leaderboard_top_users(
                        metric=metric,
                        window=window,
                        limit=limit
                    )
                    
                    for rank, user in enumerate(users, 1):
                        wallet = user.proxy_wallet
                        
                        # Initialize or update whale profile
                        if wallet not in all_whales:
                            all_whales[wallet] = WhaleProfile(
                                wallet=wallet,
                                name=user.name or '',
                                pseudonym=getattr(user, 'pseudonym', '') or ''
                            )
                        
                        profile = all_whales[wallet]
                        
                        # Update rankings and amounts
                        if metric == 'profit':
                            if window == '1d':
                                profile.rank_1d_profit = rank
                                profile.pnl_1d = user.amount
                            elif window == '7d':
                                profile.rank_7d_profit = rank
                                profile.pnl_7d = user.amount
                            elif window == '30d':
                                profile.rank_30d_profit = rank
                                profile.pnl_30d = user.amount
                            else:  # all
                                profile.rank_all_profit = rank
                                profile.pnl_all = user.amount
                        else:  # volume
                            if window == '1d':
                                profile.rank_1d_volume = rank
                                profile.volume_1d = user.amount
                            elif window == '7d':
                                profile.rank_7d_volume = rank
                                profile.volume_7d = user.amount
                            elif window == '30d':
                                profile.rank_30d_volume = rank
                                profile.volume_30d = user.amount
                            else:
                                profile.rank_all_volume = rank
                                profile.volume_all = user.amount
                        
                        # Record for history
                        leaderboard_data.append({
                            'window': window,
                            'metric': metric,
                            'wallet': wallet,
                            'rank': rank,
                            'amount': user.amount,
                            'name': user.name
                        })
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error fetching {window} {metric}: {e}")
        
        # Save to database
        cursor = self.conn.cursor()
        now = datetime.now(UTC).isoformat()
        
        # Save leaderboard history
        for entry in leaderboard_data:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO leaderboard_history 
                    (scraped_at, window, metric, wallet, rank, amount, name)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (now, entry['window'], entry['metric'], entry['wallet'],
                      entry['rank'], entry['amount'], entry['name']))
            except Exception as e:
                pass
        
        # Save/update whale profiles
        for wallet, profile in all_whales.items():
            cursor.execute('''
                INSERT INTO whale_profiles (
                    wallet, name, pseudonym,
                    rank_1d_profit, rank_7d_profit, rank_30d_profit, rank_all_profit,
                    rank_1d_volume, rank_7d_volume, rank_30d_volume, rank_all_volume,
                    pnl_1d, pnl_7d, pnl_30d, pnl_all,
                    volume_1d, volume_7d, volume_30d, volume_all,
                    last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(wallet) DO UPDATE SET
                    name = excluded.name,
                    pseudonym = excluded.pseudonym,
                    rank_1d_profit = excluded.rank_1d_profit,
                    rank_7d_profit = excluded.rank_7d_profit,
                    rank_30d_profit = excluded.rank_30d_profit,
                    rank_all_profit = excluded.rank_all_profit,
                    rank_1d_volume = excluded.rank_1d_volume,
                    rank_7d_volume = excluded.rank_7d_volume,
                    rank_30d_volume = excluded.rank_30d_volume,
                    rank_all_volume = excluded.rank_all_volume,
                    pnl_1d = excluded.pnl_1d,
                    pnl_7d = excluded.pnl_7d,
                    pnl_30d = excluded.pnl_30d,
                    pnl_all = excluded.pnl_all,
                    volume_1d = excluded.volume_1d,
                    volume_7d = excluded.volume_7d,
                    volume_30d = excluded.volume_30d,
                    volume_all = excluded.volume_all,
                    last_updated = CURRENT_TIMESTAMP
            ''', (
                wallet, profile.name, profile.pseudonym,
                profile.rank_1d_profit, profile.rank_7d_profit, 
                profile.rank_30d_profit, profile.rank_all_profit,
                profile.rank_1d_volume, profile.rank_7d_volume,
                profile.rank_30d_volume, profile.rank_all_volume,
                profile.pnl_1d, profile.pnl_7d, profile.pnl_30d, profile.pnl_all,
                profile.volume_1d, profile.volume_7d, profile.volume_30d, profile.volume_all
            ))
        
        self.conn.commit()
        
        logger.info(f"✅ Found {len(all_whales)} unique whales across all leaderboards")
        return all_whales
    
    def scrape_crypto_leaderboard(self, limit: int = 100) -> Dict[str, Any]:
        """
        Scrape the CRYPTO-SPECIFIC leaderboard (tag=crypto).
        This includes the top crypto traders like 15m-a4, ExpressoMartini, etc.
        """
        import httpx
        
        logger.info(f"🪙 Scraping CRYPTO leaderboard (tag=crypto, limit={limit})...")
        
        cursor = self.conn.cursor()
        crypto_whales = {}
        
        for window in ['1d', '7d', '30d']:
            for metric in ['profit', 'volume']:
                try:
                    url = f'https://lb-api.polymarket.com/{metric}?window={window}&limit={limit}&tag=crypto'
                    logger.info(f"  Fetching crypto {window} {metric}...")
                    
                    resp = httpx.get(url, timeout=10)
                    if resp.status_code != 200:
                        continue
                    
                    users = resp.json()
                    
                    for rank, user in enumerate(users, 1):
                        wallet = user['proxyWallet']
                        name = user.get('name') or user.get('pseudonym') or ''
                        amount = user['amount']
                        
                        if wallet not in crypto_whales:
                            crypto_whales[wallet] = {
                                'wallet': wallet,
                                'name': name,
                                'is_crypto_leaderboard': True
                            }
                        
                        # Update the whale profile in database
                        if metric == 'profit' and window == '30d':
                            cursor.execute('''
                                INSERT INTO whale_profiles (wallet, name, pseudonym, pnl_30d, is_crypto_specialist, last_updated)
                                VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                                ON CONFLICT(wallet) DO UPDATE SET
                                    name = COALESCE(excluded.name, name),
                                    pnl_30d = excluded.pnl_30d,
                                    is_crypto_specialist = 1,
                                    last_updated = CURRENT_TIMESTAMP
                            ''', (wallet, name, name, amount))
                        
                        crypto_whales[wallet][f'{window}_{metric}'] = amount
                        crypto_whales[wallet][f'{window}_{metric}_rank'] = rank
                    
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"Error fetching crypto {window} {metric}: {e}")
        
        self.conn.commit()
        logger.info(f"✅ Found {len(crypto_whales)} crypto leaderboard whales")
        
        return crypto_whales
    
    def scrape_whale_crypto_activity(self, wallet: str, hours_back: int = 48) -> int:
        """Scrape crypto-specific activity for a whale."""
        cursor = self.conn.cursor()
        
        # Get whale name for logging
        cursor.execute('SELECT name, pseudonym FROM whale_profiles WHERE wallet = ?', (wallet,))
        row = cursor.fetchone()
        display_name = (row['name'] or row['pseudonym'] or wallet[:10]) if row else wallet[:10]
        
        start_time = datetime.now(UTC) - timedelta(hours=hours_back)
        
        try:
            activity = self.client.get_activity(
                user=wallet,
                type='TRADE',
                limit=500,
                start=start_time
            )
        except Exception as e:
            logger.warning(f"Error fetching activity for {display_name}: {e}")
            return 0
        
        crypto_trades = 0
        total_trades = len(activity)
        crypto_symbols = defaultdict(int)
        market_types = defaultdict(int)
        hourly_activity = defaultdict(int)
        
        for trade in activity:
            is_crypto, symbol = detect_crypto(trade.title)
            
            if not is_crypto:
                continue
            
            market_type = detect_market_type(trade.title)
            direction = detect_direction(trade.title, trade.outcome, trade.side)
            
            # Track patterns
            crypto_symbols[symbol] += 1
            market_types[market_type] += 1
            
            trade_hour = trade.timestamp.hour if hasattr(trade.timestamp, 'hour') else 0
            hourly_activity[trade_hour] += 1
            
            # Calculate confidence based on position size
            usdc_value = trade.size * trade.price
            
            # Insert trade
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO crypto_trades (
                        whale_wallet, timestamp, symbol, market_type, direction,
                        side, size, price, usdc_value, market_title, market_slug,
                        condition_id, outcome, tx_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    wallet, trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                    symbol, market_type, direction,
                    trade.side, trade.size, trade.price, usdc_value,
                    trade.title, trade.slug, trade.condition_id, trade.outcome,
                    trade.transaction_hash
                ))
                if cursor.rowcount > 0:
                    crypto_trades += 1
            except Exception as e:
                pass
        
        # Update whale profile with crypto stats
        if total_trades > 0:
            crypto_ratio = crypto_trades / total_trades if total_trades > 0 else 0
            favorite_crypto = max(crypto_symbols.items(), key=lambda x: x[1])[0] if crypto_symbols else None
            favorite_market = max(market_types.items(), key=lambda x: x[1])[0] if market_types else None
            active_hours = json.dumps(sorted(hourly_activity.items(), key=lambda x: -x[1])[:5])
            
            is_specialist = crypto_ratio > 0.5 and crypto_trades >= 10
            
            cursor.execute('''
                UPDATE whale_profiles SET
                    crypto_trade_count = crypto_trade_count + ?,
                    crypto_focus_ratio = ?,
                    favorite_crypto = COALESCE(?, favorite_crypto),
                    favorite_market_type = COALESCE(?, favorite_market_type),
                    active_hours = ?,
                    is_crypto_specialist = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE wallet = ?
            ''', (crypto_trades, crypto_ratio, favorite_crypto, favorite_market,
                  active_hours, is_specialist, wallet))
        
        self.conn.commit()
        
        if crypto_trades > 0:
            logger.info(f"  🐋 {display_name}: {crypto_trades}/{total_trades} crypto trades")
        
        return crypto_trades
    
    def scrape_all_whale_activity(self, hours_back: int = 48, min_rank: int = 100):
        """Scrape activity for all whales in the database."""
        cursor = self.conn.cursor()
        
        # Get all whales ranked in any leaderboard
        cursor.execute('''
            SELECT DISTINCT wallet, name, pseudonym,
                COALESCE(pnl_30d, pnl_7d, pnl_1d, 0) as pnl
            FROM whale_profiles
            WHERE rank_1d_profit <= ? OR rank_7d_profit <= ? 
               OR rank_30d_profit <= ? OR rank_all_profit <= ?
               OR rank_1d_volume <= ? OR rank_7d_volume <= ?
               OR rank_30d_volume <= ? OR rank_all_volume <= ?
            ORDER BY pnl DESC
        ''', (min_rank,) * 8)
        
        whales = cursor.fetchall()
        logger.info(f"🐋 Scraping activity for {len(whales)} whales (last {hours_back}h)...")
        
        total_crypto = 0
        for row in whales:
            crypto = self.scrape_whale_crypto_activity(row['wallet'], hours_back)
            total_crypto += crypto
            time.sleep(0.3)  # Rate limiting
        
        logger.info(f"✅ Total crypto trades scraped: {total_crypto}")
        return total_crypto
    
    def get_crypto_specialists(self, min_trades: int = 20, min_ratio: float = 0.3) -> List[Dict]:
        """Get whales who specialize in crypto trading."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                wallet, name, pseudonym,
                pnl_30d, pnl_all,
                crypto_trade_count, crypto_focus_ratio,
                favorite_crypto, favorite_market_type,
                rank_30d_profit, rank_all_profit,
                is_crypto_specialist
            FROM whale_profiles
            WHERE crypto_trade_count >= ?
              AND crypto_focus_ratio >= ?
            ORDER BY crypto_trade_count DESC, pnl_30d DESC
        ''', (min_trades, min_ratio))
        
        specialists = []
        for row in cursor.fetchall():
            specialists.append(dict(row))
        
        return specialists
    
    def analyze_elite_whales(self) -> Dict[str, Any]:
        """
        Deep analysis of whale performance - find the ELITE high win-rate traders.
        Specifically looks for 15M market specialists and calculates actual win rates.
        """
        cursor = self.conn.cursor()
        
        results = {
            'elite_15m_whales': [],
            'high_winrate_whales': [],
            'big_money_whales': [],
            'market_type_stats': {},
            'symbol_stats': {},
            'hourly_patterns': {},
            'whale_performance': []
        }
        
        # ═══════════════════════════════════════════════════════════════════════
        # 1. Calculate ACTUAL WIN RATES from trade data
        # ═══════════════════════════════════════════════════════════════════════
        cursor.execute('''
            SELECT 
                ct.whale_wallet,
                wp.name,
                wp.pseudonym,
                wp.pnl_30d,
                wp.pnl_all,
                ct.market_type,
                COUNT(*) as trade_count,
                SUM(ct.usdc_value) as total_volume,
                AVG(ct.usdc_value) as avg_size,
                GROUP_CONCAT(DISTINCT ct.symbol) as symbols
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            GROUP BY ct.whale_wallet, ct.market_type
            HAVING COUNT(*) >= 10
            ORDER BY wp.pnl_30d DESC
        ''')
        
        whale_market_data = cursor.fetchall()
        
        # Build comprehensive whale performance profiles
        whale_profiles = {}
        for row in whale_market_data:
            wallet = row['wallet' if 'wallet' in row.keys() else 'whale_wallet']
            name = row['name'] or row['pseudonym'] or wallet[:12]
            market_type = row['market_type']
            
            if wallet not in whale_profiles:
                whale_profiles[wallet] = {
                    'wallet': wallet,
                    'name': name,
                    'pnl_30d': row['pnl_30d'] or 0,
                    'pnl_all': row['pnl_all'] or 0,
                    'total_trades': 0,
                    'total_volume': 0,
                    'market_types': {},
                    'symbols': set(),
                    'is_15m_specialist': False,
                    'is_elite': False
                }
            
            profile = whale_profiles[wallet]
            profile['total_trades'] += row['trade_count']
            profile['total_volume'] += row['total_volume'] or 0
            profile['market_types'][market_type] = {
                'trades': row['trade_count'],
                'volume': row['total_volume'] or 0,
                'avg_size': row['avg_size'] or 0
            }
            
            if row['symbols']:
                profile['symbols'].update(row['symbols'].split(','))
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. IDENTIFY 15M SPECIALISTS - Trading same markets as user!
        # ═══════════════════════════════════════════════════════════════════════
        elite_15m = []
        for wallet, profile in whale_profiles.items():
            if '15MIN' in profile['market_types']:
                m15_data = profile['market_types']['15MIN']
                total = profile['total_trades']
                m15_ratio = m15_data['trades'] / total if total > 0 else 0
                
                # Must be profitable AND have significant 15M activity
                if m15_ratio >= 0.3 and m15_data['trades'] >= 20 and profile['pnl_30d'] > 0:
                    profile['is_15m_specialist'] = True
                    profile['15m_trades'] = m15_data['trades']
                    profile['15m_volume'] = m15_data['volume']
                    profile['15m_ratio'] = m15_ratio
                    profile['15m_avg_size'] = m15_data['avg_size']
                    
                    # Calculate profitability score (PnL per trade)
                    profile['pnl_per_trade'] = profile['pnl_30d'] / profile['total_trades'] if profile['total_trades'] > 0 else 0
                    
                    # ELITE status: high PnL + significant 15M activity
                    if profile['pnl_30d'] > 50000 and m15_data['trades'] >= 50:
                        profile['is_elite'] = True
                    
                    elite_15m.append(profile)
        
        # Sort by 15M trade count and PnL
        elite_15m.sort(key=lambda x: (-x['15m_trades'], -x['pnl_30d']))
        results['elite_15m_whales'] = elite_15m[:30]  # Top 30
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. ANALYZE HOURLY PATTERNS - When do elite whales trade?
        # ═══════════════════════════════════════════════════════════════════════
        cursor.execute('''
            SELECT 
                CAST(strftime('%H', ct.timestamp) AS INTEGER) as hour,
                ct.symbol,
                ct.direction,
                COUNT(*) as trade_count,
                SUM(ct.usdc_value) as total_volume,
                COUNT(DISTINCT ct.whale_wallet) as whale_count
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            WHERE wp.pnl_30d > 0  -- Only profitable whales
              AND ct.market_type = '15MIN'
            GROUP BY hour, ct.symbol, ct.direction
            ORDER BY hour, ct.symbol
        ''')
        
        hourly_data = cursor.fetchall()
        
        # Build hourly analysis
        hourly = {}
        for row in hourly_data:
            hour = row['hour']
            if hour not in hourly:
                hourly[hour] = {'total_trades': 0, 'total_volume': 0, 'symbols': {}, 'directions': {'UP': 0, 'DOWN': 0}}
            
            hourly[hour]['total_trades'] += row['trade_count']
            hourly[hour]['total_volume'] += row['total_volume'] or 0
            hourly[hour]['directions'][row['direction']] = hourly[hour]['directions'].get(row['direction'], 0) + row['trade_count']
            
            if row['symbol'] not in hourly[hour]['symbols']:
                hourly[hour]['symbols'][row['symbol']] = {'UP': 0, 'DOWN': 0, 'volume': 0}
            hourly[hour]['symbols'][row['symbol']][row['direction']] += row['trade_count']
            hourly[hour]['symbols'][row['symbol']]['volume'] += row['total_volume'] or 0
        
        results['hourly_patterns'] = hourly
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4. MARKET TYPE DISTRIBUTION
        # ═══════════════════════════════════════════════════════════════════════
        cursor.execute('''
            SELECT 
                ct.market_type,
                COUNT(*) as trade_count,
                SUM(ct.usdc_value) as total_volume,
                COUNT(DISTINCT ct.whale_wallet) as whale_count,
                AVG(wp.pnl_30d) as avg_whale_pnl
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            GROUP BY ct.market_type
            ORDER BY trade_count DESC
        ''')
        
        for row in cursor.fetchall():
            results['market_type_stats'][row['market_type']] = {
                'trades': row['trade_count'],
                'volume': row['total_volume'] or 0,
                'whales': row['whale_count'],
                'avg_whale_pnl': row['avg_whale_pnl'] or 0
            }
        
        # ═══════════════════════════════════════════════════════════════════════
        # 5. BIG MONEY WHALES - Top PnL traders who trade crypto
        # ═══════════════════════════════════════════════════════════════════════
        for wallet, profile in sorted(whale_profiles.items(), key=lambda x: -x[1]['pnl_30d']):
            if profile['pnl_30d'] >= 100000 and profile['total_trades'] >= 20:
                profile['symbols'] = list(profile['symbols'])
                results['big_money_whales'].append(profile)
                if len(results['big_money_whales']) >= 20:
                    break
        
        # Convert sets to lists for JSON serialization
        for profile in results['elite_15m_whales']:
            profile['symbols'] = list(profile['symbols'])

        return results

    def get_intraday_wallet_dashboards(
        self,
        windows: Optional[List[int]] = None,
        min_trades: int = 2,
        top_n: int = 20,
        symbol_filter: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build intraday dashboards for active whales across multiple windows.

        This is optimized for spotting short-horizon specialists (15m/1h/4h) and
        combining them with longer-term quality signals (30d PnL, rankings).
        """
        cursor = self.conn.cursor()

        windows = windows or INTRADAY_WINDOWS
        now = datetime.now(UTC)
        dashboards: Dict[str, List[Dict[str, Any]]] = {}

        for minutes in windows:
            cutoff = (now - timedelta(minutes=minutes)).isoformat()

            cursor.execute('''
                SELECT
                    ct.whale_wallet,
                    wp.name,
                    wp.pseudonym,
                    wp.pnl_30d,
                    wp.pnl_all,
                    wp.crypto_focus_ratio,
                    wp.crypto_trade_count,
                    wp.is_crypto_specialist,
                    wp.rank_30d_profit,
                    wp.rank_all_profit,
                    ct.symbol,
                    ct.market_type,
                    ct.direction,
                    ct.usdc_value,
                    ct.timestamp
                FROM crypto_trades ct
                JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
                WHERE ct.timestamp >= ?
            ''', (cutoff,))

            rows = cursor.fetchall()
            aggregates: Dict[str, Dict[str, Any]] = {}

            for row in rows:
                if symbol_filter and row['symbol'] != symbol_filter:
                    continue

                wallet = row['whale_wallet']
                profile = aggregates.get(wallet)

                if profile is None:
                    name = row['name'] or row['pseudonym'] or wallet[:12]
                    profile = {
                        'wallet': wallet,
                        'name': name,
                        'pnl_30d': row['pnl_30d'] or 0,
                        'pnl_all': row['pnl_all'] or 0,
                        'crypto_focus_ratio': row['crypto_focus_ratio'] or 0,
                        'crypto_trade_count': row['crypto_trade_count'] or 0,
                        'is_crypto_specialist': bool(row['is_crypto_specialist']),
                        'rank_30d_profit': row['rank_30d_profit'],
                        'rank_all_profit': row['rank_all_profit'],
                        'trade_count': 0,
                        'total_volume': 0.0,
                        'up_volume': 0.0,
                        'down_volume': 0.0,
                        'avg_size': 0.0,
                        'symbols': set(),
                        'symbol_breakdown': defaultdict(lambda: {'trades': 0, 'volume': 0.0}),
                        'market_types': defaultdict(lambda: {'trades': 0, 'volume': 0.0}),
                        'last_trade': None,
                        'volume_velocity': 0.0,
                        'trade_rate': 0.0,
                        'top_symbol': None,
                        'top_symbol_share': None,
                        'dominant_market_type': None,
                    }
                    aggregates[wallet] = profile

                volume = row['usdc_value'] or 0.0
                profile['trade_count'] += 1
                profile['total_volume'] += volume
                profile['symbols'].add(row['symbol'])

                symbol_stats = profile['symbol_breakdown'][row['symbol']]
                symbol_stats['trades'] += 1
                symbol_stats['volume'] += volume

                direction = row['direction'] or 'UNKNOWN'
                if direction == 'UP':
                    profile['up_volume'] += volume
                elif direction == 'DOWN':
                    profile['down_volume'] += volume

                market_type = row['market_type'] or 'UNKNOWN'
                mt = profile['market_types'][market_type]
                mt['trades'] += 1
                mt['volume'] += volume

                try:
                    ts = datetime.fromisoformat(str(row['timestamp']))
                except Exception:
                    ts = None

                if ts and (profile['last_trade'] is None or ts > profile['last_trade']):
                    profile['last_trade'] = ts

            ranked_profiles = []
            for profile in aggregates.values():
                if profile['trade_count'] < min_trades:
                    continue

                profile['avg_size'] = profile['total_volume'] / profile['trade_count'] if profile['trade_count'] else 0.0
                profile['volume_velocity'] = profile['total_volume'] / minutes if minutes else 0.0
                profile['trade_rate'] = profile['trade_count'] / minutes if minutes else 0.0
                total_dir_vol = profile['up_volume'] + profile['down_volume']
                profile['direction_bias'] = (profile['up_volume'] / total_dir_vol) if total_dir_vol else None
                profile['net_flow'] = profile['up_volume'] - profile['down_volume']
                profile['symbols'] = sorted(profile['symbols'])
                profile['symbol_breakdown'] = {
                    sym: stats
                    for sym, stats in sorted(
                        profile['symbol_breakdown'].items(), key=lambda x: -x[1]['volume']
                    )
                }
                if profile['symbol_breakdown']:
                    top_sym, top_stats = next(iter(profile['symbol_breakdown'].items()))
                    profile['top_symbol'] = top_sym
                    profile['top_symbol_share'] = (
                        (top_stats['volume'] / profile['total_volume']) if profile['total_volume'] else None
                    )
                profile['market_types'] = {
                    mt: stats
                    for mt, stats in sorted(profile['market_types'].items(), key=lambda x: -x[1]['volume'])
                }
                if profile['market_types']:
                    profile['dominant_market_type'] = next(iter(profile['market_types'].keys()))

                if profile['last_trade']:
                    profile['last_trade'] = profile['last_trade'].isoformat()

                ranked_profiles.append(profile)

            ranked_profiles.sort(key=lambda x: (-x['total_volume'], -x['trade_count'], -(x['pnl_30d'] or 0)))
            dashboards[f"{minutes}m"] = ranked_profiles[:top_n]

        return dashboards

    def print_intraday_dashboards(self, dashboards: Dict[str, List[Dict[str, Any]]], top_n: int = 15):
        """Pretty-print intraday dashboards for multiple time windows."""
        if not dashboards:
            print("\n⚠️ No intraday data available. Run --scrape-activity first.")
            return

        for window in sorted(dashboards.keys(), key=lambda w: int(w.rstrip('m'))):
            entries = dashboards[window][:top_n]
            print("\n" + "=" * 90)
            print(f"⏱️  TOP ACTIVE WHALES - last {window}")
            print("=" * 90)

            if not entries:
                print("  (no qualifying whales in this window)")
                continue

            for idx, entry in enumerate(entries, 1):
                bias = entry.get('direction_bias')
                if bias is None:
                    bias_text = "No bias"
                elif bias > 0.55:
                    bias_text = f"UP {bias*100:.0f}%"
                elif bias < 0.45:
                    bias_text = f"DOWN {(1-bias)*100:.0f}%"
                else:
                    bias_text = "Mixed"

                market_summary = [
                    f"{mt}: {stats['trades']}t/${stats['volume']:,.0f}"
                    for mt, stats in list(entry['market_types'].items())[:3]
                ]

                print(f"{idx:2d}. {entry['name'][:18]:18} | ${entry['total_volume']:>10,.0f} vol | "
                      f"{entry['trade_count']:>3} trades | avg ${entry['avg_size']:>8,.0f} | {bias_text}")
                print(f"    Wallet: {entry['wallet']}")
                print(
                    f"    PnL: 30d ${entry['pnl_30d']:,.0f} | All ${entry['pnl_all']:,.0f} "
                    f"| Focus {entry['crypto_focus_ratio']*100:.0f}%"
                )
                print(
                    f"    Pace: {entry['trade_rate']:.2f} trades/min | ${entry['volume_velocity']:,.0f} vol/min"
                )
                if entry.get('top_symbol'):
                    share = entry['top_symbol_share'] * 100 if entry['top_symbol_share'] is not None else 0
                    print(
                        f"    Concentration: {entry['top_symbol']} {share:.0f}% | Dominant type: {entry.get('dominant_market_type') or '?'}"
                    )
                if entry.get('rank_30d_profit'):
                    print(f"    Leaderboard: 30d PnL #{entry['rank_30d_profit']} | All #{entry.get('rank_all_profit') or '?'}")
                print(f"    Markets: {', '.join(entry['symbols'][:5])}")
                if market_summary:
                    print(f"    Types: {' | '.join(market_summary)}")
                if entry.get('last_trade'):
                    print(f"    Last trade: {entry['last_trade']}")

    def rank_copytrade_candidates(
        self,
        min_pnl_30d: float = 30000,
        min_focus_ratio: float = 0.6,
        min_trades: int = 50,
        lookback_days: int = 7,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """Surface whales worth copy-trading based on PnL, focus, and recent velocity."""

        cutoff = (datetime.now(UTC) - timedelta(days=lookback_days)).isoformat()
        cursor = self.conn.cursor()

        cursor.execute(
            '''
            SELECT
                wp.wallet,
                wp.name,
                wp.pseudonym,
                wp.pnl_30d,
                wp.pnl_all,
                wp.crypto_focus_ratio,
                wp.crypto_trade_count,
                wp.is_crypto_specialist,
                wp.rank_30d_profit,
                wp.rank_all_profit,
                COUNT(ct.id) as trades_lookback,
                SUM(ct.usdc_value) as volume_lookback,
                SUM(CASE WHEN ct.direction = 'UP' THEN ct.usdc_value ELSE 0 END) as up_volume,
                SUM(CASE WHEN ct.direction = 'DOWN' THEN ct.usdc_value ELSE 0 END) as down_volume,
                COUNT(DISTINCT ct.symbol) as symbol_count,
                MAX(ct.timestamp) as last_trade
            FROM whale_profiles wp
            LEFT JOIN crypto_trades ct
                ON ct.whale_wallet = wp.wallet AND ct.timestamp >= ?
            WHERE wp.pnl_30d >= ?
              AND wp.crypto_focus_ratio >= ?
              AND wp.crypto_trade_count >= ?
            GROUP BY wp.wallet
            ORDER BY wp.pnl_30d DESC, volume_lookback DESC
            LIMIT ?
            ''',
            (cutoff, min_pnl_30d, min_focus_ratio, min_trades, limit),
        )

        candidates: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            total_dir_vol = (row['up_volume'] or 0) + (row['down_volume'] or 0)
            direction_bias = (row['up_volume'] or 0) / total_dir_vol if total_dir_vol else None

            candidates.append(
                {
                    'wallet': row['wallet'],
                    'name': row['name'] or row['pseudonym'] or row['wallet'][:12],
                    'pnl_30d': row['pnl_30d'] or 0,
                    'pnl_all': row['pnl_all'] or 0,
                    'focus': row['crypto_focus_ratio'] or 0,
                    'total_trades': row['crypto_trade_count'] or 0,
                    'is_specialist': bool(row['is_crypto_specialist']),
                    'rank_30d_profit': row['rank_30d_profit'],
                    'rank_all_profit': row['rank_all_profit'],
                    'trades_lookback': row['trades_lookback'] or 0,
                    'volume_lookback': row['volume_lookback'] or 0,
                    'direction_bias': direction_bias,
                    'symbol_count': row['symbol_count'] or 0,
                    'last_trade': row['last_trade'],
                }
            )

        return candidates

    def get_wallet_deep_dive(
        self,
        wallet: str,
        lookback_hours: int = 24,
        limit: int = 80,
    ) -> Optional[Dict[str, Any]]:
        """Return a deep-dive report for a single whale."""

        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM whale_profiles WHERE wallet = ?
            ''',
            (wallet,),
        )
        profile_row = cursor.fetchone()
        if not profile_row:
            return None

        cutoff = (datetime.now(UTC) - timedelta(hours=lookback_hours)).isoformat()
        cursor.execute(
            '''
            SELECT *
            FROM crypto_trades
            WHERE whale_wallet = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''',
            (wallet, cutoff, limit),
        )

        trades = cursor.fetchall()
        symbol_breakdown: Dict[str, Dict[str, float]] = defaultdict(lambda: {'trades': 0, 'volume': 0.0})
        market_breakdown: Dict[str, Dict[str, float]] = defaultdict(lambda: {'trades': 0, 'volume': 0.0})
        up_volume = down_volume = total_volume = 0.0

        for trade in trades:
            volume = trade['usdc_value'] or 0.0
            total_volume += volume
            symbol_stats = symbol_breakdown[trade['symbol']]
            symbol_stats['trades'] += 1
            symbol_stats['volume'] += volume

            mt_stats = market_breakdown[trade['market_type'] or 'UNKNOWN']
            mt_stats['trades'] += 1
            mt_stats['volume'] += volume

            if trade['direction'] == 'UP':
                up_volume += volume
            elif trade['direction'] == 'DOWN':
                down_volume += volume

        total_dir = up_volume + down_volume
        direction_bias = (up_volume / total_dir) if total_dir else None

        summary = {
            'total_trades': len(trades),
            'total_volume': total_volume,
            'up_volume': up_volume,
            'down_volume': down_volume,
            'net_flow': up_volume - down_volume,
            'direction_bias': direction_bias,
            'symbol_breakdown': {
                sym: stats
                for sym, stats in sorted(symbol_breakdown.items(), key=lambda x: -x[1]['volume'])
            },
            'market_breakdown': {
                mt: stats
                for mt, stats in sorted(market_breakdown.items(), key=lambda x: -x[1]['volume'])
            },
        }

        profile = dict(profile_row)
        return {
            'profile': profile,
            'summary': summary,
            'recent_trades': [dict(t) for t in trades],
        }

    def print_copytrade_candidates(self, candidates: List[Dict[str, Any]]):
        if not candidates:
            print("\n⚠️ No copy-trade candidates met the thresholds.")
            return

        print("\n" + "=" * 90)
        print("🤝 COPYTRADE CANDIDATES (risk-adjusted short list)")
        print("=" * 90)
        for idx, c in enumerate(candidates, 1):
            bias = c['direction_bias']
            if bias is None:
                bias_text = "—"
            elif bias > 0.55:
                bias_text = f"UP {bias*100:.0f}%"
            elif bias < 0.45:
                bias_text = f"DOWN {(1-bias)*100:.0f}%"
            else:
                bias_text = "Mixed"

            print(
                f"{idx:2d}. {c['name'][:18]:18} | 30d ${c['pnl_30d']:>10,.0f} | "
                f"{c['trades_lookback']:>3} trades / ${c['volume_lookback']:>9,.0f} | Focus {c['focus']*100:>3.0f}%"
            )
            print(
                f"    Rank: 30d #{c.get('rank_30d_profit') or '?'} | All #{c.get('rank_all_profit') or '?'} | "
                f"Symbols: {c['symbol_count']} | Bias: {bias_text}"
            )
            if c.get('last_trade'):
                print(f"    Last trade: {c['last_trade']}")

    def print_wallet_deep_dive(self, report: Dict[str, Any]):
        if not report:
            print("\n⚠️ Wallet not found.")
            return

        profile = report['profile']
        summary = report['summary']

        print("\n" + "=" * 90)
        print(f"🔍 WALLET DEEP DIVE - {profile.get('name') or profile.get('wallet')}")
        print("=" * 90)
        print(
            f"PnL 30d: ${profile.get('pnl_30d') or 0:,.0f} | All: ${profile.get('pnl_all') or 0:,.0f} | "
            f"Focus: {(profile.get('crypto_focus_ratio') or 0)*100:.0f}% | Trades: {profile.get('crypto_trade_count') or 0}"
        )
        bias = summary.get('direction_bias')
        if bias is None:
            bias_text = "—"
        elif bias > 0.55:
            bias_text = f"UP {bias*100:.0f}%"
        elif bias < 0.45:
            bias_text = f"DOWN {(1-bias)*100:.0f}%"
        else:
            bias_text = "Mixed"
        print(
            f"Recent: {summary['total_trades']} trades | ${summary['total_volume']:,.0f} vol | Net ${summary['net_flow']:,.0f} | Bias: {bias_text}"
        )

        if summary['symbol_breakdown']:
            top_sym, top_stats = next(iter(summary['symbol_breakdown'].items()))
            share = (top_stats['volume'] / summary['total_volume']) if summary['total_volume'] else 0
            print(
                f"Top symbol: {top_sym} {share*100:.0f}% | Markets: {', '.join(list(summary['market_breakdown'].keys())[:3])}"
            )

        print("\nLast trades:")
        for t in report['recent_trades'][:10]:
            ts = str(t['timestamp'])
            emoji = '🟢' if t['direction'] == 'UP' else ('🔴' if t['direction'] == 'DOWN' else '⚪')
            print(
                f"  {ts} | {emoji} {t['symbol']:4} {t['side']:4} ${t['usdc_value']:>8,.0f} | {t['market_type'] or '?'} | {t['market_title'][:40]}"
            )
    
    def get_15m_whale_signals(self, minutes_back: int = 15) -> Dict[str, Any]:
        """
        Get LIVE signals from 15M specialist whales only.
        These are the whales trading the exact same markets as the user.
        """
        cursor = self.conn.cursor()
        
        cutoff = (datetime.now(UTC) - timedelta(minutes=minutes_back)).isoformat()
        
        # Get recent 15M trades from PROFITABLE whales only
        cursor.execute('''
            SELECT 
                ct.symbol,
                ct.direction,
                ct.side,
                ct.usdc_value,
                ct.timestamp,
                wp.name,
                wp.pseudonym,
                wp.wallet,
                wp.pnl_30d,
                wp.pnl_all
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            WHERE ct.timestamp >= ?
              AND ct.market_type = '15MIN'
              AND wp.pnl_30d > 0
            ORDER BY ct.timestamp DESC
        ''', (cutoff,))
        
        trades = cursor.fetchall()
        
        # Aggregate by symbol
        signals = {}
        for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
            signals[symbol] = {
                'up_volume': 0, 'down_volume': 0,
                'up_whales': [], 'down_whales': [],
                'up_pnl': 0, 'down_pnl': 0,  # Total PnL of whales
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'strength': 0
            }
        
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in signals:
                continue
            
            direction = trade['direction']
            name = trade['name'] or trade['pseudonym'] or trade['wallet'][:8]
            volume = trade['usdc_value'] or 0
            pnl = trade['pnl_30d'] or 0
            
            whale_info = {'name': name, 'volume': volume, 'pnl': pnl}
            
            if direction == 'UP':
                signals[symbol]['up_volume'] += volume
                signals[symbol]['up_whales'].append(whale_info)
                signals[symbol]['up_pnl'] += pnl
            elif direction == 'DOWN':
                signals[symbol]['down_volume'] += volume
                signals[symbol]['down_whales'].append(whale_info)
                signals[symbol]['down_pnl'] += pnl
        
        # Calculate signals
        for symbol, data in signals.items():
            total_vol = data['up_volume'] + data['down_volume']
            if total_vol == 0:
                continue
            
            up_ratio = data['up_volume'] / total_vol
            
            # Also weight by whale quality (PnL)
            total_pnl = data['up_pnl'] + data['down_pnl']
            if total_pnl > 0:
                pnl_up_ratio = data['up_pnl'] / total_pnl
            else:
                pnl_up_ratio = 0.5
            
            # Combined score (70% volume, 30% whale quality)
            combined = up_ratio * 0.7 + pnl_up_ratio * 0.3
            
            if combined > 0.65:
                data['signal'] = 'UP'
                data['confidence'] = combined
            elif combined < 0.35:
                data['signal'] = 'DOWN'
                data['confidence'] = 1 - combined
            else:
                data['signal'] = 'NEUTRAL'
                data['confidence'] = 0.5
            
            # Signal strength (0-100)
            whale_count = len(data['up_whales']) + len(data['down_whales'])
            volume_score = min(total_vol / 10000, 1.0) * 30
            whale_score = min(whale_count / 5, 1.0) * 30
            confidence_score = abs(combined - 0.5) * 80
            data['strength'] = int(volume_score + whale_score + confidence_score)
        
        return signals
    
    def get_whale_boost(self, symbol: str, direction: str, minutes_back: int = 15) -> Dict[str, Any]:
        """
        Get whale signal boost for a specific trade.
        Returns a boost multiplier and confidence adjustment.
        
        This is the main function to integrate with smart_signal_trader.py
        """
        signals = self.get_15m_whale_signals(minutes_back)
        
        if symbol not in signals:
            return {'boost': 1.0, 'confidence_add': 0, 'reason': 'No whale data', 'agree': None}
        
        whale_signal = signals[symbol]
        
        # Check if whales agree with our direction
        if whale_signal['signal'] == direction:
            # WHALES AGREE - boost the signal!
            whale_strength = whale_signal['strength']
            
            if whale_strength >= 70:
                boost = 1.5  # Strong whale agreement - 50% size boost
                conf_add = 0.08
                reason = f"🐋 STRONG whale signal! {len(whale_signal['up_whales' if direction == 'UP' else 'down_whales'])} whales, ${whale_signal['up_volume' if direction == 'UP' else 'down_volume']:,.0f}"
            elif whale_strength >= 50:
                boost = 1.25  # Moderate agreement - 25% boost
                conf_add = 0.05
                reason = f"🐋 Whale agreement ({whale_strength}% strength)"
            else:
                boost = 1.1  # Weak agreement - 10% boost
                conf_add = 0.02
                reason = f"🐋 Mild whale support"
            
            return {'boost': boost, 'confidence_add': conf_add, 'reason': reason, 'agree': True}
        
        elif whale_signal['signal'] != 'NEUTRAL' and whale_signal['signal'] != direction:
            # WHALES DISAGREE - reduce confidence!
            whale_strength = whale_signal['strength']
            
            if whale_strength >= 70:
                boost = 0.5  # Strong disagreement - cut size in half
                conf_add = -0.10
                reason = f"⚠️ WHALE WARNING: {whale_signal['signal']} signal (strength {whale_strength})"
            elif whale_strength >= 50:
                boost = 0.75  # Moderate disagreement
                conf_add = -0.05
                reason = f"⚠️ Whales betting {whale_signal['signal']}"
            else:
                boost = 0.9  # Weak disagreement
                conf_add = -0.02
                reason = f"⚠️ Some whale disagreement"
            
            return {'boost': boost, 'confidence_add': conf_add, 'reason': reason, 'agree': False}
        
        # Neutral - no strong whale signal
        return {'boost': 1.0, 'confidence_add': 0, 'reason': 'No clear whale signal', 'agree': None}
    
    def analyze_whale_patterns(self, hours_back: int = 24) -> Dict:
        """Analyze trading patterns from whale activity."""
        cursor = self.conn.cursor()
        
        cutoff = (datetime.now(UTC) - timedelta(hours=hours_back)).isoformat()
        
        # Get recent crypto trades
        cursor.execute('''
            SELECT 
                ct.symbol, ct.direction, ct.side, ct.size, ct.price,
                ct.usdc_value, ct.market_type, ct.timestamp,
                wp.pnl_30d, wp.crypto_trade_count, wp.is_crypto_specialist,
                wp.name, wp.pseudonym, wp.wallet
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            WHERE ct.timestamp >= ?
            ORDER BY ct.timestamp DESC
        ''', (cutoff,))
        
        trades = cursor.fetchall()
        
        # Aggregate by symbol and direction
        signals = defaultdict(lambda: {
            'trades': 0,
            'whales': set(),
            'specialists': set(),
            'total_size': 0,
            'total_usdc': 0,
            'avg_whale_pnl': [],
            'market_types': defaultdict(int),
            'hourly': defaultdict(float)
        })
        
        for trade in trades:
            symbol = trade['symbol']
            direction = trade['direction']
            
            if direction == 'UNKNOWN':
                continue
            
            key = (symbol, direction)
            signals[key]['trades'] += 1
            signals[key]['whales'].add(trade['wallet'])
            signals[key]['total_size'] += trade['size']
            signals[key]['total_usdc'] += trade['usdc_value'] or (trade['size'] * trade['price'])
            signals[key]['avg_whale_pnl'].append(trade['pnl_30d'] or 0)
            signals[key]['market_types'][trade['market_type']] += 1
            
            if trade['is_crypto_specialist']:
                signals[key]['specialists'].add(trade['wallet'])
            
            # Parse hour from timestamp
            try:
                ts = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                signals[key]['hourly'][ts.hour] += trade['usdc_value'] or 0
            except:
                pass
        
        # Calculate signal strength
        analysis = {}
        for (symbol, direction), data in signals.items():
            whale_count = len(data['whales'])
            specialist_count = len(data['specialists'])
            avg_pnl = statistics.mean(data['avg_whale_pnl']) if data['avg_whale_pnl'] else 0
            
            # Signal strength based on multiple factors
            strength = 0.0
            strength += min(whale_count / 5, 0.3)  # Up to 0.3 for whale count
            strength += min(specialist_count / 3, 0.3)  # Up to 0.3 for specialists
            strength += min(data['total_usdc'] / 50000, 0.2)  # Up to 0.2 for volume
            strength += min(max(avg_pnl, 0) / 500000, 0.2)  # Up to 0.2 for whale quality
            
            analysis[(symbol, direction)] = {
                'symbol': symbol,
                'direction': direction,
                'signal_strength': round(strength, 3),
                'whale_count': whale_count,
                'specialist_count': specialist_count,
                'total_size': data['total_size'],
                'total_usdc': data['total_usdc'],
                'avg_whale_pnl': avg_pnl,
                'trade_count': data['trades'],
                'dominant_market_type': max(data['market_types'].items(), key=lambda x: x[1])[0] if data['market_types'] else None,
                'whales': list(data['whales'])
            }
        
        return analysis
    
    def generate_trading_signals(self, hours_back: int = 4, min_strength: float = 0.3) -> List[Dict]:
        """Generate actionable trading signals from whale activity."""
        analysis = self.analyze_whale_patterns(hours_back)
        
        signals = []
        for key, data in analysis.items():
            if data['signal_strength'] >= min_strength:
                signals.append(data)
        
        # Sort by signal strength
        signals.sort(key=lambda x: -x['signal_strength'])
        
        # Save to database
        cursor = self.conn.cursor()
        for signal in signals:
            cursor.execute('''
                INSERT INTO whale_signals (
                    symbol, direction, signal_strength, whale_count,
                    specialist_count, total_size, total_usdc, avg_whale_pnl,
                    participating_whales, market_type, window_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'], signal['direction'], signal['signal_strength'],
                signal['whale_count'], signal['specialist_count'],
                signal['total_size'], signal['total_usdc'], signal['avg_whale_pnl'],
                json.dumps(signal['whales'][:10]), signal['dominant_market_type'],
                hours_back * 60
            ))
        self.conn.commit()
        
        return signals
    
    def get_live_signal(self, symbol: str, minutes_back: int = 30) -> Optional[Dict]:
        """Get current whale signal for a specific symbol."""
        cursor = self.conn.cursor()
        
        cutoff = (datetime.now(UTC) - timedelta(minutes=minutes_back)).isoformat()
        
        cursor.execute('''
            SELECT 
                direction, side, size, price, usdc_value,
                wp.pnl_30d, wp.is_crypto_specialist, wp.name, wp.pseudonym
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            WHERE ct.symbol = ? AND ct.timestamp >= ?
            ORDER BY ct.timestamp DESC
        ''', (symbol, cutoff))
        
        trades = cursor.fetchall()
        
        if not trades:
            return None
        
        # Aggregate
        up_volume = 0
        down_volume = 0
        up_whales = set()
        down_whales = set()
        up_specialists = 0
        down_specialists = 0
        
        for trade in trades:
            direction = trade['direction']
            usdc = trade['usdc_value'] or (trade['size'] * trade['price'])
            name = trade['name'] or trade['pseudonym'] or 'Unknown'
            
            if direction == 'UP':
                up_volume += usdc
                up_whales.add(name)
                if trade['is_crypto_specialist']:
                    up_specialists += 1
            elif direction == 'DOWN':
                down_volume += usdc
                down_whales.add(name)
                if trade['is_crypto_specialist']:
                    down_specialists += 1
        
        total_volume = up_volume + down_volume
        if total_volume == 0:
            return None
        
        # Determine signal
        if up_volume > down_volume * 1.5:
            direction = 'UP'
            confidence = up_volume / total_volume
            whale_names = list(up_whales)
            specialists = up_specialists
        elif down_volume > up_volume * 1.5:
            direction = 'DOWN'
            confidence = down_volume / total_volume
            whale_names = list(down_whales)
            specialists = down_specialists
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
            whale_names = list(up_whales | down_whales)
            specialists = up_specialists + down_specialists
        
        return {
            'symbol': symbol,
            'direction': direction,
            'confidence': round(confidence, 3),
            'up_volume': up_volume,
            'down_volume': down_volume,
            'up_whales': len(up_whales),
            'down_whales': len(down_whales),
            'specialists': specialists,
            'whale_names': whale_names[:5],
            'minutes_back': minutes_back
        }
    
    def print_dashboard(self):
        """Print a comprehensive whale intelligence dashboard."""
        print("\n" + "="*70)
        print("🐋 WHALE INTELLIGENCE DASHBOARD")
        print("="*70)
        
        cursor = self.conn.cursor()
        
        # Crypto specialists
        cursor.execute('''
            SELECT name, pseudonym, wallet, pnl_30d, crypto_trade_count,
                   crypto_focus_ratio, favorite_crypto, favorite_market_type
            FROM whale_profiles
            WHERE is_crypto_specialist = 1
            ORDER BY pnl_30d DESC
            LIMIT 10
        ''')
        
        print("\n📊 TOP CRYPTO SPECIALISTS (30d PnL)")
        print("-" * 70)
        for row in cursor.fetchall():
            name = row['name'] or row['pseudonym'] or row['wallet'][:12]
            print(f"  {name[:20]:20} | ${row['pnl_30d']:>12,.0f} | "
                  f"{row['crypto_trade_count']:>4} trades | "
                  f"{row['favorite_crypto'] or '?':4} | {row['favorite_market_type'] or '?'}")
        
        # Recent signals
        signals = self.generate_trading_signals(hours_back=4)
        
        print("\n🎯 CURRENT WHALE SIGNALS (last 4h)")
        print("-" * 70)
        for signal in signals[:8]:
            emoji = '🟢' if signal['direction'] == 'UP' else '🔴'
            print(f"  {emoji} {signal['symbol']:4} {signal['direction']:4} | "
                  f"Strength: {signal['signal_strength']:.2f} | "
                  f"Whales: {signal['whale_count']} ({signal['specialist_count']} specialists) | "
                  f"${signal['total_usdc']:,.0f}")
        
        # Live signals for each crypto
        print("\n⚡ LIVE SIGNALS (last 30 min)")
        print("-" * 70)
        for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
            signal = self.get_live_signal(symbol, 30)
            if signal and signal['direction'] != 'NEUTRAL':
                emoji = '🟢' if signal['direction'] == 'UP' else '🔴'
                print(f"  {emoji} {symbol}: {signal['direction']} "
                      f"({signal['confidence']:.0%} confidence) | "
                      f"UP: ${signal['up_volume']:,.0f} ({signal['up_whales']} whales) | "
                      f"DOWN: ${signal['down_volume']:,.0f} ({signal['down_whales']} whales)")
            else:
                print(f"  ⚪ {symbol}: No clear signal")
        
        # Recent notable trades
        cursor.execute('''
            SELECT ct.timestamp, ct.symbol, ct.direction, ct.side, ct.size, 
                   ct.price, ct.usdc_value, ct.market_title,
                   wp.name, wp.pseudonym, wp.pnl_30d
            FROM crypto_trades ct
            JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
            WHERE ct.usdc_value > 1000
            ORDER BY ct.timestamp DESC
            LIMIT 15
        ''')
        
        print("\n💰 RECENT BIG CRYPTO TRADES (>$1k)")
        print("-" * 70)
        for row in cursor.fetchall():
            name = row['name'] or row['pseudonym'] or 'Unknown'
            ts = str(row['timestamp'])[11:16] if row['timestamp'] else '??:??'
            direction_emoji = '🟢' if row['direction'] == 'UP' else ('🔴' if row['direction'] == 'DOWN' else '⚪')
            print(f"  {ts} | {direction_emoji} {row['symbol']:4} {row['side']:4} "
                  f"${row['usdc_value']:>8,.0f} | {name[:15]:15} (${row['pnl_30d']:,.0f} PnL)")


def main():
    parser = argparse.ArgumentParser(description='Whale Intelligence System')
    parser.add_argument('--init', action='store_true', help='Initialize database')
    parser.add_argument('--db-path', type=Path, default=DB_PATH, help='Override database path')
    parser.add_argument('--scrape-leaderboards', action='store_true', help='Scrape all leaderboards')
    parser.add_argument('--scrape-activity', action='store_true', help='Scrape whale crypto activity')
    parser.add_argument('--hours', type=int, default=48, help='Hours of activity to scrape')
    parser.add_argument('--limit', type=int, default=100, help='Leaderboard limit per category')
    parser.add_argument('--specialists', action='store_true', help='Show crypto specialists')
    parser.add_argument('--signals', action='store_true', help='Generate trading signals')
    parser.add_argument('--dashboard', action='store_true', help='Show full dashboard')
    parser.add_argument('--live', type=str, help='Get live signal for a symbol (BTC, ETH, etc)')
    parser.add_argument('--full-scan', action='store_true', help='Run complete scan (leaderboards + activity)')
    parser.add_argument('--elite', action='store_true', help='Deep analysis of elite high win-rate whales')
    parser.add_argument('--15m', dest='m15', action='store_true', help='Show 15-minute market specialists')
    parser.add_argument('--monitor', action='store_true', help='Continuous monitoring mode (refresh every 5 min)')
    parser.add_argument('--copy', action='store_true', help='Copy trade recommendations from elite whales')
    parser.add_argument('--intraday', action='store_true', help='Show 15m/1h/4h active whale dashboards')
    parser.add_argument('--symbol-filter', type=str, help='Limit intraday dashboards to a specific symbol (e.g., BTC)')
    parser.add_argument('--copytrade', action='store_true', help='Show curated copy-trade candidates')
    parser.add_argument('--deep-dive', dest='deep_dive', type=str, help='Deep dive for a wallet address')
    parser.add_argument('--top', type=int, default=15, help='Number of wallets to display per dashboard')
    
    args = parser.parse_args()

    db_path: Path = Path(args.db_path)

    # Initialize database if needed
    if not db_path.exists() or args.init:
        init_database(db_path)

    with WhaleIntelligence(db_path=db_path) as wi:
        if args.scrape_leaderboards or args.full_scan:
            wi.scrape_all_leaderboards(args.limit)
        
        if args.scrape_activity or args.full_scan:
            wi.scrape_all_whale_activity(args.hours)
        
        if args.specialists:
            specialists = wi.get_crypto_specialists()
            print("\n🐋 CRYPTO SPECIALISTS")
            print("="*60)
            for s in specialists[:20]:
                name = s['name'] or s['pseudonym'] or s['wallet'][:12]
                print(f"  {name[:20]:20} | ${s['pnl_30d']:>10,.0f} | "
                      f"{s['crypto_trade_count']:>4} trades | {s['favorite_crypto'] or '?':4}")
        
        if args.signals:
            signals = wi.generate_trading_signals()
            print("\n🎯 WHALE SIGNALS")
            print("="*60)
            for s in signals:
                emoji = '🟢' if s['direction'] == 'UP' else '🔴'
                print(f"  {emoji} {s['symbol']:4} {s['direction']:4} | "
                      f"Strength: {s['signal_strength']:.2f} | "
                      f"Whales: {s['whale_count']} | ${s['total_usdc']:,.0f}")
        
        if args.live:
            signal = wi.get_live_signal(args.live.upper(), 30)
            if signal:
                print(f"\n⚡ LIVE SIGNAL for {args.live.upper()}")
                print(f"  Direction: {signal['direction']}")
                print(f"  Confidence: {signal['confidence']:.0%}")
                print(f"  UP Volume: ${signal['up_volume']:,.0f} ({signal['up_whales']} whales)")
                print(f"  DOWN Volume: ${signal['down_volume']:,.0f} ({signal['down_whales']} whales)")
                print(f"  Top Whales: {', '.join(signal['whale_names'])}")
            else:
                print(f"No recent activity for {args.live.upper()}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # ELITE WHALE ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════
        if args.elite:
            print("\n" + "="*80)
            print("🏆 ELITE WHALE ANALYSIS - Finding the Best of the Best")
            print("="*80)
            
            elite = wi.analyze_elite_whales()
            
            # 15M Specialists
            print("\n⚡ TOP 15-MINUTE MARKET SPECIALISTS")
            print("-"*80)
            print(f"{'Name':<20} | {'30d PnL':>12} | {'15M Trades':>10} | {'15M %':>6} | {'Avg Size':>10} | {'$/Trade':>10}")
            print("-"*80)
            for whale in elite['elite_15m_whales'][:15]:
                name = whale['name'][:20]
                pnl = whale['pnl_30d']
                trades = whale['15m_trades']
                ratio = whale['15m_ratio'] * 100
                avg_size = whale['15m_avg_size']
                pnl_per = whale['pnl_per_trade']
                elite_tag = "🌟" if whale['is_elite'] else ""
                print(f"  {elite_tag}{name:<18} | ${pnl:>11,.0f} | {trades:>10} | {ratio:>5.0f}% | ${avg_size:>9,.0f} | ${pnl_per:>9,.0f}")
            
            # Big Money Whales
            print("\n💰 BIG MONEY CRYPTO WHALES (>$100k PnL)")
            print("-"*80)
            print(f"{'Name':<20} | {'30d PnL':>12} | {'Total Trades':>12} | {'Volume':>14} | {'Symbols'}")
            print("-"*80)
            for whale in elite['big_money_whales'][:10]:
                name = whale['name'][:20]
                pnl = whale['pnl_30d']
                trades = whale['total_trades']
                volume = whale['total_volume']
                symbols = ', '.join(whale['symbols'][:4])
                print(f"  {name:<20} | ${pnl:>11,.0f} | {trades:>12} | ${volume:>13,.0f} | {symbols}")
            
            # Hourly Patterns
            print("\n⏰ ELITE WHALE HOURLY PATTERNS (15M Markets Only)")
            print("-"*80)
            print(f"{'Hour (UTC)':<12} | {'Trades':>8} | {'Volume':>12} | {'UP':>6} | {'DOWN':>6} | {'Bias'}")
            print("-"*80)
            for hour in sorted(elite['hourly_patterns'].keys()):
                data = elite['hourly_patterns'][hour]
                up = data['directions'].get('UP', 0)
                down = data['directions'].get('DOWN', 0)
                total = up + down
                if total > 0:
                    bias_ratio = up / total
                    if bias_ratio > 0.6:
                        bias = "🟢 BULLISH"
                    elif bias_ratio < 0.4:
                        bias = "🔴 BEARISH"
                    else:
                        bias = "⚪ NEUTRAL"
                else:
                    bias = "—"
                print(f"  {hour:02d}:00 UTC   | {data['total_trades']:>8} | ${data['total_volume']:>11,.0f} | {up:>6} | {down:>6} | {bias}")
            
            # Market Type Stats
            print("\n📊 MARKET TYPE BREAKDOWN")
            print("-"*60)
            for mtype, stats in sorted(elite['market_type_stats'].items(), key=lambda x: -x[1]['trades']):
                print(f"  {mtype:<15} | {stats['trades']:>6} trades | ${stats['volume']:>12,.0f} | {stats['whales']} whales | Avg PnL: ${stats['avg_whale_pnl']:>10,.0f}")

        if args.intraday:
            dashboards = wi.get_intraday_wallet_dashboards(
                top_n=args.top, symbol_filter=args.symbol_filter.upper() if args.symbol_filter else None
            )
            wi.print_intraday_dashboards(dashboards, top_n=args.top)

        if args.copytrade:
            candidates = wi.rank_copytrade_candidates()
            wi.print_copytrade_candidates(candidates)

        # ═══════════════════════════════════════════════════════════════════════
        # 15M LIVE SIGNALS
        # ═══════════════════════════════════════════════════════════════════════
        if args.m15:
            print("\n" + "="*70)
            print("⚡ 15-MINUTE WHALE SIGNALS (From Elite 15M Specialists)")
            print("="*70)
            
            signals = wi.get_15m_whale_signals(minutes_back=15)
            
            for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
                sig = signals[symbol]
                if sig['signal'] == 'NEUTRAL':
                    emoji = '⚪'
                elif sig['signal'] == 'UP':
                    emoji = '🟢'
                else:
                    emoji = '🔴'
                
                print(f"\n  {emoji} {symbol}")
                print(f"     Signal: {sig['signal']} (Strength: {sig['strength']}/100)")
                print(f"     Confidence: {sig['confidence']:.0%}")
                print(f"     UP:   ${sig['up_volume']:>10,.0f} | {len(sig['up_whales']):>2} whales | PnL sum: ${sig['up_pnl']:>12,.0f}")
                print(f"     DOWN: ${sig['down_volume']:>10,.0f} | {len(sig['down_whales']):>2} whales | PnL sum: ${sig['down_pnl']:>12,.0f}")
                
                if sig['up_whales']:
                    top_up = sorted(sig['up_whales'], key=lambda x: -x['pnl'])[:3]
                    print(f"     Top UP whales: {', '.join([w['name'] for w in top_up])}")
                if sig['down_whales']:
                    top_down = sorted(sig['down_whales'], key=lambda x: -x['pnl'])[:3]
                    print(f"     Top DOWN whales: {', '.join([w['name'] for w in top_down])}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # CONTINUOUS MONITORING
        # ═══════════════════════════════════════════════════════════════════════
        if args.monitor:
            print("\n" + "="*70)
            print("🔄 CONTINUOUS WHALE MONITORING (Ctrl+C to stop)")
            print("="*70)
            
            import time as time_module
            refresh_interval = 300  # 5 minutes
            
            try:
                while True:
                    # Refresh whale activity (last 30 min only for speed)
                    wi.scrape_all_whale_activity(hours_back=0.5, min_rank=50)
                    
                    # Show current signals
                    signals = wi.get_15m_whale_signals(minutes_back=15)
                    
                    now = datetime.now(UTC).strftime("%H:%M:%S UTC")
                    print(f"\n⏰ {now} - 15M Whale Signals:")
                    
                    for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
                        sig = signals[symbol]
                        if sig['signal'] != 'NEUTRAL' and sig['strength'] >= 40:
                            emoji = '🟢' if sig['signal'] == 'UP' else '🔴'
                            whale_count = len(sig['up_whales']) + len(sig['down_whales'])
                            volume = sig['up_volume'] + sig['down_volume']
                            print(f"  {emoji} {symbol} {sig['signal']} | Strength: {sig['strength']} | ${volume:,.0f} | {whale_count} whales")
                    
                    print(f"\n  💤 Sleeping {refresh_interval}s...")
                    time_module.sleep(refresh_interval)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Monitoring stopped.")

        if args.deep_dive:
            report = wi.get_wallet_deep_dive(args.deep_dive)
            wi.print_wallet_deep_dive(report)

        if args.dashboard or args.full_scan:
            wi.print_dashboard()
        
        # ═══════════════════════════════════════════════════════════════════════
        # COPY TRADE - Quick view of what to bet
        # ═══════════════════════════════════════════════════════════════════════
        if args.copy:
            cursor = wi.conn.cursor()
            cutoff = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
            
            print("\n" + "="*70)
            print("🎯 COPY TRADE - Follow the Elite 15M Whales")
            print("="*70)
            
            # Get consensus for each symbol
            cursor.execute('''
                SELECT 
                    ct.symbol,
                    ct.direction,
                    SUM(ct.usdc_value) as total_vol,
                    COUNT(*) as trade_count,
                    COUNT(DISTINCT ct.whale_wallet) as whale_count,
                    SUM(wp.pnl_30d) as total_pnl
                FROM crypto_trades ct
                JOIN whale_profiles wp ON ct.whale_wallet = wp.wallet
                WHERE ct.timestamp >= ?
                  AND ct.market_type = '15MIN'
                  AND wp.pnl_30d > 10000
                GROUP BY ct.symbol, ct.direction
                ORDER BY ct.symbol, total_vol DESC
            ''', (cutoff,))
            
            results = cursor.fetchall()
            
            symbols = {}
            for r in results:
                sym = r['symbol']
                if sym not in symbols:
                    symbols[sym] = {}
                symbols[sym][r['direction']] = {
                    'vol': r['total_vol'],
                    'trades': r['trade_count'],
                    'whales': r['whale_count'],
                    'pnl': r['total_pnl']
                }
            
            print("\n📊 CURRENT WHALE BETS (30 min window, profitable whales only)")
            print("-"*70)
            
            recommendations = []
            
            for sym in ['BTC', 'ETH', 'SOL', 'XRP']:
                if sym not in symbols:
                    print(f"  {sym}: No recent whale activity")
                    continue
                
                up = symbols[sym].get('UP', {'vol': 0, 'trades': 0, 'whales': 0, 'pnl': 0})
                down = symbols[sym].get('DOWN', {'vol': 0, 'trades': 0, 'whales': 0, 'pnl': 0})
                
                total_vol = up['vol'] + down['vol']
                if total_vol == 0:
                    continue
                
                up_pct = up['vol'] / total_vol * 100
                
                if up_pct > 60:
                    signal = 'UP'
                    confidence = up_pct
                    strength = 'STRONG' if up_pct > 75 else 'MODERATE'
                    emoji = '🟢'
                elif up_pct < 40:
                    signal = 'DOWN'
                    confidence = 100 - up_pct
                    strength = 'STRONG' if up_pct < 25 else 'MODERATE'
                    emoji = '🔴'
                else:
                    signal = 'SKIP'
                    confidence = 50
                    strength = 'WEAK'
                    emoji = '⚪'
                
                print(f"\n  {emoji} {sym}: {signal}")
                print(f"     Confidence: {confidence:.0f}% | Strength: {strength}")
                print(f"     UP Volume:   ${up['vol']:>10,.0f} ({up['whales']} whales)")
                print(f"     DOWN Volume: ${down['vol']:>10,.0f} ({down['whales']} whales)")
                
                if signal != 'SKIP' and strength != 'WEAK':
                    recommendations.append({
                        'symbol': sym,
                        'direction': signal,
                        'confidence': confidence,
                        'strength': strength
                    })
            
            if recommendations:
                print("\n" + "="*70)
                print("💰 RECOMMENDED TRADES")
                print("="*70)
                for rec in sorted(recommendations, key=lambda x: -x['confidence']):
                    emoji = '🟢' if rec['direction'] == 'UP' else '🔴'
                    print(f"  {emoji} {rec['symbol']} {rec['direction']} - {rec['confidence']:.0f}% confidence ({rec['strength']})")
            else:
                print("\n  ⚠️ No strong signals right now - WAIT for better setup")
        
        if not any([
            args.scrape_leaderboards,
            args.scrape_activity,
            args.specialists,
            args.signals,
            args.dashboard,
            args.live,
            args.full_scan,
            args.init,
            args.elite,
            args.m15,
            args.monitor,
            args.copy,
            args.intraday,
            args.copytrade,
            args.deep_dive,
        ]):
            parser.print_help()
            print("\n💡 Quick start:")
            print("   python whale_intelligence.py --full-scan --hours 48  # Full scrape")
            print("   python whale_intelligence.py --elite                  # Elite whale analysis")
            print("   python whale_intelligence.py --15m                    # 15M specialist signals")
            print("   python whale_intelligence.py --copy                   # Copy trade recommendations")
            print("   python whale_intelligence.py --monitor                # Continuous monitoring")


if __name__ == '__main__':
    main()
