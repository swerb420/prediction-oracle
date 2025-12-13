"""
Whale Wallet Database and Tracking

Tracks:
- Known whale wallets with labels
- Historical win rates
- PnL tracking
- Trade history
- Wallet reputation scores
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WalletStats(BaseModel):
    """Stats for a tracked wallet."""
    
    address: str
    label: Optional[str] = None
    
    # Trading stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    pending_trades: int = 0
    
    # PnL
    total_pnl_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    
    # Volumes
    total_volume_usd: float = 0.0
    avg_trade_size_usd: float = 0.0
    largest_trade_usd: float = 0.0
    
    # Calculated
    win_rate: float = 0.0
    roi_pct: float = 0.0
    sharpe_ratio: Optional[float] = None
    
    # Metadata
    first_seen: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    categories: list[str] = []  # e.g., ["politics", "crypto"]
    
    # Reputation
    reputation_score: float = 0.5  # 0-1, higher = more reliable
    is_verified_sharp: bool = False
    is_arb_bot: bool = False
    
    def to_dict(self) -> dict:
        return self.model_dump()


class WalletTrade(BaseModel):
    """A single trade by a wallet."""
    
    id: Optional[int] = None
    wallet_address: str
    tx_hash: str
    
    # Trade details
    market_id: str
    market_question: str
    outcome: str  # "yes" or "no"
    direction: str  # "buy" or "sell"
    
    # Amounts
    amount_usd: float
    shares: float
    entry_price: float
    
    # Resolution
    exit_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    status: str = "open"  # "open", "won", "lost", "sold"
    
    # Timing
    timestamp: datetime
    resolved_at: Optional[datetime] = None


class WhaleDatabase:
    """
    SQLite database for tracking whale wallets and their performance.
    """
    
    def __init__(self, db_path: str = "whale_tracker.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
    
    async def connect(self):
        """Initialize database connection and tables."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        logger.info(f"WhaleDatabase connected: {self.db_path}")
    
    async def close(self):
        """Close database connection."""
        if self._db:
            await self._db.close()
    
    async def _create_tables(self):
        """Create database tables."""
        
        await self._db.executescript("""
            -- Wallet registry
            CREATE TABLE IF NOT EXISTS wallets (
                address TEXT PRIMARY KEY,
                label TEXT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_trade TIMESTAMP,
                categories TEXT,  -- JSON array
                is_verified_sharp BOOLEAN DEFAULT FALSE,
                is_arb_bot BOOLEAN DEFAULT FALSE,
                notes TEXT
            );
            
            -- Trade history
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                tx_hash TEXT UNIQUE NOT NULL,
                market_id TEXT NOT NULL,
                market_question TEXT,
                outcome TEXT NOT NULL,
                direction TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                shares REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl_usd REAL,
                status TEXT DEFAULT 'open',
                timestamp TIMESTAMP NOT NULL,
                resolved_at TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES wallets(address)
            );
            
            -- Wallet stats cache (updated periodically)
            CREATE TABLE IF NOT EXISTS wallet_stats (
                address TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl_usd REAL DEFAULT 0,
                total_volume_usd REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                roi_pct REAL DEFAULT 0,
                reputation_score REAL DEFAULT 0.5,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (address) REFERENCES wallets(address)
            );
            
            -- Market outcomes (for trade resolution)
            CREATE TABLE IF NOT EXISTS market_outcomes (
                market_id TEXT PRIMARY KEY,
                question TEXT,
                winning_outcome TEXT,  -- "yes" or "no"
                resolved_at TIMESTAMP
            );
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet_address);
            CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        """)
        
        await self._db.commit()
    
    # ========================================================================
    # WALLET MANAGEMENT
    # ========================================================================
    
    async def add_wallet(
        self,
        address: str,
        label: Optional[str] = None,
        categories: list[str] = None,
        is_arb_bot: bool = False,
    ) -> bool:
        """Add or update a wallet in the registry."""
        
        try:
            await self._db.execute("""
                INSERT INTO wallets (address, label, categories, is_arb_bot)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    label = COALESCE(excluded.label, label),
                    categories = COALESCE(excluded.categories, categories),
                    is_arb_bot = excluded.is_arb_bot
            """, (
                address.lower(),
                label,
                json.dumps(categories or []),
                is_arb_bot,
            ))
            await self._db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add wallet: {e}")
            return False
    
    async def get_wallet(self, address: str) -> Optional[WalletStats]:
        """Get wallet info and stats."""
        
        async with self._db.execute("""
            SELECT 
                w.address, w.label, w.first_seen, w.last_trade,
                w.categories, w.is_verified_sharp, w.is_arb_bot,
                COALESCE(s.total_trades, 0) as total_trades,
                COALESCE(s.winning_trades, 0) as winning_trades,
                COALESCE(s.losing_trades, 0) as losing_trades,
                COALESCE(s.total_pnl_usd, 0) as total_pnl_usd,
                COALESCE(s.total_volume_usd, 0) as total_volume_usd,
                COALESCE(s.win_rate, 0) as win_rate,
                COALESCE(s.roi_pct, 0) as roi_pct,
                COALESCE(s.reputation_score, 0.5) as reputation_score
            FROM wallets w
            LEFT JOIN wallet_stats s ON w.address = s.address
            WHERE w.address = ?
        """, (address.lower(),)) as cursor:
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            return WalletStats(
                address=row[0],
                label=row[1],
                first_seen=row[2],
                last_trade=row[3],
                categories=json.loads(row[4]) if row[4] else [],
                is_verified_sharp=bool(row[5]),
                is_arb_bot=bool(row[6]),
                total_trades=row[7],
                winning_trades=row[8],
                losing_trades=row[9],
                total_pnl_usd=row[10],
                total_volume_usd=row[11],
                win_rate=row[12],
                roi_pct=row[13],
                reputation_score=row[14],
            )
    
    async def get_all_wallets(
        self,
        min_trades: int = 0,
        min_win_rate: float = 0.0,
        exclude_arb_bots: bool = True,
    ) -> list[WalletStats]:
        """Get all tracked wallets with filtering."""
        
        query = """
            SELECT 
                w.address, w.label, w.first_seen, w.last_trade,
                w.categories, w.is_verified_sharp, w.is_arb_bot,
                COALESCE(s.total_trades, 0) as total_trades,
                COALESCE(s.winning_trades, 0) as winning_trades,
                COALESCE(s.losing_trades, 0) as losing_trades,
                COALESCE(s.total_pnl_usd, 0) as total_pnl_usd,
                COALESCE(s.total_volume_usd, 0) as total_volume_usd,
                COALESCE(s.win_rate, 0) as win_rate,
                COALESCE(s.roi_pct, 0) as roi_pct,
                COALESCE(s.reputation_score, 0.5) as reputation_score
            FROM wallets w
            LEFT JOIN wallet_stats s ON w.address = s.address
            WHERE COALESCE(s.total_trades, 0) >= ?
            AND COALESCE(s.win_rate, 0) >= ?
        """
        
        params = [min_trades, min_win_rate]
        
        if exclude_arb_bots:
            query += " AND w.is_arb_bot = FALSE"
        
        query += " ORDER BY s.reputation_score DESC, s.total_pnl_usd DESC"
        
        wallets = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                wallets.append(WalletStats(
                    address=row[0],
                    label=row[1],
                    first_seen=row[2],
                    last_trade=row[3],
                    categories=json.loads(row[4]) if row[4] else [],
                    is_verified_sharp=bool(row[5]),
                    is_arb_bot=bool(row[6]),
                    total_trades=row[7],
                    winning_trades=row[8],
                    losing_trades=row[9],
                    total_pnl_usd=row[10],
                    total_volume_usd=row[11],
                    win_rate=row[12],
                    roi_pct=row[13],
                    reputation_score=row[14],
                ))
        
        return wallets
    
    async def get_sharp_wallets(self, min_win_rate: float = 0.65) -> list[WalletStats]:
        """Get wallets with high win rates (the sharps)."""
        return await self.get_all_wallets(min_trades=10, min_win_rate=min_win_rate)
    
    # ========================================================================
    # TRADE TRACKING
    # ========================================================================
    
    async def record_trade(self, trade: WalletTrade) -> int:
        """Record a new trade."""
        
        # Ensure wallet exists
        await self.add_wallet(trade.wallet_address)
        
        try:
            cursor = await self._db.execute("""
                INSERT INTO trades (
                    wallet_address, tx_hash, market_id, market_question,
                    outcome, direction, amount_usd, shares, entry_price,
                    status, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.wallet_address.lower(),
                trade.tx_hash,
                trade.market_id,
                trade.market_question,
                trade.outcome,
                trade.direction,
                trade.amount_usd,
                trade.shares,
                trade.entry_price,
                trade.status,
                trade.timestamp.isoformat(),
            ))
            
            trade_id = cursor.lastrowid
            
            # Update wallet's last_trade
            await self._db.execute("""
                UPDATE wallets SET last_trade = ? WHERE address = ?
            """, (trade.timestamp.isoformat(), trade.wallet_address.lower()))
            
            await self._db.commit()
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return -1
    
    async def get_wallet_trades(
        self,
        address: str,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> list[WalletTrade]:
        """Get trades for a wallet."""
        
        query = """
            SELECT 
                id, wallet_address, tx_hash, market_id, market_question,
                outcome, direction, amount_usd, shares, entry_price,
                exit_price, pnl_usd, status, timestamp, resolved_at
            FROM trades
            WHERE wallet_address = ?
        """
        params = [address.lower()]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        trades = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                trades.append(WalletTrade(
                    id=row[0],
                    wallet_address=row[1],
                    tx_hash=row[2],
                    market_id=row[3],
                    market_question=row[4],
                    outcome=row[5],
                    direction=row[6],
                    amount_usd=row[7],
                    shares=row[8],
                    entry_price=row[9],
                    exit_price=row[10],
                    pnl_usd=row[11],
                    status=row[12],
                    timestamp=datetime.fromisoformat(row[13]),
                    resolved_at=datetime.fromisoformat(row[14]) if row[14] else None,
                ))
        
        return trades
    
    # ========================================================================
    # MARKET RESOLUTION
    # ========================================================================
    
    async def resolve_market(
        self,
        market_id: str,
        winning_outcome: str,  # "yes" or "no"
        question: Optional[str] = None,
    ):
        """
        Resolve a market and update all related trades.
        This is called when a market resolves to determine PnL.
        """
        
        now = datetime.now(timezone.utc)
        
        # Record the outcome
        await self._db.execute("""
            INSERT INTO market_outcomes (market_id, question, winning_outcome, resolved_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                winning_outcome = excluded.winning_outcome,
                resolved_at = excluded.resolved_at
        """, (market_id, question, winning_outcome, now.isoformat()))
        
        # Get all open trades for this market
        async with self._db.execute("""
            SELECT id, wallet_address, outcome, direction, amount_usd, shares, entry_price
            FROM trades
            WHERE market_id = ? AND status = 'open'
        """, (market_id,)) as cursor:
            
            trades = await cursor.fetchall()
        
        # Update each trade
        for trade in trades:
            trade_id, wallet, outcome, direction, amount_usd, shares, entry_price = trade
            
            # Calculate PnL
            # If they bought the winning outcome, they win
            # If they bought the losing outcome, they lose
            if direction == "buy":
                if outcome.lower() == winning_outcome.lower():
                    # Winner! Shares are worth $1 each
                    pnl = shares - amount_usd
                    status = "won"
                else:
                    # Loser - shares are worthless
                    pnl = -amount_usd
                    status = "lost"
            else:
                # Sell direction (less common)
                if outcome.lower() == winning_outcome.lower():
                    pnl = -shares + amount_usd  # They sold winning shares
                    status = "lost"
                else:
                    pnl = amount_usd  # They sold worthless shares
                    status = "won"
            
            # Update trade
            await self._db.execute("""
                UPDATE trades 
                SET status = ?, pnl_usd = ?, exit_price = ?, resolved_at = ?
                WHERE id = ?
            """, (
                status,
                pnl,
                1.0 if status == "won" else 0.0,
                now.isoformat(),
                trade_id,
            ))
        
        await self._db.commit()
        
        # Recalculate stats for affected wallets
        wallet_addresses = set(t[1] for t in trades)
        for addr in wallet_addresses:
            await self.recalculate_wallet_stats(addr)
        
        logger.info(f"Resolved market {market_id}: {winning_outcome} - {len(trades)} trades updated")
    
    # ========================================================================
    # STATS CALCULATION
    # ========================================================================
    
    async def recalculate_wallet_stats(self, address: str):
        """Recalculate and cache wallet statistics."""
        
        async with self._db.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as losses,
                SUM(COALESCE(pnl_usd, 0)) as total_pnl,
                SUM(amount_usd) as total_volume,
                MAX(amount_usd) as largest_trade
            FROM trades
            WHERE wallet_address = ?
        """, (address.lower(),)) as cursor:
            row = await cursor.fetchone()
        
        if not row or row[0] == 0:
            return
        
        total_trades, wins, losses, total_pnl, total_volume, largest_trade = row
        
        # Calculate derived stats
        completed = wins + losses
        win_rate = wins / completed if completed > 0 else 0.0
        roi_pct = (total_pnl / total_volume * 100) if total_volume > 0 else 0.0
        
        # Reputation score (0-1)
        # Based on: win rate, number of trades, total PnL
        trade_factor = min(total_trades / 100, 1.0)  # More trades = more reliable
        win_factor = win_rate
        profit_factor = min(max(total_pnl / 100000, -1), 1) / 2 + 0.5  # Normalized PnL
        
        reputation = (
            0.4 * win_factor +
            0.3 * trade_factor +
            0.3 * profit_factor
        )
        
        await self._db.execute("""
            INSERT INTO wallet_stats (
                address, total_trades, winning_trades, losing_trades,
                total_pnl_usd, total_volume_usd, win_rate, roi_pct,
                reputation_score, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(address) DO UPDATE SET
                total_trades = excluded.total_trades,
                winning_trades = excluded.winning_trades,
                losing_trades = excluded.losing_trades,
                total_pnl_usd = excluded.total_pnl_usd,
                total_volume_usd = excluded.total_volume_usd,
                win_rate = excluded.win_rate,
                roi_pct = excluded.roi_pct,
                reputation_score = excluded.reputation_score,
                updated_at = CURRENT_TIMESTAMP
        """, (
            address.lower(),
            total_trades,
            wins,
            losses,
            total_pnl,
            total_volume,
            win_rate,
            roi_pct,
            reputation,
        ))
        
        await self._db.commit()
    
    async def recalculate_all_stats(self):
        """Recalculate stats for all wallets."""
        
        async with self._db.execute("SELECT address FROM wallets") as cursor:
            wallets = await cursor.fetchall()
        
        for (address,) in wallets:
            await self.recalculate_wallet_stats(address)
        
        logger.info(f"Recalculated stats for {len(wallets)} wallets")
    
    # ========================================================================
    # LEADERBOARD
    # ========================================================================
    
    async def get_leaderboard(
        self,
        metric: str = "total_pnl",  # "total_pnl", "win_rate", "reputation"
        limit: int = 50,
        min_trades: int = 10,
    ) -> list[WalletStats]:
        """Get top wallets by metric."""
        
        order_col = {
            "total_pnl": "s.total_pnl_usd",
            "win_rate": "s.win_rate",
            "reputation": "s.reputation_score",
            "volume": "s.total_volume_usd",
        }.get(metric, "s.total_pnl_usd")
        
        wallets = await self.get_all_wallets(min_trades=min_trades)
        
        # Sort by metric
        if metric == "total_pnl":
            wallets.sort(key=lambda w: w.total_pnl_usd, reverse=True)
        elif metric == "win_rate":
            wallets.sort(key=lambda w: w.win_rate, reverse=True)
        elif metric == "reputation":
            wallets.sort(key=lambda w: w.reputation_score, reverse=True)
        
        return wallets[:limit]
    
    # ========================================================================
    # IMPORT/EXPORT
    # ========================================================================
    
    async def import_known_wallets(self, wallets_file: str):
        """Import known wallet labels from a JSON file."""
        
        try:
            with open(wallets_file) as f:
                wallets = json.load(f)
            
            for wallet in wallets:
                await self.add_wallet(
                    address=wallet["address"],
                    label=wallet.get("label"),
                    categories=wallet.get("categories", []),
                    is_arb_bot=wallet.get("is_arb_bot", False),
                )
            
            logger.info(f"Imported {len(wallets)} wallets from {wallets_file}")
            
        except Exception as e:
            logger.error(f"Failed to import wallets: {e}")
    
    async def export_stats(self, output_file: str):
        """Export all wallet stats to JSON."""
        
        wallets = await self.get_all_wallets()
        
        data = [w.to_dict() for w in wallets]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(wallets)} wallet stats to {output_file}")


# ============================================================================
# WALLET DISCOVERY (find new sharps)
# ============================================================================

class WalletDiscovery:
    """
    Discovers new sharp wallets by analyzing on-chain data.
    """
    
    def __init__(self, db: WhaleDatabase, alchemy_api_key: str):
        self.db = db
        self.alchemy_key = alchemy_api_key
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def discover_from_market(self, market_id: str, min_trades: int = 3) -> list[str]:
        """
        Find wallets that traded a specific market profitably.
        Call this after markets resolve to find good wallets.
        """
        
        # Get all trades for this resolved market
        async with self.db._db.execute("""
            SELECT DISTINCT wallet_address
            FROM trades
            WHERE market_id = ? AND status = 'won'
            GROUP BY wallet_address
            HAVING COUNT(*) >= ?
        """, (market_id, min_trades)) as cursor:
            winners = await cursor.fetchall()
        
        return [w[0] for w in winners]
    
    async def scan_polymarket_api(self, limit: int = 100) -> list[str]:
        """
        Use Polymarket's public API to find active traders.
        """
        
        new_wallets = []
        
        try:
            # Get recent large trades from Polymarket
            resp = await self.http_client.get(
                "https://gamma-api.polymarket.com/trades",
                params={"limit": limit, "min_size": 10000}
            )
            
            if resp.status_code == 200:
                trades = resp.json()
                
                for trade in trades:
                    wallet = trade.get("maker") or trade.get("taker")
                    if wallet:
                        new_wallets.append(wallet.lower())
        
        except Exception as e:
            logger.error(f"Failed to scan Polymarket API: {e}")
        
        return list(set(new_wallets))
    
    async def close(self):
        await self.http_client.aclose()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

whale_db: Optional[WhaleDatabase] = None


async def get_whale_db() -> WhaleDatabase:
    """Get or create the whale database singleton."""
    global whale_db
    
    if whale_db is None:
        whale_db = WhaleDatabase()
        await whale_db.connect()
    
    return whale_db
