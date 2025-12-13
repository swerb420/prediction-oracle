"""
Storage abstraction for SQLite + Parquet.
Handles persistence of candles, markets, trades, and model artifacts.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import aiosqlite
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.config import get_settings
from core.logging_utils import get_logger
from core.time_utils import now_utc
from .schemas import Candle, PolymarketMarket, Trade, GrokRegimeOutput

logger = get_logger(__name__)

T = TypeVar("T")


class Storage:
    """
    Unified storage layer with SQLite for metadata and Parquet for time series.
    """
    
    def __init__(
        self,
        db_path: Path | None = None,
        parquet_dir: Path | None = None
    ):
        settings = get_settings()
        self.db_path = db_path or settings.sqlite_db_path
        self.parquet_dir = parquet_dir or settings.parquet_data_dir
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
    
    async def connect(self) -> None:
        """Initialize database connection and create tables."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        logger.info(f"Connected to SQLite: {self.db_path}")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
    
    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if not self._db:
            return
        
        await self._db.executescript("""
            -- Candles metadata
            CREATE TABLE IF NOT EXISTS candles_meta (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                first_timestamp TEXT,
                last_timestamp TEXT,
                row_count INTEGER DEFAULT 0,
                updated_at TEXT,
                PRIMARY KEY (symbol, interval)
            );
            
            -- Polymarket markets
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                condition_id TEXT,
                question TEXT,
                description TEXT,
                category TEXT,
                end_date TEXT,
                outcomes_json TEXT,
                volume_24h REAL,
                liquidity REAL,
                is_active INTEGER,
                created_at TEXT,
                updated_at TEXT
            );
            
            -- Grok regime outputs
            CREATE TABLE IF NOT EXISTS grok_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime_label TEXT,
                sentiment_btc REAL,
                sentiment_eth REAL,
                sentiment_sol REAL,
                event_risk REAL,
                reasoning TEXT,
                confidence REAL,
                created_at TEXT
            );
            
            -- Model registry
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_type TEXT,
                symbol TEXT,
                created_at TEXT,
                metrics_json TEXT,
                is_champion INTEGER DEFAULT 0,
                file_path TEXT
            );
            
            -- Paper trades
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                market_id TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                size_usd REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                entry_time TEXT,
                exit_time TEXT,
                strategy TEXT,
                metadata_json TEXT
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
            CREATE INDEX IF NOT EXISTS idx_markets_end_date ON markets(end_date);
            CREATE INDEX IF NOT EXISTS idx_grok_timestamp ON grok_regimes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON paper_trades(symbol);
        """)
        await self._db.commit()
    
    # ==================== CANDLE STORAGE (Parquet) ====================
    
    async def save_candles(
        self,
        symbol: str,
        interval: str,
        candles: list[Candle]
    ) -> None:
        """Save candles to Parquet file."""
        if not candles:
            return
        
        async with self._lock:
            # Convert to DataFrame
            df = pd.DataFrame([c.model_dump() for c in candles])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            
            # Parquet path
            parquet_path = self.parquet_dir / f"candles_{symbol}_{interval}.parquet"
            
            # Append to existing if present
            if parquet_path.exists():
                existing_df = pd.read_parquet(parquet_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            
            # Write Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path, compression="snappy")
            
            # Update metadata
            if self._db:
                await self._db.execute("""
                    INSERT OR REPLACE INTO candles_meta 
                    (symbol, interval, first_timestamp, last_timestamp, row_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    interval,
                    df["timestamp"].min().isoformat(),
                    df["timestamp"].max().isoformat(),
                    len(df),
                    now_utc().isoformat()
                ))
                await self._db.commit()
            
            logger.debug(f"Saved {len(candles)} candles for {symbol}/{interval}")
    
    async def load_candles(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None
    ) -> list[Candle]:
        """Load candles from Parquet file."""
        parquet_path = self.parquet_dir / f"candles_{symbol}_{interval}.parquet"
        
        if not parquet_path.exists():
            return []
        
        df = pd.read_parquet(parquet_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by time range
        if start:
            df = df[df["timestamp"] >= start]
        if end:
            df = df[df["timestamp"] <= end]
        
        # Sort and limit
        df = df.sort_values("timestamp", ascending=False)
        if limit:
            df = df.head(limit)
        
        df = df.sort_values("timestamp")
        
        return [Candle(**row) for row in df.to_dict("records")]
    
    async def get_latest_candle(self, symbol: str, interval: str) -> Candle | None:
        """Get the most recent candle."""
        candles = await self.load_candles(symbol, interval, limit=1)
        return candles[-1] if candles else None
    
    # ==================== MARKET STORAGE (SQLite) ====================
    
    async def save_market(self, market: PolymarketMarket) -> None:
        """Save or update a Polymarket market."""
        if not self._db:
            return
        
        outcomes_json = json.dumps([o.model_dump() for o in market.outcomes])
        
        await self._db.execute("""
            INSERT OR REPLACE INTO markets
            (market_id, condition_id, question, description, category, end_date,
             outcomes_json, volume_24h, liquidity, is_active, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market.market_id,
            market.condition_id,
            market.question,
            market.description,
            market.category,
            market.end_date.isoformat() if market.end_date else None,
            outcomes_json,
            market.volume_24h,
            market.liquidity,
            1 if market.is_active else 0,
            now_utc().isoformat()
        ))
        await self._db.commit()
    
    async def save_markets(self, markets: list[PolymarketMarket]) -> None:
        """Bulk save markets."""
        for market in markets:
            await self.save_market(market)
    
    async def load_markets(
        self,
        category: str | None = None,
        active_only: bool = True,
        resolving_within_minutes: int | None = None
    ) -> list[PolymarketMarket]:
        """Load markets with optional filters."""
        if not self._db:
            return []
        
        query = "SELECT * FROM markets WHERE 1=1"
        params: list[Any] = []
        
        if active_only:
            query += " AND is_active = 1"
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if resolving_within_minutes:
            cutoff = (now_utc() + timedelta(minutes=resolving_within_minutes)).isoformat()
            query += " AND end_date IS NOT NULL AND end_date <= ? AND end_date > ?"
            params.extend([cutoff, now_utc().isoformat()])
        
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        
        markets = []
        for row in rows:
            from .schemas import PolymarketOutcome
            outcomes = [
                PolymarketOutcome(**o) 
                for o in json.loads(row[6] or "[]")
            ]
            markets.append(PolymarketMarket(
                market_id=row[0],
                condition_id=row[1],
                question=row[2],
                description=row[3] or "",
                category=row[4] or "",
                end_date=datetime.fromisoformat(row[5]) if row[5] else None,
                outcomes=outcomes,
                volume_24h=row[7] or 0,
                liquidity=row[8] or 0,
                is_active=bool(row[9])
            ))
        
        return markets
    
    # ==================== GROK REGIME STORAGE ====================
    
    async def save_grok_regime(self, regime: GrokRegimeOutput) -> None:
        """Save Grok regime output."""
        if not self._db:
            return
        
        await self._db.execute("""
            INSERT INTO grok_regimes
            (timestamp, regime_label, sentiment_btc, sentiment_eth, sentiment_sol,
             event_risk, reasoning, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            regime.timestamp.isoformat(),
            regime.regime_label,
            regime.sentiment_btc,
            regime.sentiment_eth,
            regime.sentiment_sol,
            regime.event_risk,
            regime.reasoning,
            regime.confidence,
            now_utc().isoformat()
        ))
        await self._db.commit()
    
    async def get_latest_grok_regime(self) -> GrokRegimeOutput | None:
        """Get the most recent Grok regime output."""
        if not self._db:
            return None
        
        async with self._db.execute(
            "SELECT * FROM grok_regimes ORDER BY timestamp DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        
        if not row:
            return None
        
        return GrokRegimeOutput(
            timestamp=datetime.fromisoformat(row[1]),
            regime_label=row[2],
            sentiment_btc=row[3],
            sentiment_eth=row[4],
            sentiment_sol=row[5],
            event_risk=row[6],
            reasoning=row[7] or "",
            confidence=row[8]
        )
    
    # ==================== PAPER TRADES ====================
    
    async def save_paper_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size_usd: float,
        pnl_usd: float,
        pnl_pct: float,
        entry_time: datetime,
        exit_time: datetime,
        strategy: str,
        market_id: str | None = None,
        metadata: dict | None = None
    ) -> None:
        """Save completed paper trade."""
        if not self._db:
            return
        
        await self._db.execute("""
            INSERT INTO paper_trades
            (trade_id, symbol, market_id, direction, entry_price, exit_price,
             size_usd, pnl_usd, pnl_pct, entry_time, exit_time, strategy, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            symbol,
            market_id,
            direction,
            entry_price,
            exit_price,
            size_usd,
            pnl_usd,
            pnl_pct,
            entry_time.isoformat(),
            exit_time.isoformat(),
            strategy,
            json.dumps(metadata or {})
        ))
        await self._db.commit()
    
    async def get_paper_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        limit: int = 100
    ) -> list[dict]:
        """Get paper trades with optional filters."""
        if not self._db:
            return []
        
        query = "SELECT * FROM paper_trades WHERE 1=1"
        params: list[Any] = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        
        query += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)
        
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        
        return [
            {
                "trade_id": row[0],
                "symbol": row[1],
                "market_id": row[2],
                "direction": row[3],
                "entry_price": row[4],
                "exit_price": row[5],
                "size_usd": row[6],
                "pnl_usd": row[7],
                "pnl_pct": row[8],
                "entry_time": datetime.fromisoformat(row[9]),
                "exit_time": datetime.fromisoformat(row[10]),
                "strategy": row[11],
                "metadata": json.loads(row[12] or "{}")
            }
            for row in rows
        ]


# Singleton storage instance
_storage: Storage | None = None


async def get_storage() -> Storage:
    """Get or create global storage instance."""
    global _storage
    if _storage is None:
        _storage = Storage()
        await _storage.connect()
    return _storage


class SyncStorage:
    """
    Synchronous storage wrapper for simple operations.
    Uses SQLite directly without async.
    """
    
    def __init__(self, base_path: Path):
        import sqlite3
        self.base_path = base_path
        self.db_path = base_path / "data.db"
        self.parquet_dir = base_path / "parquet"
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(str(self.db_path))
    
    def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute SQL statement."""
        self._conn.execute(sql, params)
        self._conn.commit()
    
    def fetch_one(self, sql: str, params: tuple = ()) -> tuple | None:
        """Fetch one row."""
        cursor = self._conn.execute(sql, params)
        return cursor.fetchone()
    
    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Fetch all rows."""
        cursor = self._conn.execute(sql, params)
        return cursor.fetchall()
    
    def save_parquet(self, df: pd.DataFrame, name: str) -> None:
        """Save DataFrame to Parquet."""
        path = self.parquet_dir / f"{name}.parquet"
        df.to_parquet(path, compression="snappy")
    
    def load_parquet(self, name: str) -> pd.DataFrame:
        """Load DataFrame from Parquet."""
        path = self.parquet_dir / f"{name}.parquet"
        return pd.read_parquet(path)
    
    def close(self) -> None:
        """Close connection."""
        self._conn.close()
