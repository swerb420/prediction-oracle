"""
Real Data Store - SQLite-based storage for Polymarket data.

NO FAKE DATA. Everything is real market data with timestamps.

Stores:
1. Market snapshots - 15M market prices over time
2. Outcomes - What actually happened (UP/DOWN) after resolution
3. Trades - Paper/real trades we made
4. Model performance - Accuracy metrics over time
5. Grok calls - When we called Grok and what it said

All data is timestamped and used for ML training.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of a 15M market."""
    symbol: CryptoSymbol
    timestamp: str  # ISO format
    question: str
    yes_price: float
    no_price: float
    market_direction: str  # What the market implied
    condition_id: str
    event_slug: str
    volume: float = 0.0
    liquidity: float = 0.0
    

@dataclass
class MarketOutcome:
    """The actual outcome of a 15M market after resolution."""
    symbol: CryptoSymbol
    event_slug: str
    start_time: str  # ISO format
    end_time: str  # ISO format
    actual_outcome: str  # "UP" or "DOWN"
    starting_price: Optional[float] = None  # Crypto price at start
    ending_price: Optional[float] = None  # Crypto price at end
    price_change_pct: Optional[float] = None


@dataclass
class PaperTrade:
    """A paper trade we made."""
    id: Optional[int] = None
    symbol: CryptoSymbol = "BTC"
    timestamp: str = ""
    direction: str = ""  # "UP" or "DOWN"
    entry_price: float = 0.0  # Polymarket price we entered at
    size_usd: float = 0.0
    confidence: float = 0.0
    ml_confidence: float = 0.0
    grok_used: bool = False
    grok_agreed: bool = False
    # Detailed signal data for analysis
    orderbook_signal: float = 0.0
    momentum_signal: float = 0.0
    window_number: int = 0
    secs_into_window: int = 0
    grok_confidence: float = 0.0
    grok_reasoning: str = ""
    poly_yes_price: float = 0.0
    poly_no_price: float = 0.0
    spot_price: float = 0.0
    timeframe: str = "15M"  # "15M", "1H", or "4H"
    # ═══════════════════════════════════════════════════════════════════════════
    # ENHANCED LOGGING - Full context for analysis
    # ═══════════════════════════════════════════════════════════════════════════
    # Market context
    market_spread: float = 0.0          # YES-NO spread (liquidity indicator)
    volume_24h: float = 0.0             # 24h volume on Polymarket
    liquidity: float = 0.0              # Total liquidity in market
    # Multi-venue price data
    binance_price: float = 0.0
    bybit_price: float = 0.0
    coinbase_price: float = 0.0
    venue_agreement: float = 0.0        # How aligned venues are (0-1)
    # Orderbook details
    ob_bid_depth: float = 0.0           # Total bid side depth
    ob_ask_depth: float = 0.0           # Total ask side depth
    ob_top_bid: float = 0.0             # Best bid price
    ob_top_ask: float = 0.0             # Best ask price
    ob_weighted_mid: float = 0.0        # Volume-weighted mid price
    # Momentum details
    momentum_1min: float = 0.0          # 1-min price change %
    momentum_5min: float = 0.0          # 5-min price change %
    momentum_15min: float = 0.0         # 15-min price change %
    momentum_1h: float = 0.0            # 1-hour price change %
    momentum_trend: str = ""            # "accelerating", "decelerating", "stable"
    # Signal breakdown (for ML refinement)
    signal_raw_combined: float = 0.0    # Raw combined signal before processing
    signal_weights: str = ""            # JSON: {"ob": 0.25, "mom": 0.25, ...}
    signal_reasons: str = ""            # JSON array of reasons
    # Grok analysis details
    grok_model_used: str = ""           # Which Grok model (fast/deep)
    grok_key_factors: str = ""          # JSON array of key factors
    grok_action: str = ""               # BUY/SELL/WAIT
    grok_urgency: str = ""              # immediate/soon/wait
    grok_full_response: str = ""        # Full raw response for analysis
    grok_cost: float = 0.0              # Cost of this Grok call
    # Entry timing
    time_until_window_end: int = 0      # Seconds until window closes
    entry_reason: str = ""              # Primary reason for entry
    # Filled after resolution
    exit_price: Optional[float] = None
    actual_outcome: Optional[str] = None
    pnl: Optional[float] = None
    was_correct: Optional[bool] = None
    closed_at: Optional[str] = None
    # Post-trade analysis
    price_at_close: float = 0.0         # Crypto price when trade closed
    price_change_during: float = 0.0    # % change during trade
    max_drawdown: float = 0.0           # Max loss during trade
    max_profit: float = 0.0             # Max profit during trade
    early_exit_reason: str = ""         # If exited early, why


@dataclass  
class ModelMetrics:
    """Track model performance over time."""
    timestamp: str
    model_version: str
    accuracy_7d: float
    accuracy_30d: float
    total_predictions: int
    win_rate: float
    avg_confidence_when_right: float
    avg_confidence_when_wrong: float
    edge_captured: float  # Actual returns vs expected


class RealDataStore:
    """
    SQLite-based store for real market data.
    
    All data is real - no synthetic/fake data allowed.
    Used for ML training and backtesting.
    """
    
    def __init__(self, db_path: str = "./data/polymarket_real.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market snapshots - real-time price data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                question TEXT,
                yes_price REAL NOT NULL,
                no_price REAL NOT NULL,
                market_direction TEXT,
                condition_id TEXT,
                event_slug TEXT,
                volume REAL DEFAULT 0,
                liquidity REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, event_slug)
            )
        """)
        
        # Market outcomes - what actually happened
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                event_slug TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                actual_outcome TEXT NOT NULL,
                starting_price REAL,
                ending_price REAL,
                price_change_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_slug)
            )
        """)
        
        # Paper trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size_usd REAL NOT NULL,
                confidence REAL,
                ml_confidence REAL,
                grok_used INTEGER DEFAULT 0,
                grok_agreed INTEGER DEFAULT 0,
                exit_price REAL,
                actual_outcome TEXT,
                pnl REAL,
                was_correct INTEGER,
                closed_at TEXT,
                event_slug TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                timeframe TEXT DEFAULT '15M'
            )
        """)
        
        # Add timeframe column if not exists (for existing databases)
        try:
            cursor.execute("ALTER TABLE paper_trades ADD COLUMN timeframe TEXT DEFAULT '15M'")
        except:
            pass  # Column already exists
        
        # Add all new detailed logging columns
        new_columns = [
            ("orderbook_signal", "REAL DEFAULT 0"),
            ("momentum_signal", "REAL DEFAULT 0"),
            ("window_number", "INTEGER DEFAULT 0"),
            ("secs_into_window", "INTEGER DEFAULT 0"),
            ("grok_confidence", "REAL DEFAULT 0"),
            ("grok_reasoning", "TEXT DEFAULT ''"),
            ("poly_yes_price", "REAL DEFAULT 0"),
            ("poly_no_price", "REAL DEFAULT 0"),
            ("spot_price", "REAL DEFAULT 0"),
            # Enhanced logging columns
            ("market_spread", "REAL DEFAULT 0"),
            ("volume_24h", "REAL DEFAULT 0"),
            ("liquidity", "REAL DEFAULT 0"),
            ("binance_price", "REAL DEFAULT 0"),
            ("bybit_price", "REAL DEFAULT 0"),
            ("coinbase_price", "REAL DEFAULT 0"),
            ("venue_agreement", "REAL DEFAULT 0"),
            ("ob_bid_depth", "REAL DEFAULT 0"),
            ("ob_ask_depth", "REAL DEFAULT 0"),
            ("ob_top_bid", "REAL DEFAULT 0"),
            ("ob_top_ask", "REAL DEFAULT 0"),
            ("ob_weighted_mid", "REAL DEFAULT 0"),
            ("momentum_1min", "REAL DEFAULT 0"),
            ("momentum_5min", "REAL DEFAULT 0"),
            ("momentum_15min", "REAL DEFAULT 0"),
            ("momentum_1h", "REAL DEFAULT 0"),
            ("momentum_trend", "TEXT DEFAULT ''"),
            ("signal_raw_combined", "REAL DEFAULT 0"),
            ("signal_weights", "TEXT DEFAULT ''"),
            ("signal_reasons", "TEXT DEFAULT ''"),
            ("grok_model_used", "TEXT DEFAULT ''"),
            ("grok_key_factors", "TEXT DEFAULT ''"),
            ("grok_action", "TEXT DEFAULT ''"),
            ("grok_urgency", "TEXT DEFAULT ''"),
            ("grok_full_response", "TEXT DEFAULT ''"),
            ("grok_cost", "REAL DEFAULT 0"),
            ("time_until_window_end", "INTEGER DEFAULT 0"),
            ("entry_reason", "TEXT DEFAULT ''"),
            ("price_at_close", "REAL DEFAULT 0"),
            ("price_change_during", "REAL DEFAULT 0"),
            ("max_drawdown", "REAL DEFAULT 0"),
            ("max_profit", "REAL DEFAULT 0"),
            ("early_exit_reason", "TEXT DEFAULT ''"),
        ]
        for col_name, col_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
            except:
                pass  # Column already exists
        
        # Predictions (for ML training feedback)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_slug TEXT,
                predicted_direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                ml_features TEXT,  -- JSON blob of features used
                grok_response TEXT,  -- JSON blob if Grok was used
                actual_outcome TEXT,  -- Filled after resolution
                was_correct INTEGER,  -- 1 or 0
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model metrics over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT,
                accuracy_7d REAL,
                accuracy_30d REAL,
                total_predictions INTEGER,
                win_rate REAL,
                avg_confidence_when_right REAL,
                avg_confidence_when_wrong REAL,
                edge_captured REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Grok API calls log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grok_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trigger_reasons TEXT,  -- JSON array
                prompt TEXT,
                response TEXT,  -- JSON
                direction TEXT,
                confidence REAL,
                cost_estimate REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_time ON market_snapshots(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_slug ON market_outcomes(event_slug)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON paper_trades(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_outcome ON predictions(actual_outcome)")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Market Snapshots
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """Save a market snapshot. Returns True if new, False if duplicate."""
        try:
            conn = self._conn()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO market_snapshots 
                (symbol, timestamp, question, yes_price, no_price, market_direction,
                 condition_id, event_slug, volume, liquidity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.symbol, snapshot.timestamp, snapshot.question,
                snapshot.yes_price, snapshot.no_price, snapshot.market_direction,
                snapshot.condition_id, snapshot.event_slug,
                snapshot.volume, snapshot.liquidity
            ))
            inserted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return inserted
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False
    
    def get_snapshots(
        self, 
        symbol: Optional[str] = None,
        hours: int = 24,
    ) -> list[dict]:
        """Get recent snapshots for training."""
        conn = self._conn()
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        
        if symbol:
            cursor.execute("""
                SELECT * FROM market_snapshots 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (symbol, cutoff))
        else:
            cursor.execute("""
                SELECT * FROM market_snapshots 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff,))
        
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
    
    def get_snapshot_count(self) -> int:
        """Get total number of snapshots."""
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_snapshots")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    # ─────────────────────────────────────────────────────────────────────────
    # Outcomes
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_outcome(self, outcome: MarketOutcome) -> bool:
        """Save a market outcome."""
        try:
            conn = self._conn()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_outcomes
                (symbol, event_slug, start_time, end_time, actual_outcome,
                 starting_price, ending_price, price_change_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.symbol, outcome.event_slug, outcome.start_time,
                outcome.end_time, outcome.actual_outcome,
                outcome.starting_price, outcome.ending_price, outcome.price_change_pct
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save outcome: {e}")
            return False
    
    def get_labeled_data(self, limit: int = 1000) -> list[dict]:
        """
        Get snapshots with known outcomes for ML training.
        
        Joins snapshots with their actual outcomes.
        """
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                s.symbol, s.timestamp, s.yes_price, s.no_price,
                s.market_direction, s.volume, s.liquidity,
                o.actual_outcome, o.price_change_pct
            FROM market_snapshots s
            INNER JOIN market_outcomes o 
                ON s.event_slug = o.event_slug
            ORDER BY s.timestamp DESC
            LIMIT ?
        """, (limit,))
        
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
    
    def get_outcome_count(self) -> int:
        """Get total number of outcomes."""
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_outcomes")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paper Trades
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_trade(self, trade: PaperTrade) -> int:
        """Save a paper trade with FULL detailed signal data. Returns trade ID."""
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO paper_trades
            (symbol, timestamp, direction, entry_price, size_usd, confidence,
             ml_confidence, grok_used, grok_agreed, event_slug,
             orderbook_signal, momentum_signal, window_number, secs_into_window,
             grok_confidence, grok_reasoning, poly_yes_price, poly_no_price, spot_price, timeframe,
             market_spread, volume_24h, liquidity,
             binance_price, bybit_price, coinbase_price, venue_agreement,
             ob_bid_depth, ob_ask_depth, ob_top_bid, ob_top_ask, ob_weighted_mid,
             momentum_1min, momentum_5min, momentum_15min, momentum_1h, momentum_trend,
             signal_raw_combined, signal_weights, signal_reasons,
             grok_model_used, grok_key_factors, grok_action, grok_urgency, grok_full_response, grok_cost,
             time_until_window_end, entry_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.symbol, trade.timestamp, trade.direction, trade.entry_price,
            trade.size_usd, trade.confidence, trade.ml_confidence,
            1 if trade.grok_used else 0, 1 if trade.grok_agreed else 0,
            getattr(trade, 'event_slug', None),
            trade.orderbook_signal, trade.momentum_signal,
            trade.window_number, trade.secs_into_window,
            trade.grok_confidence, trade.grok_reasoning,
            trade.poly_yes_price, trade.poly_no_price, trade.spot_price,
            getattr(trade, 'timeframe', '15M'),
            # Enhanced logging fields
            getattr(trade, 'market_spread', 0.0),
            getattr(trade, 'volume_24h', 0.0),
            getattr(trade, 'liquidity', 0.0),
            getattr(trade, 'binance_price', 0.0),
            getattr(trade, 'bybit_price', 0.0),
            getattr(trade, 'coinbase_price', 0.0),
            getattr(trade, 'venue_agreement', 0.0),
            getattr(trade, 'ob_bid_depth', 0.0),
            getattr(trade, 'ob_ask_depth', 0.0),
            getattr(trade, 'ob_top_bid', 0.0),
            getattr(trade, 'ob_top_ask', 0.0),
            getattr(trade, 'ob_weighted_mid', 0.0),
            getattr(trade, 'momentum_1min', 0.0),
            getattr(trade, 'momentum_5min', 0.0),
            getattr(trade, 'momentum_15min', 0.0),
            getattr(trade, 'momentum_1h', 0.0),
            getattr(trade, 'momentum_trend', ''),
            getattr(trade, 'signal_raw_combined', 0.0),
            getattr(trade, 'signal_weights', ''),
            getattr(trade, 'signal_reasons', ''),
            getattr(trade, 'grok_model_used', ''),
            getattr(trade, 'grok_key_factors', ''),
            getattr(trade, 'grok_action', ''),
            getattr(trade, 'grok_urgency', ''),
            getattr(trade, 'grok_full_response', ''),
            getattr(trade, 'grok_cost', 0.0),
            getattr(trade, 'time_until_window_end', 0),
            getattr(trade, 'entry_reason', ''),
        ))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id
    
    def close_trade(
        self, 
        trade_id: int, 
        exit_price: float,
        actual_outcome: str,
    ) -> bool:
        """Close a trade with outcome."""
        conn = self._conn()
        cursor = conn.cursor()
        
        # Get trade details
        cursor.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
        
        columns = [d[0] for d in cursor.description]
        trade = dict(zip(columns, row))
        
        # Calculate PnL
        # For early exits (EARLY:*), calculate based on price movement
        if actual_outcome.startswith("EARLY:"):
            # PnL = size * (exit_price - entry_price) / entry_price
            # This works for both UP (YES token) and DOWN (NO token)
            price_change_pct = (exit_price - trade['entry_price']) / trade['entry_price'] if trade['entry_price'] > 0 else 0
            pnl = trade['size_usd'] * price_change_pct
            was_correct = pnl > 0
        else:
            # Binary outcome (UP/DOWN)
            was_correct = (trade['direction'] == actual_outcome)
            if was_correct:
                # We bought at entry_price, sold at 1.0 (winning)
                pnl = trade['size_usd'] * (1.0 - trade['entry_price']) / trade['entry_price']
            else:
                # We bought at entry_price, sold at 0.0 (losing)
                pnl = -trade['size_usd']
        
        cursor.execute("""
            UPDATE paper_trades 
            SET exit_price = ?, actual_outcome = ?, pnl = ?, was_correct = ?, closed_at = ?
            WHERE id = ?
        """, (exit_price, actual_outcome, pnl, 1 if was_correct else 0,
              datetime.now(timezone.utc).isoformat(), trade_id))
        
        conn.commit()
        conn.close()
        return True
    
    def get_open_trades(self) -> list[dict]:
        """Get all open (unclosed) trades."""
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM paper_trades WHERE closed_at IS NULL")
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
    
    def get_trading_stats(self, days: int = 30) -> dict:
        """Get trading statistics."""
        conn = self._conn()
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN was_correct = 1 THEN confidence ELSE NULL END) as avg_conf_wins,
                AVG(CASE WHEN was_correct = 0 THEN confidence ELSE NULL END) as avg_conf_losses,
                SUM(CASE WHEN grok_used = 1 THEN 1 ELSE 0 END) as grok_trades,
                SUM(CASE WHEN grok_used = 1 AND was_correct = 1 THEN 1 ELSE 0 END) as grok_wins
            FROM paper_trades
            WHERE closed_at IS NOT NULL AND timestamp > ?
        """, (cutoff,))
        
        row = cursor.fetchone()
        conn.close()
        
        total = row[0] or 0
        wins = row[1] or 0
        
        return {
            "total_trades": total,
            "wins": wins,
            "losses": row[2] or 0,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": row[3] or 0,
            "avg_confidence_wins": row[4] or 0,
            "avg_confidence_losses": row[5] or 0,
            "grok_trades": row[6] or 0,
            "grok_win_rate": (row[7] or 0) / (row[6] or 1),
        }
    
    def get_stats_by_hours(self, hours: int = 6) -> dict:
        """Get trading statistics for the last N hours."""
        conn = self._conn()
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl
            FROM paper_trades
            WHERE closed_at IS NOT NULL AND timestamp > ?
        """, (cutoff,))
        
        row = cursor.fetchone()
        conn.close()
        
        total = row[0] or 0
        wins = row[1] or 0
        
        return {
            "total": total,
            "wins": wins,
            "losses": row[2] or 0,
            "win_rate": wins / total if total > 0 else 0,
            "pnl": row[3] or 0,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Predictions
    # ─────────────────────────────────────────────────────────────────────────
    
    def save_prediction(
        self,
        symbol: str,
        event_slug: str,
        direction: str,
        confidence: float,
        features: Optional[dict] = None,
        grok_response: Optional[dict] = None,
    ) -> int:
        """Save a prediction for later evaluation."""
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions
            (symbol, timestamp, event_slug, predicted_direction, confidence,
             ml_features, grok_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, datetime.now(timezone.utc).isoformat(), event_slug,
            direction, confidence,
            json.dumps(features) if features else None,
            json.dumps(grok_response) if grok_response else None,
        ))
        pred_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return pred_id
    
    def update_prediction_outcome(self, event_slug: str, actual_outcome: str):
        """Update predictions with actual outcome."""
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE predictions 
            SET actual_outcome = ?,
                was_correct = CASE WHEN predicted_direction = ? THEN 1 ELSE 0 END
            WHERE event_slug = ? AND actual_outcome IS NULL
        """, (actual_outcome, actual_outcome, event_slug))
        conn.commit()
        conn.close()
    
    def get_prediction_accuracy(self, days: int = 7) -> dict:
        """Get prediction accuracy stats."""
        conn = self._conn()
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(was_correct) as correct,
                AVG(CASE WHEN was_correct = 1 THEN confidence ELSE NULL END) as conf_correct,
                AVG(CASE WHEN was_correct = 0 THEN confidence ELSE NULL END) as conf_wrong
            FROM predictions
            WHERE actual_outcome IS NOT NULL AND timestamp > ?
        """, (cutoff,))
        
        row = cursor.fetchone()
        conn.close()
        
        total = row[0] or 0
        correct = row[1] or 0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "avg_confidence_correct": row[2] or 0,
            "avg_confidence_wrong": row[3] or 0,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Grok Calls
    # ─────────────────────────────────────────────────────────────────────────
    
    def log_grok_call(
        self,
        symbol: str,
        trigger_reasons: list[str],
        prompt: str,
        response: dict,
        cost_estimate: float = 0.01,
    ):
        """Log a Grok API call."""
        conn = self._conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO grok_calls
            (timestamp, symbol, trigger_reasons, prompt, response, 
             direction, confidence, cost_estimate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol,
            json.dumps(trigger_reasons),
            prompt,
            json.dumps(response),
            response.get("direction"),
            response.get("confidence"),
            cost_estimate,
        ))
        conn.commit()
        conn.close()
    
    def get_grok_stats(self, days: int = 7) -> dict:
        """Get Grok usage stats."""
        conn = self._conn()
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_calls,
                SUM(cost_estimate) as total_cost,
                AVG(confidence) as avg_confidence
            FROM grok_calls
            WHERE timestamp > ?
        """, (cutoff,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            "total_calls": row[0] or 0,
            "total_cost": row[1] or 0,
            "avg_confidence": row[2] or 0,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_summary(self) -> dict:
        """Get overall data summary."""
        return {
            "snapshots": self.get_snapshot_count(),
            "outcomes": self.get_outcome_count(),
            "labeled_examples": len(self.get_labeled_data(limit=100000)),
            "trading_stats": self.get_trading_stats(),
            "prediction_accuracy": self.get_prediction_accuracy(),
            "grok_stats": self.get_grok_stats(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton access
# ─────────────────────────────────────────────────────────────────────────────

_store: Optional[RealDataStore] = None

def get_store() -> RealDataStore:
    """Get the singleton data store."""
    global _store
    if _store is None:
        _store = RealDataStore()
    return _store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = get_store()
    print("Data store summary:", store.get_summary())
