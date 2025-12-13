"""
Comprehensive Logger - Structured logging for everything.

Logs to:
1. Console (human readable)
2. JSON file (machine parseable for analysis)
3. SQLite (via real_data_store for persistence)

Tracks:
- Market snapshots
- ML predictions  
- Grok calls
- Betting decisions
- Trades (entries and exits)
- PnL
- Errors and warnings
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Literal
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: str
    level: str
    category: str  # market, prediction, grok, trade, system
    symbol: Optional[str]
    message: str
    data: Optional[dict]


class JsonFileHandler(logging.Handler):
    """Handler that writes JSON lines to a file."""
    
    def __init__(self, log_dir: str = "./logs"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._get_file()
    
    def _get_file(self):
        """Get current log file (one per day)."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.current_file = self.log_dir / f"trading_{date_str}.jsonl"
    
    def emit(self, record: logging.LogRecord):
        try:
            # Get structured data if available
            data = getattr(record, 'data', None)
            category = getattr(record, 'category', 'system')
            symbol = getattr(record, 'symbol', None)
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "category": category,
                "symbol": symbol,
                "message": record.getMessage(),
                "data": data,
            }
            
            with self._lock:
                # Check if we need new file (day changed)
                self._get_file()
                
                with open(self.current_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
                    
        except Exception as e:
            print(f"Logging error: {e}", file=sys.stderr)


class ColorFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
    }
    RESET = '\033[0m'
    
    SYMBOLS = {
        'market': '📊',
        'prediction': '🔮',
        'grok': '🤖',
        'trade': '💰',
        'system': '⚙️',
    }
    
    def format(self, record: logging.LogRecord):
        # Get category symbol
        category = getattr(record, 'category', 'system')
        symbol_emoji = self.SYMBOLS.get(category, '•')
        
        # Get crypto symbol if present
        crypto = getattr(record, 'symbol', '')
        crypto_str = f"[{crypto}] " if crypto else ""
        
        # Color based on level
        color = self.COLORS.get(record.levelname, '')
        
        # Format timestamp
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        
        # Build message
        msg = f"{color}{ts} {symbol_emoji} {crypto_str}{record.getMessage()}{self.RESET}"
        
        return msg


class TradingLogger:
    """
    Comprehensive logger for the trading system.
    
    Usage:
        log = TradingLogger()
        log.market("BTC", "Price updated", {"yes_price": 0.55})
        log.prediction("ETH", "ML predicts UP", {"confidence": 0.72})
        log.trade("SOL", "Entered position", {"direction": "UP", "size": 100})
    """
    
    def __init__(
        self,
        name: str = "trading",
        log_dir: str = "./logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler with colors
        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(ColorFormatter())
        self.logger.addHandler(console)
        
        # JSON file handler
        json_handler = JsonFileHandler(log_dir)
        json_handler.setLevel(file_level)
        self.logger.addHandler(json_handler)
        
        self.log_dir = Path(log_dir)
    
    def _log(
        self,
        level: int,
        category: str,
        symbol: Optional[str],
        message: str,
        data: Optional[dict] = None,
    ):
        """Internal log method with structured data."""
        extra = {
            'category': category,
            'symbol': symbol,
            'data': data,
        }
        self.logger.log(level, message, extra=extra)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Category-specific methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def market(
        self, 
        symbol: str, 
        message: str, 
        data: Optional[dict] = None,
        level: int = logging.INFO,
    ):
        """Log market data events."""
        self._log(level, "market", symbol, message, data)
    
    def prediction(
        self,
        symbol: str,
        message: str,
        data: Optional[dict] = None,
        level: int = logging.INFO,
    ):
        """Log prediction events."""
        self._log(level, "prediction", symbol, message, data)
    
    def grok(
        self,
        symbol: str,
        message: str,
        data: Optional[dict] = None,
        level: int = logging.INFO,
    ):
        """Log Grok API events."""
        self._log(level, "grok", symbol, message, data)
    
    def trade(
        self,
        symbol: str,
        message: str,
        data: Optional[dict] = None,
        level: int = logging.INFO,
    ):
        """Log trade events."""
        self._log(level, "trade", symbol, message, data)
    
    def system(
        self,
        message: str,
        data: Optional[dict] = None,
        level: int = logging.INFO,
    ):
        """Log system events."""
        self._log(level, "system", None, message, data)
    
    def error(
        self,
        message: str,
        data: Optional[dict] = None,
        symbol: Optional[str] = None,
    ):
        """Log errors."""
        self._log(logging.ERROR, "system", symbol, f"ERROR: {message}", data)
    
    def warning(
        self,
        message: str,
        data: Optional[dict] = None,
        symbol: Optional[str] = None,
    ):
        """Log warnings."""
        self._log(logging.WARNING, "system", symbol, f"WARNING: {message}", data)
    
    # ─────────────────────────────────────────────────────────────────────────
    # High-level logging methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def log_snapshot(
        self,
        symbol: str,
        yes_price: float,
        no_price: float,
        market_direction: str,
        **extra,
    ):
        """Log a market snapshot."""
        self.market(
            symbol,
            f"YES={yes_price:.3f} NO={no_price:.3f} → {market_direction}",
            {
                "yes_price": yes_price,
                "no_price": no_price,
                "market_direction": market_direction,
                **extra,
            }
        )
    
    def log_ml_prediction(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        should_bet: bool,
        reasoning: str,
        **extra,
    ):
        """Log an ML prediction."""
        bet_str = "✓ BET" if should_bet else "✗ SKIP"
        self.prediction(
            symbol,
            f"ML → {direction} ({confidence:.1%}) {bet_str}",
            {
                "direction": direction,
                "confidence": confidence,
                "should_bet": should_bet,
                "reasoning": reasoning,
                **extra,
            }
        )
    
    def log_grok_call(
        self,
        symbol: str,
        triggered_by: list[str],
        direction: str,
        confidence: float,
        reasoning: str,
    ):
        """Log a Grok API call."""
        self.grok(
            symbol,
            f"Grok → {direction} ({confidence:.1%}) [triggered: {', '.join(triggered_by)}]",
            {
                "direction": direction,
                "confidence": confidence,
                "triggered_by": triggered_by,
                "reasoning": reasoning,
            }
        )
    
    def log_bet_decision(
        self,
        symbol: str,
        should_bet: bool,
        direction: str,
        confidence: float,
        position_size_pct: float,
        passed_filters: list[str],
        failed_filters: list[str],
    ):
        """Log a betting decision."""
        if should_bet:
            self.trade(
                symbol,
                f"BET {direction} @ {confidence:.1%} (size={position_size_pct:.1%})",
                {
                    "action": "bet",
                    "direction": direction,
                    "confidence": confidence,
                    "position_size_pct": position_size_pct,
                    "passed_filters": passed_filters,
                }
            )
        else:
            self.trade(
                symbol,
                f"SKIP - Failed: {', '.join(failed_filters)}",
                {
                    "action": "skip",
                    "failed_filters": failed_filters,
                    "passed_filters": passed_filters,
                },
                level=logging.DEBUG,
            )
    
    def log_trade_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size_usd: float,
        trade_id: int,
    ):
        """Log a trade entry."""
        self.trade(
            symbol,
            f"ENTRY #{trade_id}: {direction} @ {entry_price:.3f} (${size_usd:.2f})",
            {
                "trade_id": trade_id,
                "action": "entry",
                "direction": direction,
                "entry_price": entry_price,
                "size_usd": size_usd,
            }
        )
    
    def log_trade_exit(
        self,
        symbol: str,
        trade_id: int,
        exit_price: float,
        pnl: float,
        was_correct: bool,
    ):
        """Log a trade exit."""
        result = "WIN ✓" if was_correct else "LOSS ✗"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        
        self.trade(
            symbol,
            f"EXIT #{trade_id}: {result} {pnl_str}",
            {
                "trade_id": trade_id,
                "action": "exit",
                "exit_price": exit_price,
                "pnl": pnl,
                "was_correct": was_correct,
            }
        )
    
    def log_daily_summary(self, stats: dict):
        """Log daily trading summary."""
        self.system(
            f"DAILY: {stats.get('total_trades', 0)} trades, "
            f"{stats.get('win_rate', 0):.1%} win rate, "
            f"${stats.get('total_pnl', 0):.2f} PnL",
            stats,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Log Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_recent_logs(
        self,
        category: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Read recent logs from JSON file."""
        logs = []
        
        # Get today's log file
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"trading_{date_str}.jsonl"
        
        if not log_file.exists():
            return logs
        
        try:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        
                        # Filter by category
                        if category and event.get("category") != category:
                            continue
                        
                        # Filter by symbol
                        if symbol and event.get("symbol") != symbol:
                            continue
                        
                        logs.append(event)
                        
                    except json.JSONDecodeError:
                        continue
            
            # Return most recent
            return logs[-limit:]
            
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_logger: Optional[TradingLogger] = None

def get_logger() -> TradingLogger:
    """Get the singleton trading logger."""
    global _logger
    if _logger is None:
        _logger = TradingLogger()
    return _logger


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log = get_logger()
    
    # Test all log types
    log.system("Trading system starting")
    
    log.log_snapshot("BTC", 0.55, 0.45, "UP", volume=1000)
    log.log_snapshot("ETH", 0.48, 0.52, "DOWN", volume=500)
    
    log.log_ml_prediction(
        "BTC", "UP", 0.72, True,
        "Strong momentum indicators"
    )
    
    log.log_grok_call(
        "BTC", 
        ["high_value", "market_divergence"],
        "UP", 0.68,
        "Technical indicators suggest upward pressure"
    )
    
    log.log_bet_decision(
        "BTC", True, "UP", 0.72, 0.15,
        ["DATA: OK", "CONFIDENCE: OK"],
        []
    )
    
    log.log_trade_entry("BTC", "UP", 0.55, 100, 1)
    log.log_trade_exit("BTC", 1, 1.0, 45.5, True)
    
    log.log_daily_summary({
        "total_trades": 10,
        "wins": 6,
        "losses": 4,
        "win_rate": 0.6,
        "total_pnl": 125.50,
    })
    
    log.error("Test error message")
    log.warning("Test warning message")
    
    print("\n" + "="*60)
    print("Recent logs from file:")
    for event in log.get_recent_logs(limit=5):
        print(f"  {event['category']}: {event['message']}")
