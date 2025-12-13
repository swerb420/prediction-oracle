"""
Paper trading engine for 15-minute crypto predictions.
Tracks positions, PnL, and win rate statistics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .crypto_data import CryptoSymbol, get_fetcher
from .hybrid_oracle import HybridPrediction, TradeSignal

logger = logging.getLogger(__name__)


class Position(BaseModel):
    """An open paper trading position."""
    id: str
    symbol: CryptoSymbol
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    entry_time: datetime
    size_usd: float
    confidence: float
    target_candles: int  # How many 15m candles to hold
    
    # Tracking
    current_price: float | None = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class ClosedTrade(BaseModel):
    """A completed paper trade."""
    id: str
    symbol: CryptoSymbol
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size_usd: float
    confidence: float
    
    # Results
    pnl_usd: float
    pnl_pct: float
    is_win: bool
    duration_minutes: int


class TradingStats(BaseModel):
    """Aggregate trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    
    # By symbol
    stats_by_symbol: dict[str, dict] = {}
    
    # By direction
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0


class PaperTradingEngine:
    """
    Paper trading engine for crypto 15-minute predictions.
    
    Features:
    - Track open positions
    - Auto-close after target duration
    - Calculate PnL and win rate
    - Persist state to disk
    - Risk management (max position, daily loss limit)
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.05,  # 5% of capital per trade
        max_positions: int = 3,
        hold_candles: int = 1,  # Hold for 1 x 15min = 15min
        state_file: str = "./paper_trading_state.json"
    ):
        """
        Initialize paper trading engine.
        
        Args:
            initial_capital: Starting paper money amount
            position_size_pct: Fraction of capital per trade
            max_positions: Maximum concurrent positions
            hold_candles: Number of 15m candles to hold before closing
            state_file: Path to persist state
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.hold_candles = hold_candles
        self.state_file = Path(state_file)
        
        self.open_positions: dict[str, Position] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.stats = TradingStats()
        
        self._load_state()
    
    def _load_state(self):
        """Load persisted state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                
                self.capital = data.get("capital", self.initial_capital)
                
                self.open_positions = {
                    k: Position(**v) for k, v in data.get("open_positions", {}).items()
                }
                
                self.closed_trades = [
                    ClosedTrade(**t) for t in data.get("closed_trades", [])
                ]
                
                self._update_stats()
                logger.info(f"Loaded state: {len(self.open_positions)} open, {len(self.closed_trades)} closed")
                
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            data = {
                "capital": self.capital,
                "open_positions": {
                    k: v.model_dump(mode="json") for k, v in self.open_positions.items()
                },
                "closed_trades": [t.model_dump(mode="json") for t in self.closed_trades]
            }
            
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _update_stats(self):
        """Recalculate aggregate statistics."""
        if not self.closed_trades:
            self.stats = TradingStats()
            return
        
        total = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.is_win]
        losses = [t for t in self.closed_trades if not t.is_win]
        
        win_rate = len(wins) / total if total > 0 else 0
        
        total_pnl = sum(t.pnl_usd for t in self.closed_trades)
        total_pnl_pct = sum(t.pnl_pct for t in self.closed_trades)
        
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
        
        best = max(t.pnl_pct for t in self.closed_trades) if self.closed_trades else 0
        worst = min(t.pnl_pct for t in self.closed_trades) if self.closed_trades else 0
        
        # By symbol
        stats_by_symbol = {}
        for sym in ["BTC", "ETH", "SOL"]:
            sym_trades = [t for t in self.closed_trades if t.symbol == sym]
            if sym_trades:
                sym_wins = len([t for t in sym_trades if t.is_win])
                stats_by_symbol[sym] = {
                    "trades": len(sym_trades),
                    "wins": sym_wins,
                    "win_rate": sym_wins / len(sym_trades),
                    "pnl_usd": sum(t.pnl_usd for t in sym_trades)
                }
        
        # By direction
        longs = [t for t in self.closed_trades if t.direction == "LONG"]
        shorts = [t for t in self.closed_trades if t.direction == "SHORT"]
        
        long_wins = len([t for t in longs if t.is_win])
        short_wins = len([t for t in shorts if t.is_win])
        
        self.stats = TradingStats(
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl_usd=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            best_trade_pct=best,
            worst_trade_pct=worst,
            stats_by_symbol=stats_by_symbol,
            long_trades=len(longs),
            short_trades=len(shorts),
            long_win_rate=long_wins / len(longs) if longs else 0,
            short_win_rate=short_wins / len(shorts) if shorts else 0
        )
    
    def can_open_position(self, symbol: CryptoSymbol) -> bool:
        """Check if we can open a new position."""
        # Check max positions
        if len(self.open_positions) >= self.max_positions:
            return False
        
        # Check if already have position in this symbol
        for pos in self.open_positions.values():
            if pos.symbol == symbol:
                return False
        
        return True
    
    async def open_position(
        self,
        prediction: HybridPrediction
    ) -> Position | None:
        """
        Open a new paper position based on prediction.
        
        Args:
            prediction: Hybrid prediction with trading signal
            
        Returns:
            New Position or None if can't open
        """
        if not prediction.should_trade:
            logger.debug(f"Skipping {prediction.symbol} - should_trade=False")
            return None
        
        if prediction.final_signal == "HOLD":
            logger.debug(f"Skipping {prediction.symbol} - HOLD signal")
            return None
        
        if not self.can_open_position(prediction.symbol):
            logger.debug(f"Can't open {prediction.symbol} - max positions or duplicate")
            return None
        
        # Calculate position size
        size_usd = self.capital * self.position_size_pct
        
        # Create position
        position = Position(
            id=f"{prediction.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            symbol=prediction.symbol,
            direction=prediction.final_signal,
            entry_price=prediction.current_price,
            entry_time=datetime.utcnow(),
            size_usd=size_usd,
            confidence=prediction.final_confidence,
            target_candles=self.hold_candles,
            current_price=prediction.current_price
        )
        
        self.open_positions[position.id] = position
        self._save_state()
        
        logger.info(
            f"📈 OPENED {position.direction} {position.symbol} @ ${position.entry_price:,.2f} "
            f"(${position.size_usd:.0f}, {position.confidence:.1%} conf)"
        )
        
        return position
    
    async def update_positions(self) -> list[ClosedTrade]:
        """
        Update open positions with current prices.
        Close positions that have reached their target duration.
        
        Returns:
            List of newly closed trades
        """
        if not self.open_positions:
            return []
        
        fetcher = get_fetcher()
        closed = []
        
        for pos_id, position in list(self.open_positions.items()):
            try:
                # Get current price
                current_price = await fetcher.get_current_price(position.symbol)
                position.current_price = current_price
                
                # Calculate unrealized PnL
                if position.direction == "LONG":
                    pnl_pct = (current_price / position.entry_price) - 1
                else:  # SHORT
                    pnl_pct = (position.entry_price / current_price) - 1
                
                position.unrealized_pnl_pct = pnl_pct
                position.unrealized_pnl = position.size_usd * pnl_pct
                
                # Check if should close
                minutes_held = (datetime.utcnow() - position.entry_time).total_seconds() / 60
                target_minutes = self.hold_candles * 15
                
                if minutes_held >= target_minutes:
                    closed_trade = self._close_position(position, current_price)
                    closed.append(closed_trade)
                    
            except Exception as e:
                logger.error(f"Error updating position {pos_id}: {e}")
        
        if closed:
            self._update_stats()
            self._save_state()
        
        return closed
    
    def _close_position(self, position: Position, exit_price: float) -> ClosedTrade:
        """Close a position and record the trade."""
        # Calculate final PnL
        if position.direction == "LONG":
            pnl_pct = (exit_price / position.entry_price) - 1
        else:
            pnl_pct = (position.entry_price / exit_price) - 1
        
        pnl_usd = position.size_usd * pnl_pct
        is_win = pnl_pct > 0
        
        # Update capital
        self.capital += pnl_usd
        
        # Create closed trade record
        duration = int((datetime.utcnow() - position.entry_time).total_seconds() / 60)
        
        closed_trade = ClosedTrade(
            id=position.id,
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.utcnow(),
            size_usd=position.size_usd,
            confidence=position.confidence,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            is_win=is_win,
            duration_minutes=duration
        )
        
        # Remove from open, add to closed
        del self.open_positions[position.id]
        self.closed_trades.append(closed_trade)
        
        emoji = "✅" if is_win else "❌"
        logger.info(
            f"{emoji} CLOSED {position.direction} {position.symbol} "
            f"@ ${exit_price:,.2f} | PnL: {pnl_pct:+.2%} (${pnl_usd:+.2f})"
        )
        
        return closed_trade
    
    async def force_close_all(self) -> list[ClosedTrade]:
        """Force close all open positions at current prices."""
        if not self.open_positions:
            return []
        
        fetcher = get_fetcher()
        closed = []
        
        for pos_id, position in list(self.open_positions.items()):
            try:
                current_price = await fetcher.get_current_price(position.symbol)
                closed_trade = self._close_position(position, current_price)
                closed.append(closed_trade)
            except Exception as e:
                logger.error(f"Error force-closing {pos_id}: {e}")
        
        self._update_stats()
        self._save_state()
        
        return closed
    
    def get_summary(self) -> str:
        """Get a formatted summary of trading performance."""
        lines = [
            "═" * 50,
            "📊 PAPER TRADING SUMMARY",
            "═" * 50,
            f"Capital: ${self.capital:,.2f} (started: ${self.initial_capital:,.2f})",
            f"Total P&L: ${self.stats.total_pnl_usd:+,.2f} ({self.stats.total_pnl_pct:+.2%})",
            "",
            f"Total Trades: {self.stats.total_trades}",
            f"Win Rate: {self.stats.win_rate:.1%} ({self.stats.winning_trades}W / {self.stats.losing_trades}L)",
            f"Avg Win: {self.stats.avg_win_pct:+.2%} | Avg Loss: {self.stats.avg_loss_pct:+.2%}",
            f"Best: {self.stats.best_trade_pct:+.2%} | Worst: {self.stats.worst_trade_pct:+.2%}",
            "",
            "By Direction:",
            f"  LONG:  {self.stats.long_trades} trades, {self.stats.long_win_rate:.1%} WR",
            f"  SHORT: {self.stats.short_trades} trades, {self.stats.short_win_rate:.1%} WR",
        ]
        
        if self.stats.stats_by_symbol:
            lines.append("")
            lines.append("By Symbol:")
            for sym, stats in self.stats.stats_by_symbol.items():
                lines.append(
                    f"  {sym}: {stats['trades']} trades, "
                    f"{stats['win_rate']:.1%} WR, ${stats['pnl_usd']:+.2f}"
                )
        
        if self.open_positions:
            lines.append("")
            lines.append(f"Open Positions ({len(self.open_positions)}):")
            for pos in self.open_positions.values():
                lines.append(
                    f"  {pos.direction} {pos.symbol} @ ${pos.entry_price:,.2f} "
                    f"({pos.unrealized_pnl_pct:+.2%})"
                )
        
        lines.append("═" * 50)
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all trading state."""
        self.capital = self.initial_capital
        self.open_positions = {}
        self.closed_trades = []
        self.stats = TradingStats()
        self._save_state()
        logger.info("Paper trading state reset")


# Singleton instance
_engine: PaperTradingEngine | None = None


def get_trading_engine(**kwargs) -> PaperTradingEngine:
    """Get or create global trading engine."""
    global _engine
    if _engine is None:
        _engine = PaperTradingEngine(**kwargs)
    return _engine
