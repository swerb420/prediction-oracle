"""
Paper Trading Engine - Track virtual positions and PnL.

Features:
1. Track open positions per symbol
2. Calculate PnL on resolution
3. Portfolio-level risk management
4. Full audit trail of all trades
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Literal
from dataclasses import dataclass, field

from real_data_store import get_store, RealDataStore, PaperTrade
from trading_logger import get_logger, TradingLogger

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


@dataclass
class Position:
    """An open position."""
    trade_id: int
    symbol: str
    direction: str
    entry_price: float
    size_usd: float
    confidence: float
    grok_used: bool
    opened_at: str
    event_slug: Optional[str] = None


@dataclass  
class Portfolio:
    """Portfolio state."""
    starting_capital: float = 1000.0
    current_capital: float = 1000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        return (self.current_capital - self.starting_capital) / self.starting_capital


class PaperTradingEngine:
    """
    Paper trading engine for crypto 15M markets.
    
    Tracks:
    - Open positions
    - Portfolio value
    - PnL (realized and unrealized)
    - Win/loss statistics
    
    Usage:
        engine = PaperTradingEngine()
        
        # Enter a position
        trade_id = engine.enter_position("BTC", "UP", 0.55, 100.0, confidence=0.72)
        
        # Close on resolution
        pnl = engine.close_position(trade_id, actual_outcome="UP")
        
        # Get portfolio state
        portfolio = engine.get_portfolio()
    """
    
    def __init__(
        self,
        store: Optional[RealDataStore] = None,
        log: Optional[TradingLogger] = None,
        starting_capital: float = 1000.0,
    ):
        self.store = store or get_store()
        self.log = log or get_logger()
        
        # Portfolio
        self.portfolio = Portfolio(
            starting_capital=starting_capital,
            current_capital=starting_capital,
        )
        
        # Open positions by trade_id
        self.positions: dict[int, Position] = {}
        
        # Load existing open trades
        self._load_open_trades()
    
    def _load_open_trades(self):
        """Load open trades from database."""
        open_trades = self.store.get_open_trades()
        
        for trade in open_trades:
            self.positions[trade["id"]] = Position(
                trade_id=trade["id"],
                symbol=trade["symbol"],
                direction=trade["direction"],
                entry_price=trade["entry_price"],
                size_usd=trade["size_usd"],
                confidence=trade["confidence"] or 0.5,
                grok_used=bool(trade["grok_used"]),
                opened_at=trade["timestamp"],
                event_slug=trade.get("event_slug"),
            )
        
        if self.positions:
            self.log.system(
                f"Loaded {len(self.positions)} open positions",
                {"trade_ids": list(self.positions.keys())}
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Position Management
    # ─────────────────────────────────────────────────────────────────────────
    
    def enter_position(
        self,
        symbol: CryptoSymbol,
        direction: str,
        entry_price: float,
        size_usd: float,
        confidence: float = 0.5,
        ml_confidence: float = 0.5,
        grok_used: bool = False,
        grok_agreed: bool = False,
        event_slug: Optional[str] = None,
    ) -> int:
        """
        Enter a new position.
        
        Returns trade_id.
        """
        # Check if we have capital
        if size_usd > self.portfolio.current_capital:
            self.log.warning(
                f"Insufficient capital: ${size_usd:.2f} > ${self.portfolio.current_capital:.2f}",
                symbol=symbol
            )
            size_usd = self.portfolio.current_capital * 0.9
        
        # Create trade record
        trade = PaperTrade(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction=direction,
            entry_price=entry_price,
            size_usd=size_usd,
            confidence=confidence,
            ml_confidence=ml_confidence,
            grok_used=grok_used,
            grok_agreed=grok_agreed,
        )
        
        # Add event_slug if available
        if event_slug:
            trade.event_slug = event_slug
        
        # Save to database
        trade_id = self.store.save_trade(trade)
        
        # Track position
        self.positions[trade_id] = Position(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size_usd=size_usd,
            confidence=confidence,
            grok_used=grok_used,
            opened_at=trade.timestamp,
            event_slug=event_slug,
        )
        
        # Update capital (lock up the position size)
        self.portfolio.current_capital -= size_usd
        
        # Log
        self.log.log_trade_entry(symbol, direction, entry_price, size_usd, trade_id)
        
        return trade_id
    
    def close_position(
        self,
        trade_id: int,
        actual_outcome: str,
        exit_price: Optional[float] = None,
    ) -> float:
        """
        Close a position and calculate PnL.
        
        Returns PnL.
        """
        if trade_id not in self.positions:
            self.log.warning(f"Trade {trade_id} not found in open positions")
            return 0.0
        
        position = self.positions[trade_id]
        
        # Calculate exit price (1.0 if won, 0.0 if lost)
        was_correct = (position.direction == actual_outcome)
        
        if exit_price is None:
            exit_price = 1.0 if was_correct else 0.0
        
        # Calculate PnL
        if was_correct:
            # Bought at entry_price, sold at 1.0
            # Profit = size * (1.0 - entry_price) / entry_price
            pnl = position.size_usd * (1.0 - position.entry_price) / position.entry_price
        else:
            # Lost the position
            pnl = -position.size_usd
        
        # Update database
        self.store.close_trade(trade_id, exit_price, actual_outcome)
        
        # Update portfolio
        self.portfolio.current_capital += position.size_usd + pnl
        self.portfolio.realized_pnl += pnl
        self.portfolio.total_trades += 1
        
        if was_correct:
            self.portfolio.wins += 1
        else:
            self.portfolio.losses += 1
        
        # Remove from open positions
        del self.positions[trade_id]
        
        # Log
        self.log.log_trade_exit(position.symbol, trade_id, exit_price, pnl, was_correct)
        
        return pnl
    
    def close_all_for_symbol(self, symbol: str, actual_outcome: str) -> float:
        """Close all positions for a symbol."""
        total_pnl = 0.0
        
        # Find matching positions
        trade_ids = [
            tid for tid, pos in self.positions.items()
            if pos.symbol == symbol
        ]
        
        for trade_id in trade_ids:
            pnl = self.close_position(trade_id, actual_outcome)
            total_pnl += pnl
        
        return total_pnl
    
    def close_by_event_slug(self, event_slug: str, actual_outcome: str) -> float:
        """Close positions for a specific event."""
        total_pnl = 0.0
        
        trade_ids = [
            tid for tid, pos in self.positions.items()
            if pos.event_slug == event_slug
        ]
        
        for trade_id in trade_ids:
            pnl = self.close_position(trade_id, actual_outcome)
            total_pnl += pnl
        
        return total_pnl
    
    # ─────────────────────────────────────────────────────────────────────────
    # Position Sizing
    # ─────────────────────────────────────────────────────────────────────────
    
    def calculate_position_size(
        self,
        position_pct: float,
        max_risk_usd: Optional[float] = None,
    ) -> float:
        """Calculate position size in USD."""
        size = self.portfolio.current_capital * position_pct
        
        if max_risk_usd:
            size = min(size, max_risk_usd)
        
        # Never bet more than we have
        size = min(size, self.portfolio.current_capital * 0.95)
        
        return max(size, 0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Portfolio Info
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state."""
        # Calculate unrealized PnL
        # For now, just use 0 since we don't have real-time prices
        self.portfolio.unrealized_pnl = 0.0
        
        return self.portfolio
    
    def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get open position for a symbol."""
        for pos in self.positions.values():
            if pos.symbol == symbol:
                return pos
        return None
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for a symbol."""
        return self.get_position_by_symbol(symbol) is not None
    
    def get_stats(self) -> dict:
        """Get trading statistics."""
        portfolio = self.get_portfolio()
        
        return {
            "capital": portfolio.current_capital,
            "starting_capital": portfolio.starting_capital,
            "realized_pnl": portfolio.realized_pnl,
            "return_pct": portfolio.return_pct,
            "total_trades": portfolio.total_trades,
            "wins": portfolio.wins,
            "losses": portfolio.losses,
            "win_rate": portfolio.win_rate,
            "open_positions": len(self.positions),
        }
    
    def print_summary(self):
        """Print portfolio summary."""
        stats = self.get_stats()
        portfolio = self.get_portfolio()
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Capital: ${portfolio.current_capital:.2f} / ${portfolio.starting_capital:.2f}")
        print(f"Return: {portfolio.return_pct:+.1%}")
        print(f"Realized PnL: ${portfolio.realized_pnl:+.2f}")
        print(f"Total Trades: {portfolio.total_trades}")
        print(f"Win Rate: {portfolio.win_rate:.1%} ({portfolio.wins}W / {portfolio.losses}L)")
        
        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for pos in self.positions.values():
                print(f"  {pos.symbol}: {pos.direction} @ {pos.entry_price:.3f} (${pos.size_usd:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = PaperTradingEngine(starting_capital=1000.0)
    
    # Enter some test positions
    print("Entering test positions...")
    
    t1 = engine.enter_position(
        "BTC", "UP", 0.55, 100.0, 
        confidence=0.72, grok_used=True
    )
    
    t2 = engine.enter_position(
        "ETH", "DOWN", 0.60, 75.0,
        confidence=0.65, grok_used=False
    )
    
    engine.print_summary()
    
    # Close positions
    print("\nClosing positions...")
    
    engine.close_position(t1, "UP")  # Win
    engine.close_position(t2, "UP")  # Loss
    
    engine.print_summary()
    
    print(f"\nStats: {engine.get_stats()}")
