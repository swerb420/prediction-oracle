"""
Paper trading engine for simulated trading.
Tracks positions, P&L, and performance metrics.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json

from pydantic import BaseModel, Field

from core.logging_utils import get_logger
from strategy.risk_manager import RiskManager, RiskConfig, PortfolioState

logger = get_logger(__name__)


class PaperTrade(BaseModel):
    """A single paper trade."""
    
    trade_id: str
    timestamp: datetime
    
    # Asset info
    symbol: str
    market_id: str | None = None  # For Polymarket trades
    
    # Trade details
    direction: str  # "long", "short", "buy", "sell"
    entry_price: float
    size_usd: float
    
    # Exit info
    exit_price: float | None = None
    exit_timestamp: datetime | None = None
    is_closed: bool = False
    
    # P&L
    unrealized_pnl_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    
    # Metadata
    strategy: str = "unknown"
    signal_confidence: float = 0.5
    regime: str = "unknown"
    
    # Resolution (for prediction markets)
    predicted_outcome: str | None = None
    actual_outcome: str | None = None
    correct: bool | None = None
    
    def update_pnl(self, current_price: float) -> None:
        """Update unrealized P&L."""
        if self.is_closed:
            return
        
        if self.direction in ("long", "buy"):
            pnl = (current_price - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - current_price) / self.entry_price
        
        self.pnl_pct = pnl
        self.unrealized_pnl_usd = pnl * self.size_usd
    
    def close(
        self,
        exit_price: float,
        exit_timestamp: datetime | None = None,
        actual_outcome: str | None = None
    ) -> None:
        """Close the trade."""
        from core.time_utils import now_utc
        
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp or now_utc()
        self.is_closed = True
        
        if self.direction in ("long", "buy"):
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        self.realized_pnl_usd = self.pnl_pct * self.size_usd
        self.unrealized_pnl_usd = 0.0
        
        if actual_outcome:
            self.actual_outcome = actual_outcome
            self.correct = self.predicted_outcome == actual_outcome


class PaperPortfolio(BaseModel):
    """Paper trading portfolio state."""
    
    # Value
    initial_value_usd: float = 10000.0
    cash_usd: float = 10000.0
    deployed_usd: float = 0.0
    
    # P&L
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    peak_value_usd: float = 10000.0
    
    # Tracking
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    
    # Time tracking
    start_time: datetime | None = None
    last_update: datetime | None = None
    
    @property
    def total_value(self) -> float:
        return self.cash_usd + self.deployed_usd + self.total_unrealized_pnl
    
    @property
    def total_pnl(self) -> float:
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    @property
    def total_return_pct(self) -> float:
        if self.initial_value_usd == 0:
            return 0.0
        return (self.total_value - self.initial_value_usd) / self.initial_value_usd
    
    @property
    def win_rate(self) -> float:
        total = self.n_wins + self.n_losses
        if total == 0:
            return 0.0
        return self.n_wins / total
    
    @property
    def current_drawdown(self) -> float:
        if self.peak_value_usd == 0:
            return 0.0
        return (self.peak_value_usd - self.total_value) / self.peak_value_usd


class PaperTrader:
    """
    Paper trading engine.
    
    Features:
    - Simulated order execution
    - Position tracking
    - P&L calculation
    - Performance metrics
    - Trade persistence
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_config: RiskConfig | None = None,
        trades_file: Path | None = None
    ):
        self.portfolio = PaperPortfolio(
            initial_value_usd=initial_capital,
            cash_usd=initial_capital,
            peak_value_usd=initial_capital
        )
        
        self.risk_manager = RiskManager(risk_config)
        self.trades_file = trades_file
        
        self._open_trades: dict[str, PaperTrade] = {}
        self._closed_trades: list[PaperTrade] = []
        self._trade_counter: int = 0
        
        # Initialize portfolio state for risk manager
        self._update_risk_state()
        
        # Load existing trades if file exists
        if trades_file and trades_file.exists():
            self._load_trades()
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"trade_{self._trade_counter:06d}"
    
    def _update_risk_state(self) -> None:
        """Update risk manager with current portfolio state."""
        from core.time_utils import now_utc
        
        # Calculate positions by asset
        positions_by_asset: dict[str, float] = {}
        for trade in self._open_trades.values():
            positions_by_asset[trade.symbol] = positions_by_asset.get(trade.symbol, 0.0) + trade.size_usd
        
        state = PortfolioState(
            timestamp=now_utc(),
            total_value_usd=self.portfolio.total_value,
            cash_usd=self.portfolio.cash_usd,
            deployed_usd=self.portfolio.deployed_usd,
            daily_pnl_usd=0.0,  # Would need daily tracking
            weekly_pnl_usd=0.0,
            total_pnl_usd=self.portfolio.total_pnl,
            peak_value_usd=self.portfolio.peak_value_usd,
            n_open_positions=len(self._open_trades),
            positions_by_asset=positions_by_asset,
            trades_today=self.portfolio.n_trades  # Simplified
        )
        
        self.risk_manager.set_portfolio_state(state)
    
    def open_trade(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        entry_price: float,
        strategy: str = "unknown",
        confidence: float = 0.5,
        regime: str = "unknown",
        market_id: str | None = None,
        predicted_outcome: str | None = None
    ) -> PaperTrade | None:
        """
        Open a new paper trade.
        
        Args:
            symbol: Asset symbol
            direction: "long" or "short"
            size_usd: Position size in USD
            entry_price: Entry price
            strategy: Strategy name
            confidence: Signal confidence
            regime: Market regime
            market_id: Polymarket market ID (optional)
            predicted_outcome: Predicted outcome for prediction markets
            
        Returns:
            PaperTrade or None if risk checks fail
        """
        from core.time_utils import now_utc
        
        # Run risk checks
        checks = self.risk_manager.check_trade(symbol, direction, size_usd, confidence)
        failed_checks = [c for c in checks if not c.passed]
        
        if failed_checks:
            for check in failed_checks:
                logger.warning(f"Risk check failed: {check.check_name} - {check.message}")
            return None
        
        # Check sufficient capital
        if size_usd > self.portfolio.cash_usd:
            logger.warning(f"Insufficient capital: need ${size_usd:.2f}, have ${self.portfolio.cash_usd:.2f}")
            return None
        
        # Create trade
        trade = PaperTrade(
            trade_id=self._generate_trade_id(),
            timestamp=now_utc(),
            symbol=symbol,
            market_id=market_id,
            direction=direction,
            entry_price=entry_price,
            size_usd=size_usd,
            strategy=strategy,
            signal_confidence=confidence,
            regime=regime,
            predicted_outcome=predicted_outcome
        )
        
        # Update portfolio
        self.portfolio.cash_usd -= size_usd
        self.portfolio.deployed_usd += size_usd
        self.portfolio.n_trades += 1
        
        if self.portfolio.start_time is None:
            self.portfolio.start_time = now_utc()
        
        # Store trade
        self._open_trades[trade.trade_id] = trade
        
        # Update risk state
        self._update_risk_state()
        
        logger.info(
            f"Opened {direction} {symbol} @ ${entry_price:.2f}, "
            f"size=${size_usd:.2f}, confidence={confidence:.2f}"
        )
        
        return trade
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        actual_outcome: str | None = None
    ) -> PaperTrade | None:
        """
        Close an open trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            actual_outcome: Actual outcome for prediction markets
            
        Returns:
            Closed PaperTrade or None
        """
        if trade_id not in self._open_trades:
            logger.warning(f"Trade not found: {trade_id}")
            return None
        
        trade = self._open_trades.pop(trade_id)
        trade.close(exit_price, actual_outcome=actual_outcome)
        
        # Update portfolio
        self.portfolio.deployed_usd -= trade.size_usd
        self.portfolio.cash_usd += trade.size_usd + trade.realized_pnl_usd
        self.portfolio.total_realized_pnl += trade.realized_pnl_usd
        
        if trade.realized_pnl_usd >= 0:
            self.portfolio.n_wins += 1
        else:
            self.portfolio.n_losses += 1
        
        # Update peak
        if self.portfolio.total_value > self.portfolio.peak_value_usd:
            self.portfolio.peak_value_usd = self.portfolio.total_value
        
        # Store closed trade
        self._closed_trades.append(trade)
        
        # Update risk state
        self._update_risk_state()
        
        logger.info(
            f"Closed {trade.symbol} @ ${exit_price:.2f}, "
            f"P&L: ${trade.realized_pnl_usd:+.2f} ({trade.pnl_pct:+.1%})"
        )
        
        # Persist trades
        if self.trades_file:
            self._save_trades()
        
        return trade
    
    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update open positions with current prices.
        
        Args:
            prices: Dict of symbol -> current price
        """
        total_unrealized = 0.0
        
        for trade in self._open_trades.values():
            if trade.symbol in prices:
                trade.update_pnl(prices[trade.symbol])
                total_unrealized += trade.unrealized_pnl_usd
        
        self.portfolio.total_unrealized_pnl = total_unrealized
        
        # Update peak
        if self.portfolio.total_value > self.portfolio.peak_value_usd:
            self.portfolio.peak_value_usd = self.portfolio.total_value
        
        from core.time_utils import now_utc
        self.portfolio.last_update = now_utc()
    
    def get_open_trades(self) -> list[PaperTrade]:
        """Get all open trades."""
        return list(self._open_trades.values())
    
    def get_closed_trades(self) -> list[PaperTrade]:
        """Get all closed trades."""
        return list(self._closed_trades)
    
    def get_trades_by_symbol(self, symbol: str) -> list[PaperTrade]:
        """Get all trades for a symbol."""
        trades = []
        trades.extend(t for t in self._open_trades.values() if t.symbol == symbol)
        trades.extend(t for t in self._closed_trades if t.symbol == symbol)
        return trades
    
    def get_stats(self) -> dict[str, Any]:
        """Get trading statistics."""
        closed = self._closed_trades
        
        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0
            }
        
        wins = [t for t in closed if t.realized_pnl_usd >= 0]
        losses = [t for t in closed if t.realized_pnl_usd < 0]
        
        avg_win = sum(t.realized_pnl_usd for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.realized_pnl_usd for t in losses) / len(losses) if losses else 0
        
        return {
            "total_trades": len(closed),
            "open_trades": len(self._open_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed),
            "total_realized_pnl": self.portfolio.total_realized_pnl,
            "total_unrealized_pnl": self.portfolio.total_unrealized_pnl,
            "total_pnl": self.portfolio.total_pnl,
            "total_return_pct": self.portfolio.total_return_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            "current_drawdown": self.portfolio.current_drawdown,
            "portfolio_value": self.portfolio.total_value,
            "cash": self.portfolio.cash_usd,
            "deployed": self.portfolio.deployed_usd
        }
    
    def print_summary(self) -> None:
        """Print trading summary."""
        stats = self.get_stats()
        
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        table = Table(title="Paper Trading Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Trades", str(stats["total_trades"]))
        table.add_row("Open Positions", str(stats["open_trades"]))
        table.add_row("Win Rate", f"{stats['win_rate']:.1%}")
        table.add_row("Total P&L", f"${stats['total_pnl']:+,.2f}")
        table.add_row("Total Return", f"{stats['total_return_pct']:+.2%}")
        table.add_row("Profit Factor", f"{stats['profit_factor']:.2f}")
        table.add_row("Current Drawdown", f"{stats['current_drawdown']:.2%}")
        table.add_row("Portfolio Value", f"${stats['portfolio_value']:,.2f}")
        
        console.print(table)
    
    def _save_trades(self) -> None:
        """Save trades to file."""
        if not self.trades_file:
            return
        
        self.trades_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "portfolio": self.portfolio.model_dump(),
            "open_trades": [t.model_dump() for t in self._open_trades.values()],
            "closed_trades": [t.model_dump() for t in self._closed_trades],
            "trade_counter": self._trade_counter
        }
        
        # Convert datetime to string
        def convert_dates(obj):
            if isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(i) for i in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(self.trades_file, "w") as f:
            json.dump(convert_dates(data), f, indent=2)
    
    def _load_trades(self) -> None:
        """Load trades from file."""
        if not self.trades_file or not self.trades_file.exists():
            return
        
        try:
            with open(self.trades_file) as f:
                data = json.load(f)
            
            # Convert string dates back to datetime
            def parse_dates(obj):
                if isinstance(obj, dict):
                    return {k: parse_dates(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [parse_dates(i) for i in obj]
                elif isinstance(obj, str):
                    try:
                        return datetime.fromisoformat(obj)
                    except (ValueError, TypeError):
                        return obj
                return obj
            
            data = parse_dates(data)
            
            self.portfolio = PaperPortfolio(**data["portfolio"])
            self._trade_counter = data.get("trade_counter", 0)
            
            for trade_data in data.get("open_trades", []):
                trade = PaperTrade(**trade_data)
                self._open_trades[trade.trade_id] = trade
            
            for trade_data in data.get("closed_trades", []):
                self._closed_trades.append(PaperTrade(**trade_data))
            
            logger.info(f"Loaded {len(self._open_trades)} open, {len(self._closed_trades)} closed trades")
            
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
