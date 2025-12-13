"""
Backtesting engine for strategy evaluation.
Replays historical data to evaluate strategy performance.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging_utils import get_logger
from exec.paper_trader import PaperTrader, PaperTrade

logger = get_logger(__name__)


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    
    # Time range
    start_date: datetime
    end_date: datetime
    
    # Capital
    initial_capital: float = 10000.0
    
    # Costs
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.001  # 0.1% slippage
    
    # Execution
    fill_probability: float = 1.0  # Probability of order fill
    
    # Strategy params
    max_positions: int = 10
    position_sizing: str = "kelly"  # "kelly", "equal", "risk_parity"
    
    # Output
    save_trades: bool = True
    output_dir: Path = Path("backtest_results")


class BacktestResult(BaseModel):
    """Results from a backtest run."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Time info
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    
    # Trade stats
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    
    # Holding periods
    avg_holding_period_hours: float
    max_holding_period_hours: float
    
    # Time series
    equity_curve: list[float] = Field(default_factory=list)
    drawdown_curve: list[float] = Field(default_factory=list)
    daily_returns: list[float] = Field(default_factory=list)
    
    # Per-symbol breakdown
    returns_by_symbol: dict[str, float] = Field(default_factory=dict)
    trades_by_symbol: dict[str, int] = Field(default_factory=dict)
    
    @property
    def is_profitable(self) -> bool:
        return self.total_return_pct > 0
    
    @property
    def is_good(self) -> bool:
        """Check if backtest shows promising results."""
        return (
            self.sharpe_ratio > 1.0 and
            self.win_rate > 0.50 and
            self.max_drawdown_pct < 0.20
        )


class Backtester:
    """
    Backtesting engine for strategy evaluation.
    
    Features:
    - Event-driven simulation
    - Realistic execution modeling (slippage, fills)
    - Risk metrics calculation
    - Trade-by-trade analysis
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self._trader: PaperTrader | None = None
        self._equity_curve: list[tuple[datetime, float]] = []
        self._trades_log: list[dict[str, Any]] = []
    
    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, int, PaperTrader], list[dict[str, Any]]],
        price_column: str = "close"
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data and features
            strategy_func: Function that generates signals given data and current index
            price_column: Column to use for prices
            
        Returns:
            BacktestResult
        """
        logger.info(
            f"Starting backtest from {self.config.start_date} to {self.config.end_date}"
        )
        
        # Initialize trader
        self._trader = PaperTrader(
            initial_capital=self.config.initial_capital,
            trades_file=self.config.output_dir / "trades.json" if self.config.save_trades else None
        )
        
        # Filter data to date range
        data = data.copy()
        if "timestamp" in data.columns:
            data = data[(data["timestamp"] >= self.config.start_date) & 
                       (data["timestamp"] <= self.config.end_date)]
        
        if len(data) == 0:
            raise ValueError("No data in specified date range")
        
        # Main simulation loop
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = row.get("timestamp", datetime.now())
            
            # Get current prices for open positions
            current_prices = self._get_current_prices(data, i, price_column)
            self._trader.update_prices(current_prices)
            
            # Record equity
            self._equity_curve.append((timestamp, self._trader.portfolio.total_value))
            
            # Generate signals from strategy
            signals = strategy_func(data, i, self._trader)
            
            # Execute signals
            for signal in signals:
                self._execute_signal(signal, current_prices, timestamp)
        
        # Close any remaining positions at end
        final_prices = self._get_current_prices(data, len(data) - 1, price_column)
        for trade_id in list(self._trader._open_trades.keys()):
            trade = self._trader._open_trades[trade_id]
            exit_price = final_prices.get(trade.symbol, trade.entry_price)
            self._trader.close_trade(trade_id, exit_price)
        
        # Calculate results
        result = self._calculate_results(data)
        
        logger.info(
            f"Backtest complete: {result.total_return_pct:+.1%} return, "
            f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown_pct:.1%}"
        )
        
        return result
    
    def _get_current_prices(
        self,
        data: pd.DataFrame,
        index: int,
        price_column: str
    ) -> dict[str, float]:
        """Extract current prices for all symbols."""
        prices = {}
        
        row = data.iloc[index]
        
        # Check if we have multi-symbol data
        if "symbol" in data.columns:
            # Group by symbol and get latest price
            symbols = data["symbol"].unique()
            for symbol in symbols:
                symbol_data = data[data["symbol"] == symbol]
                if index < len(symbol_data):
                    prices[symbol] = float(symbol_data.iloc[min(index, len(symbol_data) - 1)][price_column])
        else:
            # Single symbol
            if price_column in row:
                prices["default"] = float(row[price_column])
        
        return prices
    
    def _execute_signal(
        self,
        signal: dict[str, Any],
        prices: dict[str, float],
        timestamp: datetime
    ) -> None:
        """Execute a trading signal."""
        action = signal.get("action", "hold")
        
        if action == "hold":
            return
        
        symbol = signal.get("symbol", "default")
        
        if action in ("buy", "long"):
            # Open long position
            size = signal.get("size_usd", 100.0)
            entry_price = prices.get(symbol, signal.get("price", 100.0))
            
            # Apply slippage
            entry_price *= (1 + self.config.slippage_pct)
            
            # Apply commission
            size *= (1 - self.config.commission_pct)
            
            # Check fill probability
            if np.random.random() > self.config.fill_probability:
                return
            
            self._trader.open_trade(
                symbol=symbol,
                direction="long",
                size_usd=size,
                entry_price=entry_price,
                strategy=signal.get("strategy", "backtest"),
                confidence=signal.get("confidence", 0.5)
            )
            
        elif action in ("sell", "short", "close"):
            # Close position
            trade_id = signal.get("trade_id")
            
            if trade_id:
                trade = self._trader._open_trades.get(trade_id)
                if trade:
                    exit_price = prices.get(symbol, trade.entry_price)
                    exit_price *= (1 - self.config.slippage_pct)  # Slippage against us
                    self._trader.close_trade(trade_id, exit_price)
            else:
                # Close all positions for symbol
                for tid, trade in list(self._trader._open_trades.items()):
                    if trade.symbol == symbol:
                        exit_price = prices.get(symbol, trade.entry_price)
                        exit_price *= (1 - self.config.slippage_pct)
                        self._trader.close_trade(tid, exit_price)
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate backtest results and metrics."""
        # Extract equity curve
        timestamps = [t for t, _ in self._equity_curve]
        equity = [e for _, e in self._equity_curve]
        
        if len(equity) < 2:
            return self._empty_result()
        
        # Calculate returns
        equity_arr = np.array(equity)
        returns = np.diff(equity_arr) / equity_arr[:-1]
        returns = np.nan_to_num(returns, 0.0)
        
        # Daily returns (assuming 15min data = 96 bars per day)
        bars_per_day = 96
        daily_returns = []
        for i in range(0, len(returns), bars_per_day):
            chunk = returns[i:i + bars_per_day]
            if len(chunk) > 0:
                daily_ret = (1 + chunk).prod() - 1
                daily_returns.append(daily_ret)
        
        daily_returns = np.array(daily_returns) if daily_returns else np.array([0.0])
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Annualized return
        duration_days = max(1, (self.config.end_date - self.config.start_date).days)
        annualized_return = ((1 + total_return) ** (365 / duration_days)) - 1
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe = 0.0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = np.sqrt(252) * daily_returns.mean() / downside_returns.std()
        else:
            sortino = sharpe
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown = float(np.max(drawdown))
        
        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Trade statistics
        closed_trades = self._trader.get_closed_trades()
        n_trades = len(closed_trades)
        
        if n_trades > 0:
            wins = [t for t in closed_trades if t.realized_pnl_usd >= 0]
            losses = [t for t in closed_trades if t.realized_pnl_usd < 0]
            
            win_rate = len(wins) / n_trades
            
            avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
            avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
            avg_trade = np.mean([t.pnl_pct for t in closed_trades])
            
            total_wins = sum(t.realized_pnl_usd for t in wins)
            total_losses = abs(sum(t.realized_pnl_usd for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Holding periods
            holding_periods = []
            for t in closed_trades:
                if t.exit_timestamp and t.timestamp:
                    hours = (t.exit_timestamp - t.timestamp).total_seconds() / 3600
                    holding_periods.append(hours)
            
            avg_holding = np.mean(holding_periods) if holding_periods else 0.0
            max_holding = np.max(holding_periods) if holding_periods else 0.0
            
            # Per-symbol breakdown
            returns_by_symbol = {}
            trades_by_symbol = {}
            for t in closed_trades:
                if t.symbol not in returns_by_symbol:
                    returns_by_symbol[t.symbol] = 0.0
                    trades_by_symbol[t.symbol] = 0
                returns_by_symbol[t.symbol] += t.realized_pnl_usd
                trades_by_symbol[t.symbol] += 1
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            avg_trade = 0.0
            profit_factor = 0.0
            avg_holding = 0.0
            max_holding = 0.0
            returns_by_symbol = {}
            trades_by_symbol = {}
        
        return BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration_days=duration_days,
            total_return_pct=total_return,
            annualized_return_pct=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown,
            calmar_ratio=calmar,
            n_trades=n_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return_pct=avg_trade,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            avg_holding_period_hours=avg_holding,
            max_holding_period_hours=max_holding,
            equity_curve=equity,
            drawdown_curve=drawdown.tolist(),
            daily_returns=daily_returns.tolist(),
            returns_by_symbol=returns_by_symbol,
            trades_by_symbol=trades_by_symbol
        )
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result for failed backtests."""
        return BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration_days=0,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
            n_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_return_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            avg_holding_period_hours=0.0,
            max_holding_period_hours=0.0
        )
    
    def print_summary(self, result: BacktestResult) -> None:
        """Print backtest summary."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Period", f"{result.start_date.date()} to {result.end_date.date()}")
        table.add_row("Duration", f"{result.duration_days} days")
        table.add_row("", "")
        table.add_row("Total Return", f"{result.total_return_pct:+.2%}")
        table.add_row("Annualized Return", f"{result.annualized_return_pct:+.2%}")
        table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        table.add_row("Sortino Ratio", f"{result.sortino_ratio:.2f}")
        table.add_row("Max Drawdown", f"{result.max_drawdown_pct:.2%}")
        table.add_row("Calmar Ratio", f"{result.calmar_ratio:.2f}")
        table.add_row("", "")
        table.add_row("Total Trades", str(result.n_trades))
        table.add_row("Win Rate", f"{result.win_rate:.1%}")
        table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
        table.add_row("Avg Trade", f"{result.avg_trade_return_pct:+.2%}")
        table.add_row("Avg Holding", f"{result.avg_holding_period_hours:.1f}h")
        
        console.print(table)
        
        # Status
        if result.is_good:
            console.print("[green]✓ Strategy shows promising results[/green]")
        elif result.is_profitable:
            console.print("[yellow]○ Strategy is profitable but needs improvement[/yellow]")
        else:
            console.print("[red]✗ Strategy is not profitable[/red]")
