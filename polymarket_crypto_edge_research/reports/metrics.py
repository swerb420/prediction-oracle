"""
Performance metrics calculations for strategy evaluation.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


def calculate_returns(
    equity_curve: list[float] | np.ndarray,
    log_returns: bool = False
) -> np.ndarray:
    """
    Calculate returns from equity curve.
    
    Args:
        equity_curve: List of portfolio values over time
        log_returns: If True, calculate log returns
        
    Returns:
        Array of returns
    """
    equity = np.array(equity_curve)
    
    if len(equity) < 2:
        return np.array([])
    
    if log_returns:
        returns = np.diff(np.log(equity))
    else:
        returns = np.diff(equity) / equity[:-1]
    
    return returns


def calculate_sharpe_ratio(
    returns: list[float] | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Annualized Sharpe ratio
    """
    returns = np.array(returns)
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return float(sharpe * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: list[float] | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio.
    Uses downside deviation instead of standard deviation.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Annualized Sortino ratio
    """
    returns = np.array(returns)
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Calculate downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return float("inf") if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(negative_returns ** 2))
    
    if downside_std == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / downside_std
    return float(sortino * np.sqrt(periods_per_year))


def calculate_max_drawdown(
    equity_curve: list[float] | np.ndarray
) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: List of portfolio values over time
        
    Returns:
        Tuple of (max_drawdown_pct, peak_index, trough_index)
    """
    equity = np.array(equity_curve)
    
    if len(equity) < 2:
        return 0.0, 0, 0
    
    # Running maximum
    running_max = np.maximum.accumulate(equity)
    
    # Drawdown at each point
    drawdowns = (running_max - equity) / running_max
    
    # Maximum drawdown
    max_dd_idx = np.argmax(drawdowns)
    max_dd = float(drawdowns[max_dd_idx])
    
    # Find peak (running max at that point)
    peak_idx = int(np.argmax(equity[:max_dd_idx + 1]))
    
    return max_dd, peak_idx, max_dd_idx


def calculate_calmar_ratio(
    equity_curve: list[float] | np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        equity_curve: List of portfolio values
        periods_per_year: Trading periods per year
        
    Returns:
        Calmar ratio
    """
    equity = np.array(equity_curve)
    
    if len(equity) < 2:
        return 0.0
    
    # Annualized return
    total_return = equity[-1] / equity[0] - 1
    n_periods = len(equity) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(equity)
    
    if max_dd == 0:
        return float("inf") if annualized_return > 0 else 0.0
    
    return float(annualized_return / max_dd)


def calculate_win_rate(
    trades: list[dict[str, Any]]
) -> float:
    """
    Calculate win rate from list of trades.
    
    Args:
        trades: List of trade dicts with 'pnl' field
        
    Returns:
        Win rate as decimal (0-1)
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return winning_trades / len(trades)


def calculate_profit_factor(
    trades: list[dict[str, Any]]
) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dicts with 'pnl' field
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
    
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expectancy(
    trades: list[dict[str, Any]]
) -> float:
    """
    Calculate expectancy (expected profit per trade).
    
    Args:
        trades: List of trade dicts with 'pnl' field
        
    Returns:
        Average P&L per trade
    """
    if not trades:
        return 0.0
    
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    return total_pnl / len(trades)


def calculate_risk_reward_ratio(
    trades: list[dict[str, Any]]
) -> float:
    """
    Calculate average risk/reward ratio.
    
    Args:
        trades: List of trade dicts with 'pnl' field
        
    Returns:
        Average winner size / average loser size
    """
    winning_trades = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [abs(t.get("pnl", 0)) for t in trades if t.get("pnl", 0) < 0]
    
    if not winning_trades or not losing_trades:
        return 0.0
    
    avg_win = sum(winning_trades) / len(winning_trades)
    avg_loss = sum(losing_trades) / len(losing_trades)
    
    if avg_loss == 0:
        return float("inf")
    
    return avg_win / avg_loss


def calculate_consecutive_stats(
    trades: list[dict[str, Any]]
) -> dict[str, int]:
    """
    Calculate consecutive wins/losses statistics.
    
    Args:
        trades: List of trade dicts with 'pnl' field
        
    Returns:
        Dict with max_consecutive_wins, max_consecutive_losses, current_streak
    """
    if not trades:
        return {
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "current_streak": 0
        }
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for trade in trades:
        pnl = trade.get("pnl", 0)
        
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
    
    # Current streak
    current_streak = current_wins if current_wins > 0 else -current_losses
    
    return {
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "current_streak": current_streak
    }


def calculate_monthly_returns(
    equity_curve: list[float] | np.ndarray,
    dates: list[Any]
) -> dict[str, float]:
    """
    Calculate monthly returns.
    
    Args:
        equity_curve: Portfolio values
        dates: Corresponding dates
        
    Returns:
        Dict mapping "YYYY-MM" to return
    """
    if len(equity_curve) != len(dates):
        return {}
    
    monthly_returns = {}
    month_start_value = equity_curve[0]
    current_month = None
    
    for i, (value, date) in enumerate(zip(equity_curve, dates)):
        month_key = date.strftime("%Y-%m") if hasattr(date, "strftime") else str(date)[:7]
        
        if current_month is None:
            current_month = month_key
            month_start_value = value
        elif month_key != current_month:
            # End of previous month
            if month_start_value > 0:
                monthly_returns[current_month] = (equity_curve[i-1] / month_start_value) - 1
            current_month = month_key
            month_start_value = value
    
    # Last month
    if current_month and month_start_value > 0:
        monthly_returns[current_month] = (equity_curve[-1] / month_start_value) - 1
    
    return monthly_returns


class StrategyMetrics(BaseModel):
    """Comprehensive strategy performance metrics."""
    
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    total_pnl: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Per-trade stats
    avg_trade_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Time metrics
    avg_trade_duration_hours: float = 0.0
    total_time_in_market_hours: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Strategy Performance Summary
============================
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Total P&L: ${self.total_pnl:,.2f}

Risk Metrics:
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Sortino Ratio: {self.sortino_ratio:.2f}
  Calmar Ratio: {self.calmar_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2%}
  Volatility: {self.volatility:.2%}

Trade Metrics:
  Total Trades: {self.total_trades}
  Win Rate: {self.win_rate:.2%}
  Profit Factor: {self.profit_factor:.2f}
  Expectancy: ${self.expectancy:.2f}
  Risk/Reward: {self.risk_reward_ratio:.2f}

Trade Stats:
  Avg P&L: ${self.avg_trade_pnl:.2f}
  Avg Win: ${self.avg_winning_trade:.2f}
  Avg Loss: ${self.avg_losing_trade:.2f}
  Largest Win: ${self.largest_win:.2f}
  Largest Loss: ${self.largest_loss:.2f}

Streaks:
  Max Consecutive Wins: {self.max_consecutive_wins}
  Max Consecutive Losses: {self.max_consecutive_losses}
"""


def compute_strategy_metrics(
    equity_curve: list[float] | np.ndarray,
    trades: list[dict[str, Any]],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> StrategyMetrics:
    """
    Compute comprehensive strategy metrics.
    
    Args:
        equity_curve: List of portfolio values
        trades: List of trade dicts
        periods_per_year: Trading periods per year
        risk_free_rate: Annual risk-free rate
        
    Returns:
        StrategyMetrics object
    """
    equity = np.array(equity_curve)
    returns = calculate_returns(equity)
    
    # Basic metrics
    total_return = (equity[-1] / equity[0] - 1) if len(equity) > 1 else 0.0
    n_periods = len(equity) - 1 if len(equity) > 1 else 1
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    
    # Risk metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(equity, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(equity)
    volatility = float(np.std(returns) * np.sqrt(periods_per_year)) if len(returns) > 0 else 0.0
    
    # Trade metrics
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    expectancy = calculate_expectancy(trades)
    rr_ratio = calculate_risk_reward_ratio(trades)
    
    # Per-trade stats
    avg_trade = total_pnl / len(trades) if trades else 0.0
    avg_win = sum(t.get("pnl", 0) for t in winning) / len(winning) if winning else 0.0
    avg_loss = sum(t.get("pnl", 0) for t in losing) / len(losing) if losing else 0.0
    
    all_pnls = [t.get("pnl", 0) for t in trades]
    largest_win = max(all_pnls) if all_pnls else 0.0
    largest_loss = min(all_pnls) if all_pnls else 0.0
    
    # Streaks
    streak_stats = calculate_consecutive_stats(trades)
    
    # Duration (if available)
    total_duration = 0.0
    for t in trades:
        if "duration_seconds" in t:
            total_duration += t["duration_seconds"] / 3600
    avg_duration = total_duration / len(trades) if trades else 0.0
    
    return StrategyMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        volatility=volatility,
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        risk_reward_ratio=rr_ratio,
        avg_trade_pnl=avg_trade,
        avg_winning_trade=avg_win,
        avg_losing_trade=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        max_consecutive_wins=streak_stats["max_consecutive_wins"],
        max_consecutive_losses=streak_stats["max_consecutive_losses"],
        avg_trade_duration_hours=avg_duration,
        total_time_in_market_hours=total_duration
    )
