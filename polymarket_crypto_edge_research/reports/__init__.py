"""
Reports module for performance metrics and visualization.
"""

from reports.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_expectancy,
    StrategyMetrics,
    compute_strategy_metrics,
)
from reports.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_trade_analysis,
    create_performance_dashboard,
)

__all__ = [
    "calculate_returns",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_expectancy",
    "StrategyMetrics",
    "compute_strategy_metrics",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
    "plot_trade_analysis",
    "create_performance_dashboard",
]
