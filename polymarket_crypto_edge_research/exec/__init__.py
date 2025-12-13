"""Execution module for paper trading, backtesting, and live orchestration."""

from .paper_trader import (
    PaperTrader,
    PaperTrade,
    PaperPortfolio,
)
from .backtester import (
    Backtester,
    BacktestResult,
    BacktestConfig,
)
from .live_orchestrator import (
    LiveOrchestrator,
    OrchestratorConfig,
)

__all__ = [
    # Paper trading
    "PaperTrader",
    "PaperTrade",
    "PaperPortfolio",
    # Backtesting
    "Backtester",
    "BacktestResult",
    "BacktestConfig",
    # Live orchestration
    "LiveOrchestrator",
    "OrchestratorConfig",
]
