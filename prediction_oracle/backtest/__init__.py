"""Backtesting helpers for the prediction oracle."""

from .replay import BacktestReport, ReplayEngine
from .utils import (
    HedgeEvaluator,
    HedgeReport,
    OrderBookLevel,
    OrderBookReplayer,
    OrderBookSlice,
    RiskStats,
    summarize_risk,
)

__all__ = [
    "ReplayEngine",
    "BacktestReport",
    "OrderBookLevel",
    "OrderBookSlice",
    "OrderBookReplayer",
    "HedgeEvaluator",
    "HedgeReport",
    "RiskStats",
    "summarize_risk",
]

