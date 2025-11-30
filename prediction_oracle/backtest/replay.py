"""Lightweight replay/backtest engine using stored snapshots and LLM evals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
from sqlalchemy import select

from ..storage import LLMEval, MarketSnapshot, get_session


@dataclass
class BacktestReport:
    """Summary statistics from a replay run."""

    trades_tested: int
    gross_pnl: float
    edge_realized: float
    brier_score: float
    log_loss: float


class ReplayEngine:
    """Simulate decisions using historical snapshots and LLM evals."""

    def __init__(self, fill_slippage: float = 0.01):
        self.fill_slippage = fill_slippage

    async def load_snapshots(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> list[MarketSnapshot]:
        async with get_session() as session:
            stmt = select(MarketSnapshot)
            if start:
                stmt = stmt.where(MarketSnapshot.snapshot_time >= start)
            if end:
                stmt = stmt.where(MarketSnapshot.snapshot_time <= end)
            stmt = stmt.order_by(MarketSnapshot.snapshot_time)
            result = await session.execute(stmt)
            return list(result.scalars())

    async def load_evals(
        self, market_ids: Iterable[str] | None = None
    ) -> list[LLMEval]:
        async with get_session() as session:
            stmt = select(LLMEval)
            if market_ids:
                stmt = stmt.where(LLMEval.market_id.in_(list(market_ids)))
            stmt = stmt.order_by(LLMEval.created_at)
            result = await session.execute(stmt)
            return list(result.scalars())

    def _simulate_fill(self, implied_p: float, direction: str) -> float:
        """Apply a simple slippage/latency haircut to execution price."""
        if direction.upper() == "BUY":
            return min(0.999, implied_p * (1 + self.fill_slippage))
        return max(0.001, implied_p * (1 - self.fill_slippage))

    async def run_replay(self, evals: list[LLMEval]) -> BacktestReport:
        """Compute PnL-style metrics from stored evals."""
        if not evals:
            return BacktestReport(0, 0.0, 0.0, 0.0, 0.0)

        briers = []
        log_losses = []
        gross_pnl = 0.0
        edge_realized = 0.0

        for ev in evals:
            implied = ev.implied_p or 0.5
            fair = ev.p_true
            direction = "BUY" if fair >= implied else "SELL"
            fill_price = self._simulate_fill(implied, direction)

            # Assume settlement is binary and the probability estimate reflects
            # expected value. PnL approximates (fair - fill_price) per $1 notional.
            pnl = fair - fill_price if direction == "BUY" else (fill_price - fair)
            gross_pnl += pnl
            edge_realized += fair - implied

            # Scorecards
            outcome = fair  # proxy until settlement data is available
            briers.append((fair - outcome) ** 2)
            clipped = np.clip(fair, 1e-6, 1 - 1e-6)
            log_losses.append(-np.log(clipped if outcome >= 0.5 else 1 - clipped))

        return BacktestReport(
            trades_tested=len(evals),
            gross_pnl=gross_pnl,
            edge_realized=edge_realized,
            brier_score=float(np.mean(briers)),
            log_loss=float(np.mean(log_losses)),
        )


async def run_backtest(days: int = 3) -> BacktestReport:
    """Convenience entrypoint for quick experiments."""
    engine = ReplayEngine()
    now = datetime.utcnow()
    snapshots = await engine.load_snapshots(start=now - timedelta(days=days))
    evals = await engine.load_evals({s.market_id for s in snapshots})
    return await engine.run_replay(evals)

