"""Research utilities for replaying markets and benchmarking hedges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import random
from typing import Iterable, Sequence

import numpy as np

from ..storage import MarketSnapshot


@dataclass
class OrderBookLevel:
    """Single price/size level in an order book."""

    price: float
    size: float


@dataclass
class OrderBookSlice:
    """Approximate order book view reconstructed from a snapshot."""

    snapshot_time: datetime
    midpoint: float
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]


class OrderBookReplayer:
    """Replay historical snapshots to model slippage impacts."""

    def __init__(self, spread: float = 0.01, levels: int = 5, min_depth: float = 50.0):
        self.spread = spread
        self.levels = levels
        self.min_depth = min_depth

    def _base_price(self, snapshot: MarketSnapshot) -> float:
        prices = snapshot.prices_json or {}
        if not prices:
            return 0.5
        return float(np.mean([float(p) for p in prices.values()]))

    def _base_depth(self, snapshot: MarketSnapshot) -> float:
        volumes = snapshot.volume_json or {}
        if not volumes:
            return self.min_depth
        return max(self.min_depth, float(np.mean([float(v) for v in volumes.values()])))

    def _approximate_slice(self, snapshot: MarketSnapshot) -> OrderBookSlice:
        mid = self._base_price(snapshot)
        depth = self._base_depth(snapshot)
        level_size = depth / self.levels
        bids: list[OrderBookLevel] = []
        asks: list[OrderBookLevel] = []

        for idx in range(self.levels):
            offset = self.spread * (idx + 1)
            bid_price = max(0.001, mid * (1 - offset))
            ask_price = min(0.999, mid * (1 + offset))
            bids.append(OrderBookLevel(price=bid_price, size=level_size))
            asks.append(OrderBookLevel(price=ask_price, size=level_size))

        return OrderBookSlice(snapshot_time=snapshot.snapshot_time, midpoint=mid, bids=bids, asks=asks)

    def replay_slices(self, snapshots: Iterable[MarketSnapshot]) -> list[OrderBookSlice]:
        """Build approximate order book slices from stored snapshots."""

        return [self._approximate_slice(s) for s in snapshots]

    def simulate_market_order(self, ob: OrderBookSlice, side: str, size: float) -> tuple[float, float]:
        """Estimate average fill price and slippage for a market order.

        Returns a tuple of (avg_fill_price, slippage_vs_mid).
        """

        ladder = ob.asks if side.upper() == "BUY" else list(reversed(ob.bids))
        remaining = size
        cost = 0.0

        for level in ladder:
            take = min(remaining, level.size)
            cost += take * level.price
            remaining -= take
            if remaining <= 1e-9:
                break

        if remaining > 0:
            # Walk beyond quoted depth at worst observed price.
            terminal_price = ladder[-1].price
            cost += remaining * terminal_price

        avg_price = cost / size if size else ob.midpoint
        slippage = avg_price - ob.midpoint if side.upper() == "BUY" else ob.midpoint - avg_price
        return avg_price, slippage

    def replay_slippage(self, slices: Iterable[OrderBookSlice], side: str, size: float) -> list[float]:
        """Compute slippage across historical slices for a consistent order size."""

        return [self.simulate_market_order(ob, side, size)[1] for ob in slices]


@dataclass
class RiskStats:
    """Risk summary for an equity or return series."""

    max_drawdown: float
    variance: float
    tail_loss: float


@dataclass
class HedgeReport:
    """Comparison of hedge performance vs. benchmarks."""

    strategy_returns: list[float]
    random_returns: list[float]
    long_only_returns: list[float]
    strategy_risk: RiskStats
    random_risk: RiskStats
    long_only_risk: RiskStats


def _equity_curve(returns: Sequence[float]) -> np.ndarray:
    return np.cumsum(np.array(returns, dtype=float))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    return float(np.max(drawdowns))


def _tail_loss(returns: Sequence[float], quantile: float = 0.05) -> float:
    if not returns:
        return 0.0
    sorted_ret = sorted(returns)
    cutoff = max(1, int(len(sorted_ret) * quantile))
    tail = sorted_ret[:cutoff]
    return float(np.mean(tail)) if tail else 0.0


def summarize_risk(returns: Sequence[float]) -> RiskStats:
    equity = _equity_curve(returns)
    return RiskStats(
        max_drawdown=_max_drawdown(equity),
        variance=float(np.var(returns)) if returns else 0.0,
        tail_loss=_tail_loss(returns),
    )


class HedgeEvaluator:
    """Benchmark hedge strategies against naive baselines."""

    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def _random_signals(self, n: int) -> list[int]:
        return [self.random.choice([-1, 0, 1]) for _ in range(n)]

    def evaluate(self, signals: Sequence[int], base_returns: Sequence[float]) -> HedgeReport:
        if len(signals) != len(base_returns):
            raise ValueError("Signals and returns must have identical lengths.")

        strategy_returns = [sig * ret for sig, ret in zip(signals, base_returns)]
        random_returns = [sig * ret for sig, ret in zip(self._random_signals(len(signals)), base_returns)]
        long_only_returns = list(base_returns)

        return HedgeReport(
            strategy_returns=strategy_returns,
            random_returns=random_returns,
            long_only_returns=long_only_returns,
            strategy_risk=summarize_risk(strategy_returns),
            random_risk=summarize_risk(random_returns),
            long_only_risk=summarize_risk(long_only_returns),
        )
