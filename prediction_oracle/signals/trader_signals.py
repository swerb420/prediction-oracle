"""Signals for tracking top trader position changes.

This module detects when tracked "top" traders materially change
their position sizes, aggregates those signals across short windows,
and enriches them with market screening scores.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable


@dataclass(frozen=True)
class MarketScreenScore:
    """Lightweight representation of a market's screening result."""

    score: float
    reason: str = ""


@dataclass(frozen=True)
class TraderPositionSnapshot:
    """Trader position at a point in time."""

    trader_id: str
    market_id: str
    outcome_id: str
    size: float  # Shares/contracts
    notional_usd: float
    open_interest_usd: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class TraderSignal:
    """A single position-change signal from a tracked trader."""

    trader_id: str
    market_id: str
    outcome_id: str
    change_size: float
    change_notional: float
    change_pct_open_interest: float | None
    direction: str  # "increase" or "decrease"
    triggered_by: str  # "notional" or "open_interest_pct"
    timestamp: datetime
    screen_score: float | None = None
    screen_reason: str | None = None


@dataclass(frozen=True)
class AggregatedTraderSignal:
    """Aggregate view of trader flows for a market/outcome window."""

    market_id: str
    outcome_id: str
    window_start: datetime
    window_end: datetime
    trader_count: int
    total_notional_change: float
    net_size_change: float
    dominant_direction: str  # "increase", "decrease", or "mixed"
    screen_score: float | None = None
    screen_reason: str | None = None


@dataclass
class TraderSignalConfig:
    """Configuration for signal thresholds and aggregation."""

    min_notional_change: float = 5_000.0
    min_pct_open_interest: float = 0.02
    aggregation_window_seconds: int = 300
    max_buffered_signals: int = 1_000


class TraderSignalDetector:
    """Detect and aggregate position change signals for top traders."""

    def __init__(
        self,
        top_trader_ids: Iterable[str],
        *,
        config: TraderSignalConfig | None = None,
        market_screen_scores: dict[str, MarketScreenScore] | None = None,
    ) -> None:
        self.top_traders = set(top_trader_ids)
        self.config = config or TraderSignalConfig()
        self.market_screen_scores = market_screen_scores or {}
        self._signals: deque[TraderSignal] = deque()

    def record_position_change(
        self,
        previous: TraderPositionSnapshot,
        current: TraderPositionSnapshot,
    ) -> TraderSignal | None:
        """Return a signal if a tracked trader materially changed size.

        Only evaluates updates for traders in ``top_trader_ids``. A signal is
        triggered when the notional change exceeds ``min_notional_change`` or
        when the percentage of open interest exceeds ``min_pct_open_interest``.
        """

        if current.trader_id not in self.top_traders:
            return None

        change_size = current.size - previous.size
        change_notional = current.notional_usd - previous.notional_usd

        if abs(change_size) < 1e-9 and abs(change_notional) < 1e-9:
            return None

        pct_oi = None
        if current.open_interest_usd:
            pct_oi = abs(change_notional) / current.open_interest_usd

        triggered_by = None
        if abs(change_notional) >= self.config.min_notional_change:
            triggered_by = "notional"
        elif pct_oi is not None and pct_oi >= self.config.min_pct_open_interest:
            triggered_by = "open_interest_pct"

        if not triggered_by:
            return None

        direction = "increase" if change_size > 0 else "decrease"
        screen_data = self.market_screen_scores.get(current.market_id)

        signal = TraderSignal(
            trader_id=current.trader_id,
            market_id=current.market_id,
            outcome_id=current.outcome_id,
            change_size=change_size,
            change_notional=change_notional,
            change_pct_open_interest=pct_oi,
            direction=direction,
            triggered_by=triggered_by,
            timestamp=current.timestamp,
            screen_score=screen_data.score if screen_data else None,
            screen_reason=screen_data.reason if screen_data else None,
        )

        self._append_signal(signal)
        return signal

    def aggregate_recent(self, now: datetime | None = None) -> list[AggregatedTraderSignal]:
        """Aggregate signals by market/outcome within the configured window."""

        if now is None:
            now = datetime.now(timezone.utc)

        self._prune_old_signals(now)

        grouped: dict[tuple[str, str], list[TraderSignal]] = defaultdict(list)
        for signal in self._signals:
            grouped[(signal.market_id, signal.outcome_id)].append(signal)

        aggregated: list[AggregatedTraderSignal] = []
        window_start = now - timedelta(seconds=self.config.aggregation_window_seconds)

        for (market_id, outcome_id), signals in grouped.items():
            trader_ids = {s.trader_id for s in signals}
            total_notional = sum(s.change_notional for s in signals)
            net_size = sum(s.change_size for s in signals)

            dominant_direction = "mixed"
            if net_size > 0:
                dominant_direction = "increase"
            elif net_size < 0:
                dominant_direction = "decrease"

            # Prefer the latest screen data in the window
            last_signal = signals[-1]

            aggregated.append(
                AggregatedTraderSignal(
                    market_id=market_id,
                    outcome_id=outcome_id,
                    window_start=window_start,
                    window_end=now,
                    trader_count=len(trader_ids),
                    total_notional_change=total_notional,
                    net_size_change=net_size,
                    dominant_direction=dominant_direction,
                    screen_score=last_signal.screen_score,
                    screen_reason=last_signal.screen_reason,
                )
            )

        return aggregated

    def update_screen_scores(self, scores: dict[str, MarketScreenScore]) -> None:
        """Update market screening scores used to enrich signals."""

        self.market_screen_scores.update(scores)

    def _append_signal(self, signal: TraderSignal) -> None:
        """Append a signal and keep the buffer bounded."""

        self._signals.append(signal)
        while len(self._signals) > self.config.max_buffered_signals:
            self._signals.popleft()

    def _prune_old_signals(self, now: datetime) -> None:
        """Drop signals outside of the aggregation window."""

        cutoff = now - timedelta(seconds=self.config.aggregation_window_seconds)
        while self._signals and self._signals[0].timestamp < cutoff:
            self._signals.popleft()
