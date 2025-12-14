"""Longshot performance metrics for evaluating trader skill on low-probability outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from sqlalchemy import delete

from ..storage import get_session
from ..storage.models import TraderLongshotScore

# Probability buckets expressed as fractions (e.g., 0.05 == 5%).
BUCKETS = [
    ("0-5%", 0.0, 0.05),
    ("5-10%", 0.05, 0.10),
    ("10-20%", 0.10, 0.20),
]


@dataclass
class TradeOutcome:
    """Resolved trade outcome used for longshot evaluation."""

    trader_id: str
    implied_prob: float
    won: bool
    resolved_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not 0 <= self.implied_prob <= 1:
            raise ValueError("implied_prob must be between 0 and 1")


@dataclass
class LongshotBucketStats:
    """Performance statistics for a probability bucket."""

    name: str
    lower: float
    upper: float
    trades: int
    wins: int
    hit_rate: float
    market_baseline: float
    lift: float


@dataclass
class TraderLongshotMetrics:
    """Aggregate longshot metrics for a single trader."""

    trader_id: str
    as_of: date
    bucket_stats: Dict[str, LongshotBucketStats]
    average_lift: float
    stability: float
    longshot_edge_score: float
    decayed_recent_score: float

    def to_record(self) -> TraderLongshotScore:
        """Convert the metrics into a database model instance."""

        bucket_payload: MutableMapping[str, MutableMapping[str, float]] = {}
        for name, stats in self.bucket_stats.items():
            bucket_payload[name] = {
                "lower": stats.lower,
                "upper": stats.upper,
                "trades": stats.trades,
                "wins": stats.wins,
                "hit_rate": stats.hit_rate,
                "market_baseline": stats.market_baseline,
                "lift": stats.lift,
            }

        return TraderLongshotScore(
            trader_id=self.trader_id,
            as_of_date=self.as_of,
            bucket_stats_json=bucket_payload,
            average_lift=self.average_lift,
            stability=self.stability,
            longshot_edge_score=self.longshot_edge_score,
            decayed_recent_score=self.decayed_recent_score,
        )


def _bucket_name(implied_prob: float) -> Optional[str]:
    for name, lower, upper in BUCKETS:
        if lower <= implied_prob <= upper:
            return name
    return None


def _compute_bucket_stats(trades: List[TradeOutcome]) -> Dict[str, LongshotBucketStats]:
    accumulators: Dict[str, Dict[str, float]] = {}
    for name, lower, upper in BUCKETS:
        accumulators[name] = {
            "lower": lower,
            "upper": upper,
            "trades": 0,
            "wins": 0,
            "implied_sum": 0.0,
        }

    for trade in trades:
        bucket = _bucket_name(trade.implied_prob)
        if bucket is None:
            continue

        acc = accumulators[bucket]
        acc["trades"] += 1
        acc["wins"] += 1 if trade.won else 0
        acc["implied_sum"] += trade.implied_prob

    stats: Dict[str, LongshotBucketStats] = {}
    for name, acc in accumulators.items():
        trades_count = int(acc["trades"])
        wins = int(acc["wins"])
        hit_rate = wins / trades_count if trades_count else 0.0
        market_baseline = (acc["implied_sum"] / trades_count) if trades_count else 0.0
        lift = (
            (hit_rate - market_baseline) / market_baseline if market_baseline > 0 else 0.0
        )

        stats[name] = LongshotBucketStats(
            name=name,
            lower=acc["lower"],
            upper=acc["upper"],
            trades=trades_count,
            wins=wins,
            hit_rate=hit_rate,
            market_baseline=market_baseline,
            lift=lift,
        )

    return stats


def _sharpe_like(edges: List[float]) -> float:
    if len(edges) < 2:
        return 0.0

    mean_edge = sum(edges) / len(edges)
    variance = sum((edge - mean_edge) ** 2 for edge in edges) / (len(edges) - 1)
    std_edge = math.sqrt(variance)
    if std_edge == 0:
        return 0.0

    return (mean_edge / std_edge) * math.sqrt(len(edges))


def _longshot_edge_score(average_lift: float, stability: float) -> float:
    lift_component = math.tanh(average_lift)
    stability_component = math.tanh(stability / 2)
    combined = 0.6 * lift_component + 0.4 * stability_component
    return 50.0 * (combined + 1.0)


def _decayed_score(trades: List[TradeOutcome], as_of: datetime, half_life_days: float) -> float:
    if not trades:
        return 0.0

    weights = []
    edges = []
    for trade in trades:
        reference_time = trade.resolved_at or as_of
        days = max((as_of - reference_time).total_seconds() / 86400.0, 0.0)
        weight = math.exp(-days / half_life_days) if half_life_days > 0 else 1.0
        weights.append(weight)
        edges.append((1.0 if trade.won else 0.0) - trade.implied_prob)

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_edge = sum(w * e for w, e in zip(weights, edges)) / total_weight
    return 50.0 * (math.tanh(weighted_edge) + 1.0)


def compute_trader_longshot_metrics(
    trades: Iterable[TradeOutcome],
    *,
    as_of: Optional[datetime] = None,
    half_life_days: float = 14.0,
) -> Dict[str, TraderLongshotMetrics]:
    """
    Compute longshot metrics for each trader in the provided trade outcomes.

    Args:
        trades: Iterable of resolved trade outcomes.
        as_of: Timestamp used for recency decay; defaults to now.
        half_life_days: Half-life for the decayed recent performance score.

    Returns:
        Mapping of trader_id to TraderLongshotMetrics.
    """

    as_of_ts = as_of or datetime.utcnow()

    per_trader: Dict[str, List[TradeOutcome]] = {}
    for trade in trades:
        per_trader.setdefault(trade.trader_id, []).append(trade)

    metrics: Dict[str, TraderLongshotMetrics] = {}
    for trader_id, trader_trades in per_trader.items():
        bucket_stats = _compute_bucket_stats(trader_trades)
        edges = [
            (1.0 if t.won else 0.0) - t.implied_prob
            for t in trader_trades
            if _bucket_name(t.implied_prob) is not None
        ]
        stability = _sharpe_like(edges)

        total_bucket_trades = sum(stat.trades for stat in bucket_stats.values())
        weighted_lift = (
            sum(stat.lift * stat.trades for stat in bucket_stats.values()) / total_bucket_trades
            if total_bucket_trades
            else 0.0
        )

        longshot_edge_score = _longshot_edge_score(weighted_lift, stability)
        decayed_recent_score = _decayed_score(trader_trades, as_of_ts, half_life_days)

        metrics[trader_id] = TraderLongshotMetrics(
            trader_id=trader_id,
            as_of=as_of_ts.date(),
            bucket_stats=bucket_stats,
            average_lift=weighted_lift,
            stability=stability,
            longshot_edge_score=longshot_edge_score,
            decayed_recent_score=decayed_recent_score,
        )

    return metrics


async def persist_trader_longshot_scores(
    metrics: Mapping[str, TraderLongshotMetrics],
    *,
    as_of_date: Optional[date] = None,
) -> None:
    """Persist daily longshot scores to the database.

    Existing records for the given trader/date pair are replaced to allow daily refreshes.
    """

    target_date = as_of_date or (next(iter(metrics.values())).as_of if metrics else date.today())

    async with get_session() as session:
        for trader_id, metric in metrics.items():
            await session.execute(
                delete(TraderLongshotScore).where(
                    TraderLongshotScore.trader_id == trader_id,
                    TraderLongshotScore.as_of_date == target_date,
                )
            )
            record = metric.to_record()
            record.as_of_date = target_date
            session.add(record)
