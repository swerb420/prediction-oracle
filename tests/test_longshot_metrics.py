from datetime import datetime

import pytest

from prediction_oracle.metrics import compute_trader_longshot_metrics, TradeOutcome


def test_longshot_bucket_and_scores():
    as_of = datetime(2024, 1, 15)
    trades = [
        TradeOutcome("alice", 0.03, True, resolved_at=datetime(2024, 1, 1)),
        TradeOutcome("alice", 0.08, False, resolved_at=datetime(2024, 1, 10)),
        TradeOutcome("alice", 0.15, True, resolved_at=datetime(2024, 1, 12)),
        TradeOutcome("bob", 0.12, True, resolved_at=datetime(2024, 1, 14)),
        TradeOutcome("bob", 0.18, False, resolved_at=datetime(2023, 12, 20)),
    ]

    metrics = compute_trader_longshot_metrics(trades, as_of=as_of, half_life_days=7)

    alice = metrics["alice"]
    stats = alice.bucket_stats

    assert stats["0-5%"].trades == 1
    assert stats["0-5%"].wins == 1
    assert stats["0-5%"].hit_rate == pytest.approx(1.0)
    assert stats["0-5%"].market_baseline == pytest.approx(0.03)
    assert stats["0-5%"].lift == pytest.approx((1.0 - 0.03) / 0.03)

    assert stats["5-10%"].trades == 1
    assert stats["5-10%"].wins == 0
    assert stats["5-10%"].hit_rate == pytest.approx(0.0)
    assert stats["5-10%"].market_baseline == pytest.approx(0.08)

    assert stats["10-20%"].trades == 1
    assert stats["10-20%"].wins == 1
    assert stats["10-20%"].hit_rate == pytest.approx(1.0)
    assert stats["10-20%"].market_baseline == pytest.approx(0.15)

    expected_average_lift = ((32.333333333333336) + (-1.0) + (5.666666666666667)) / 3
    assert alice.average_lift == pytest.approx(expected_average_lift)
    assert alice.stability == pytest.approx(1.7479, rel=1e-3)
    assert 90 <= alice.longshot_edge_score <= 100

    bob = metrics["bob"]
    assert bob.bucket_stats["10-20%"].trades == 2
    assert bob.bucket_stats["10-20%"].wins == 1
    assert bob.stability == pytest.approx(0.6603, rel=1e-3)
    assert bob.decayed_recent_score > 80
    assert 0 <= bob.longshot_edge_score <= 100

    for trader_metrics in metrics.values():
        assert trader_metrics.as_of == as_of.date()
        assert trader_metrics.decayed_recent_score >= 0
        assert trader_metrics.decayed_recent_score <= 100
