from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight stub so imports succeed without the real dependency.
if "polymarket_apis" not in sys.modules:
    stub_module = types.SimpleNamespace(PolymarketDataClient=types.SimpleNamespace)
    sys.modules["polymarket_apis"] = stub_module

from datetime import datetime, timedelta, UTC

import pytest

from prediction_oracle.llm.whale_intelligence import init_database, WhaleIntelligence


def seed_sample_data(conn, now: datetime) -> None:
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO whale_profiles (
            wallet, name, pseudonym, pnl_30d, pnl_all, crypto_focus_ratio,
            crypto_trade_count, is_crypto_specialist, rank_30d_profit, rank_all_profit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("w1", "Alpha", "alpha", 50_000, 120_000, 0.9, 120, 1, 5, 10),
    )
    cursor.execute(
        """
        INSERT INTO whale_profiles (
            wallet, name, pseudonym, pnl_30d, pnl_all, crypto_focus_ratio,
            crypto_trade_count, is_crypto_specialist, rank_30d_profit, rank_all_profit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("w2", "Beta", "beta", 25_000, 40_000, 0.6, 80, 0, 20, 50),
    )

    trades = [
        # Wallet 1: heavily long
        ("w1", now - timedelta(minutes=5), "BTC", "15MIN", "UP", "BUY", 10, 1.0, 1_000, "Up or Down", "m1", "up", "tx1"),
        ("w1", now - timedelta(minutes=4), "BTC", "15MIN", "UP", "BUY", 10, 1.0, 1_200, "Up or Down", "m1", "up", "tx2"),
        ("w1", now - timedelta(minutes=3), "ETH", "HOURLY", "DOWN", "SELL", 8, 1.0, 500, "Hourly move", "m2", "down", "tx3"),
        ("w1", now - timedelta(minutes=2), "BTC", "15MIN", "UP", "BUY", 12, 1.0, 1_500, "Up or Down", "m1", "up", "tx4"),
        # Wallet 2: mostly short
        ("w2", now - timedelta(minutes=10), "BTC", "15MIN", "DOWN", "SELL", 15, 1.0, 700, "Up or Down", "m1", "down", "tx5"),
        ("w2", now - timedelta(minutes=8), "ETH", "HOURLY", "DOWN", "SELL", 10, 1.0, 1_800, "Hourly move", "m2", "down", "tx6"),
        ("w2", now - timedelta(minutes=6), "SOL", "15MIN", "UP", "BUY", 5, 1.0, 600, "Up swing", "m3", "up", "tx7"),
    ]

    cursor.executemany(
        """
        INSERT INTO crypto_trades (
            whale_wallet, timestamp, symbol, market_type, direction, side, size, price,
            usdc_value, market_title, condition_id, outcome, tx_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                wallet,
                ts.isoformat(),
                symbol,
                market_type,
                direction,
                side,
                size,
                price,
                usdc_value,
                market_title,
                condition_id,
                outcome,
                tx_hash,
            )
            for (
                wallet,
                ts,
                symbol,
                market_type,
                direction,
                side,
                size,
                price,
                usdc_value,
                market_title,
                condition_id,
                outcome,
                tx_hash,
            ) in trades
        ],
    )

    conn.commit()


@pytest.fixture()
def seeded_whale_db(tmp_path):
    db_path = tmp_path / "whales.db"
    init_database(db_path)

    now = datetime.now(UTC)
    with WhaleIntelligence(db_path=db_path) as wi:
        seed_sample_data(wi.conn, now)
    return db_path, now


def test_intraday_dashboards_rank_and_bias(seeded_whale_db):
    db_path, now = seeded_whale_db

    with WhaleIntelligence(db_path=db_path) as wi:
        dashboards = wi.get_intraday_wallet_dashboards(windows=[15], min_trades=2, top_n=10)

    assert "15m" in dashboards
    alpha = next(entry for entry in dashboards["15m"] if entry["wallet"] == "w1")

    assert alpha["trade_count"] == 4
    assert alpha["total_volume"] == pytest.approx(4_200)
    assert alpha["direction_bias"] > 0.55  # heavy UP flow
    assert alpha["market_types"]["15MIN"]["trades"] == 3
    assert alpha["last_trade"].startswith(str(now.year))


def test_get_live_signal_prefers_stronger_flow(seeded_whale_db):
    db_path, _ = seeded_whale_db

    with WhaleIntelligence(db_path=db_path) as wi:
        signal = wi.get_live_signal("BTC", minutes_back=30)

    assert signal is not None
    assert signal["direction"] == "UP"
    assert signal["confidence"] > 0.5
    assert signal["up_whales"] >= signal["down_whales"]
    assert signal["specialists"] >= 1  # wallet w1 is marked specialist


def test_intraday_symbol_filter_and_velocity(seeded_whale_db):
    db_path, _ = seeded_whale_db

    with WhaleIntelligence(db_path=db_path) as wi:
        dashboards = wi.get_intraday_wallet_dashboards(windows=[15], symbol_filter="BTC")

    assert "15m" in dashboards
    alpha = next(entry for entry in dashboards["15m"] if entry["wallet"] == "w1")

    assert alpha["top_symbol"] == "BTC"
    assert alpha["top_symbol_share"] == pytest.approx(1.0)
    assert alpha["volume_velocity"] == pytest.approx(3700 / 15)
    assert alpha["trade_rate"] == pytest.approx(3 / 15)


def test_copytrade_candidates_respect_thresholds(seeded_whale_db):
    db_path, _ = seeded_whale_db

    with WhaleIntelligence(db_path=db_path) as wi:
        candidates = wi.rank_copytrade_candidates(min_focus_ratio=0.7, min_trades=100)

    assert candidates
    assert candidates[0]["wallet"] == "w1"
    assert candidates[0]["trades_lookback"] >= 1
    assert candidates[0]["quality_score"] > 0
    assert candidates[0]["recency_multiplier"] >= 0.4


def test_wallet_deep_dive_reports_breakdowns(seeded_whale_db):
    db_path, _ = seeded_whale_db

    with WhaleIntelligence(db_path=db_path) as wi:
        report = wi.get_wallet_deep_dive("w1", lookback_hours=12)

    assert report is not None
    summary = report["summary"]

    assert summary["net_flow"] > 0
    assert "BTC" in summary["symbol_breakdown"]
    assert summary["symbol_breakdown"]["BTC"]["volume"] == pytest.approx(3700)
    assert len(report["recent_trades"]) == 4


def test_backtest_copytrade_strategy_uses_edge(seeded_whale_db):
    db_path, _ = seeded_whale_db

    with WhaleIntelligence(db_path=db_path) as wi:
        results = wi.backtest_copytrade_strategy(lookback_days=1, top_k=2, min_focus_ratio=0.0)

    # Wallet w1 has higher expected edge and should rank first
    assert results["wallets"][0]["wallet"] == "w1"
    assert results["wallets"][0]["expected_pnl"] > results["wallets"][1]["expected_pnl"]
    expected_total = sum(w["expected_pnl"] for w in results["wallets"])
    assert results["total_expected_pnl"] == pytest.approx(expected_total)
    risk_adjusted_total = sum(w["risk_adjusted_expected_pnl"] for w in results["wallets"])
    assert results["total_risk_adjusted_pnl"] == pytest.approx(risk_adjusted_total)
    assert results["wallets"][0]["risk_adjusted_expected_pnl"] >= results["wallets"][0]["expected_pnl"] * 0.4


def test_export_ml_training_data_builds_feature_rows(seeded_whale_db):
    db_path, _ = seeded_whale_db

    with WhaleIntelligence(db_path=db_path) as wi:
        rows = wi.export_ml_training_data(lookback_days=2)

    assert len(rows) == 7  # all seeded trades
    sample = next(row for row in rows if row["wallet"] == "w1")
    assert "quality_score" in sample and sample["quality_score"] > 0
    assert sample["hour"] in range(0, 24)
    assert "consistency" in sample and sample["consistency"] >= 0.1
