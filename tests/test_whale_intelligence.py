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
