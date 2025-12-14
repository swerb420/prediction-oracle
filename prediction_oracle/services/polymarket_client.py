"""Lightweight Polymarket API client and incremental data fetcher.

This module wraps public Polymarket endpoints for leaderboard, markets, order
books, and trader positions. It also ships with an incremental fetcher that
normalizes the data and stores both the raw payloads and curated snapshots in a
DuckDB database so the system can track trader performance and market quality
over time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import duckdb
import httpx

logger = logging.getLogger(__name__)

LeaderboardWindow = Literal["1d", "7d", "30d", "all"]
LeaderboardMetric = Literal["profit", "volume"]


@dataclass
class LeaderboardEntry:
    """Structured leaderboard record."""

    trader_id: str
    username: str | None
    pnl: float | None
    roe: float | None
    win_rate: float | None
    hit_rate_buckets: dict[str, Any]
    window: LeaderboardWindow
    metric: LeaderboardMetric
    observed_at: datetime


@dataclass
class MarketSnapshot:
    """Snapshot of a market's state at a moment in time."""

    market_id: str
    question: str
    status: str | None
    fee: float | None
    liquidity: float | None
    spread_bps: float | None
    volume_24h: float | None
    expiry: datetime | None
    observed_at: datetime


@dataclass
class OrderBookSnapshot:
    """Aggregated order book state for an outcome token."""

    market_id: str
    token_id: str
    best_bid: float | None
    best_ask: float | None
    bid_depth: float | None
    ask_depth: float | None
    spread_bps: float | None
    observed_at: datetime


@dataclass
class TraderPosition:
    """Position snapshot for a trader in a market."""

    trader_id: str
    market_id: str
    outcome: str | None
    quantity: float | None
    avg_price: float | None
    pnl: float | None
    value: float | None
    position_change: float | None
    pnl_change: float | None
    observed_at: datetime


class PolymarketClient:
    """Async client for Polymarket public APIs."""

    DATA_API_BASE = "https://data-api.polymarket.com"
    LEADERBOARD_API = "https://lb-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(self, timeout: float = 15.0):
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("PolymarketClient must be used as an async context manager")
        return self._client

    async def fetch_leaderboard(
        self,
        *,
        window: LeaderboardWindow = "all",
        metric: LeaderboardMetric = "profit",
        limit: int = 50,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch leaderboard entries."""

        params: dict[str, Any] = {"window": window, "limit": limit}
        if tag:
            params["tag"] = tag

        # Leaderboard is hosted on lb-api while the rest of the data lives under data-api.
        url = f"{self.LEADERBOARD_API}/{metric}"
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

        return payload.get("users", payload)  # API sometimes nests under "users"

    async def fetch_markets(self, *, active: bool = True, limit: int = 512) -> list[dict[str, Any]]:
        """Fetch markets from the CLOB API."""

        params = {"active": str(active).lower(), "limit": limit}
        url = f"{self.CLOB_API}/markets"
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        return payload.get("markets", payload)

    async def fetch_order_book(self, token_id: str) -> dict[str, Any]:
        """Fetch the order book for an outcome token."""

        url = f"{self.CLOB_API}/orderbook"
        response = await self.client.get(url, params={"tokenId": token_id})
        response.raise_for_status()
        return response.json()

    async def fetch_trader_positions(self, trader_id: str) -> list[dict[str, Any]]:
        """Fetch current positions for a trader."""

        url = f"{self.DATA_API_BASE}/positions"
        response = await self.client.get(url, params={"user": trader_id})
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", payload)


class PolymarketDuckDBStore:
    """DuckDB-backed storage for Polymarket raw and normalized data."""

    def __init__(self, db_path: str | Path = "./data/polymarket.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_leaderboard (
                observed_at TIMESTAMP,
                window VARCHAR,
                metric VARCHAR,
                payload JSON
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_markets (
                observed_at TIMESTAMP,
                payload JSON
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_orderbooks (
                observed_at TIMESTAMP,
                market_id VARCHAR,
                token_id VARCHAR,
                payload JSON
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_positions (
                observed_at TIMESTAMP,
                trader_id VARCHAR,
                payload JSON
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leaderboard_metrics (
                observed_at TIMESTAMP,
                trader_id VARCHAR,
                username VARCHAR,
                window VARCHAR,
                metric VARCHAR,
                pnl DOUBLE,
                roe DOUBLE,
                win_rate DOUBLE,
                hit_rate JSON
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_metadata (
                observed_at TIMESTAMP,
                market_id VARCHAR,
                question VARCHAR,
                status VARCHAR,
                fee DOUBLE,
                liquidity DOUBLE,
                spread_bps DOUBLE,
                volume_24h DOUBLE,
                expiry TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                observed_at TIMESTAMP,
                market_id VARCHAR,
                token_id VARCHAR,
                best_bid DOUBLE,
                best_ask DOUBLE,
                bid_depth DOUBLE,
                ask_depth DOUBLE,
                spread_bps DOUBLE
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trader_positions (
                observed_at TIMESTAMP,
                trader_id VARCHAR,
                market_id VARCHAR,
                outcome VARCHAR,
                quantity DOUBLE,
                avg_price DOUBLE,
                pnl DOUBLE,
                value DOUBLE,
                position_change DOUBLE,
                pnl_change DOUBLE
            )
            """
        )

    # Raw insert helpers -------------------------------------------------
    def insert_raw_leaderboard(
        self, observed_at: datetime, window: str, metric: str, payload: list[dict[str, Any]]
    ) -> None:
        self.conn.execute(
            "INSERT INTO raw_leaderboard VALUES (?, ?, ?, ?)",
            [observed_at, window, metric, json.dumps(payload)],
        )

    def insert_raw_markets(self, observed_at: datetime, payload: list[dict[str, Any]]) -> None:
        self.conn.execute(
            "INSERT INTO raw_markets VALUES (?, ?)",
            [observed_at, json.dumps(payload)],
        )

    def insert_raw_orderbook(
        self, observed_at: datetime, market_id: str, token_id: str, payload: dict[str, Any]
    ) -> None:
        self.conn.execute(
            "INSERT INTO raw_orderbooks VALUES (?, ?, ?, ?)",
            [observed_at, market_id, token_id, json.dumps(payload)],
        )

    def insert_raw_positions(
        self, observed_at: datetime, trader_id: str, payload: list[dict[str, Any]]
    ) -> None:
        self.conn.execute(
            "INSERT INTO raw_positions VALUES (?, ?, ?)",
            [observed_at, trader_id, json.dumps(payload)],
        )

    # Normalized insert helpers ------------------------------------------
    def insert_leaderboard_entries(self, entries: Iterable[LeaderboardEntry]) -> None:
        self.conn.executemany(
            """
            INSERT INTO leaderboard_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry.observed_at,
                    entry.trader_id,
                    entry.username,
                    entry.window,
                    entry.metric,
                    entry.pnl,
                    entry.roe,
                    entry.win_rate,
                    json.dumps(entry.hit_rate_buckets),
                )
                for entry in entries
            ],
        )

    def insert_market_snapshots(self, snapshots: Iterable[MarketSnapshot]) -> None:
        self.conn.executemany(
            """
            INSERT INTO market_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot.observed_at,
                    snapshot.market_id,
                    snapshot.question,
                    snapshot.status,
                    snapshot.fee,
                    snapshot.liquidity,
                    snapshot.spread_bps,
                    snapshot.volume_24h,
                    snapshot.expiry,
                )
                for snapshot in snapshots
            ],
        )

    def insert_orderbook_snapshots(self, snapshots: Iterable[OrderBookSnapshot]) -> None:
        self.conn.executemany(
            """
            INSERT INTO orderbook_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot.observed_at,
                    snapshot.market_id,
                    snapshot.token_id,
                    snapshot.best_bid,
                    snapshot.best_ask,
                    snapshot.bid_depth,
                    snapshot.ask_depth,
                    snapshot.spread_bps,
                )
                for snapshot in snapshots
            ],
        )

    def insert_trader_positions(self, positions: Iterable[TraderPosition]) -> None:
        self.conn.executemany(
            """
            INSERT INTO trader_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    position.observed_at,
                    position.trader_id,
                    position.market_id,
                    position.outcome,
                    position.quantity,
                    position.avg_price,
                    position.pnl,
                    position.value,
                    position.position_change,
                    position.pnl_change,
                )
                for position in positions
            ],
        )

    # Incremental helpers -------------------------------------------------
    def latest_position(self, trader_id: str, market_id: str) -> tuple[float | None, float | None]:
        """Return the latest quantity and pnl for a trader/market."""

        row = self.conn.execute(
            """
            SELECT quantity, pnl
            FROM trader_positions
            WHERE trader_id = ? AND market_id = ?
            ORDER BY observed_at DESC
            LIMIT 1
            """,
            [trader_id, market_id],
        ).fetchone()
        if row:
            return row[0], row[1]
        return None, None


class PolymarketIncrementalFetcher:
    """Fetch Polymarket public data and persist raw + normalized snapshots."""

    def __init__(
        self,
        *,
        db_path: str | Path = "./data/polymarket.duckdb",
        leaderboard_window: LeaderboardWindow = "all",
        leaderboard_metric: LeaderboardMetric = "profit",
        leaderboard_tag: str | None = None,
    ):
        self.store = PolymarketDuckDBStore(db_path)
        self.leaderboard_window = leaderboard_window
        self.leaderboard_metric = leaderboard_metric
        self.leaderboard_tag = leaderboard_tag

    async def fetch_and_store(self) -> None:
        """Fetch leaderboard, markets, order books, and positions."""

        observed_at = datetime.now(timezone.utc)
        async with PolymarketClient() as client:
            leaderboard_raw = await client.fetch_leaderboard(
                window=self.leaderboard_window,
                metric=self.leaderboard_metric,
                tag=self.leaderboard_tag,
            )
            self.store.insert_raw_leaderboard(
                observed_at, self.leaderboard_window, self.leaderboard_metric, leaderboard_raw
            )
            leaderboard_entries = [
                self._normalize_leaderboard_entry(entry, observed_at)
                for entry in leaderboard_raw
            ]
            self.store.insert_leaderboard_entries(leaderboard_entries)

            markets = await client.fetch_markets()
            self.store.insert_raw_markets(observed_at, markets)
            market_snapshots = [self._normalize_market(market, observed_at) for market in markets]
            self.store.insert_market_snapshots(market_snapshots)

            orderbooks: list[OrderBookSnapshot] = []
            for market in markets:
                token_ids = self._collect_token_ids(market)
                for token_id in token_ids:
                    try:
                        ob = await client.fetch_order_book(token_id)
                    except httpx.HTTPError as exc:  # pragma: no cover - network variability
                        logger.warning("Orderbook fetch failed for %s: %s", token_id, exc)
                        continue
                    self.store.insert_raw_orderbook(observed_at, market.get("id", ""), token_id, ob)
                    orderbooks.append(self._normalize_orderbook(market, token_id, ob, observed_at))
            if orderbooks:
                self.store.insert_orderbook_snapshots(orderbooks)

            trader_positions: list[TraderPosition] = []
            for entry in leaderboard_entries:
                try:
                    positions_raw = await client.fetch_trader_positions(entry.trader_id)
                except httpx.HTTPError as exc:  # pragma: no cover - network variability
                    logger.warning("Position fetch failed for %s: %s", entry.trader_id, exc)
                    continue
                self.store.insert_raw_positions(observed_at, entry.trader_id, positions_raw)
                trader_positions.extend(
                    self._normalize_positions(entry.trader_id, positions_raw, observed_at)
                )
            if trader_positions:
                self.store.insert_trader_positions(trader_positions)

    # Normalization helpers ----------------------------------------------
    def _normalize_leaderboard_entry(
        self, raw: dict[str, Any], observed_at: datetime
    ) -> LeaderboardEntry:
        hit_rate = raw.get("hitRates") or raw.get("odds_buckets") or {}
        username = raw.get("username")
        trader_id = raw.get("address") or raw.get("user") or ""
        return LeaderboardEntry(
            trader_id=trader_id,
            username=username,
            pnl=_to_float(raw.get("pnl")),
            roe=_to_float(raw.get("roe")),
            win_rate=_to_float(raw.get("winRate")),
            hit_rate_buckets=hit_rate,
            window=self.leaderboard_window,
            metric=self.leaderboard_metric,
            observed_at=observed_at,
        )

    def _normalize_market(self, raw: dict[str, Any], observed_at: datetime) -> MarketSnapshot:
        best_bid = _to_float(raw.get("bestBid"))
        best_ask = _to_float(raw.get("bestAsk"))
        spread_bps = None
        if best_bid is not None and best_ask is not None and best_bid > 0:
            spread_bps = (best_ask - best_bid) / best_bid * 10_000

        expiry = _parse_datetime(raw.get("endDate") or raw.get("expiry"))
        return MarketSnapshot(
            market_id=str(raw.get("id", "")),
            question=raw.get("question") or raw.get("title") or "",
            status=raw.get("status"),
            fee=_to_float(raw.get("fee")),
            liquidity=_to_float(raw.get("liquidity")),
            spread_bps=spread_bps,
            volume_24h=_to_float(raw.get("volume24h") or raw.get("volume")),
            expiry=expiry,
            observed_at=observed_at,
        )

    def _normalize_orderbook(
        self, market: dict[str, Any], token_id: str, raw: dict[str, Any], observed_at: datetime
    ) -> OrderBookSnapshot:
        bids = raw.get("bids") or []
        asks = raw.get("asks") or []
        best_bid = _to_float(bids[0].get("price")) if bids else None
        best_ask = _to_float(asks[0].get("price")) if asks else None
        bid_depth = sum(_to_float(bid.get("size")) or 0.0 for bid in bids)
        ask_depth = sum(_to_float(ask.get("size")) or 0.0 for ask in asks)

        spread_bps = None
        if best_bid is not None and best_ask is not None and best_bid > 0:
            spread_bps = (best_ask - best_bid) / best_bid * 10_000

        return OrderBookSnapshot(
            market_id=str(market.get("id", "")),
            token_id=token_id,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            spread_bps=spread_bps,
            observed_at=observed_at,
        )

    def _normalize_positions(
        self, trader_id: str, positions_raw: list[dict[str, Any]], observed_at: datetime
    ) -> list[TraderPosition]:
        normalized: list[TraderPosition] = []
        for position in positions_raw:
            market_id = str(position.get("market_id") or position.get("marketId") or "")
            previous_qty, previous_pnl = self.store.latest_position(trader_id, market_id)
            quantity = _to_float(position.get("quantity") or position.get("balance"))
            pnl = _to_float(position.get("pnl"))

            normalized.append(
                TraderPosition(
                    trader_id=trader_id,
                    market_id=market_id,
                    outcome=position.get("outcome") or position.get("side"),
                    quantity=quantity,
                    avg_price=_to_float(position.get("avgPrice") or position.get("price")),
                    pnl=pnl,
                    value=_to_float(position.get("value")),
                    position_change=_delta(quantity, previous_qty),
                    pnl_change=_delta(pnl, previous_pnl),
                    observed_at=observed_at,
                )
            )
        return normalized

    @staticmethod
    def _collect_token_ids(market: dict[str, Any]) -> list[str]:
        token_ids: list[str] = []
        outcomes = market.get("outcomes") or []
        for outcome in outcomes:
            token_id = outcome.get("tokenId") or outcome.get("token_id")
            if token_id:
                token_ids.append(str(token_id))
        return token_ids


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


async def main() -> None:  # pragma: no cover - convenience entry point
    """Small CLI helper for manual testing."""

    fetcher = PolymarketIncrementalFetcher()
    await fetcher.fetch_and_store()
    logger.info("Completed Polymarket incremental fetch")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
