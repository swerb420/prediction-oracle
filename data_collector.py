"""Utility helpers to backfill historical data for backtesting."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Iterable

import requests

from prediction_oracle.markets import Venue
from prediction_oracle.markets.router import MarketRouter
from prediction_oracle.storage import MarketSnapshot, create_tables, get_session

logger = logging.getLogger(__name__)


def fetch_manifold(market_id):
    url = f"https://api.manifold.markets/v0/market/{market_id}"
    return requests.get(url, timeout=10).json()


def fetch_polymarket(market_id):
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    return requests.get(url, timeout=10).json()


def fetch_kalshi(market_id):
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}"
    return requests.get(url, timeout=10).json()


def fetch_espn(league, game_id):
    sport_map = {
        "nfl": "football",
        "nba": "basketball",
        "nhl": "hockey",
        "mlb": "baseball",
    }
    sport = sport_map.get(league.lower())
    if not sport:
        return None
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={game_id}"
    )
    return requests.get(url, timeout=15).json()


async def _persist_snapshots(markets: Iterable, snapshot_time: datetime) -> None:
    """Write a batch of markets into MarketSnapshot rows."""
    async with get_session() as session:
        for market in markets:
            prices = {o.id: o.price for o in market.outcomes}
            volumes = {o.id: getattr(o, "volume_24h", None) for o in market.outcomes}

            snapshot = MarketSnapshot(
                venue=market.venue.value,
                market_id=market.market_id,
                snapshot_time=snapshot_time,
                question=market.question,
                prices_json=prices,
                volume_json=volumes,
                metadata_json={
                    "rules": market.rules,
                    "category": market.category,
                    "close_time": market.close_time.isoformat(),
                    "tags": market.tags,
                    "volume_24h": market.volume_24h,
                },
            )
            session.add(snapshot)


async def backfill_history(days: int = 7, page_size: int = 50, mock_mode: bool = False):
    """Pull recent markets from Kalshi and Polymarket and store snapshots.

    Args:
        days: How many days back to request (best-effort depending on venue pagination).
        page_size: Pagination chunk size per API call.
        mock_mode: Use mock routers instead of hitting real APIs.
    """

    await create_tables()
    router = MarketRouter(mock_mode=mock_mode)

    cutoff = datetime.utcnow() - timedelta(days=days)
    snapshot_time = datetime.utcnow()
    all_markets = []

    async def _drain_client(venue: Venue):
        markets: list = []
        client = router.get_client(venue)
        page = 0
        while True:
            try:
                page += 1
                market_batch = await client.list_markets(limit=page_size)
                if not market_batch:
                    break
                markets.extend([m for m in market_batch if m.close_time >= cutoff])
                if len(market_batch) < page_size:
                    break
            except Exception as exc:
                logger.warning("%s pagination halted: %s", venue.value, exc)
                break
        return markets

    tasks = [_drain_client(Venue.KALSHI), _drain_client(Venue.POLYMARKET)]
    for result in await asyncio.gather(*tasks):
        all_markets.extend(result)

    if not all_markets:
        logger.warning("No markets fetched for backfill")
        return

    await _persist_snapshots(all_markets, snapshot_time)
    logger.info("Stored %s historical snapshots", len(all_markets))


__all__ = [
    "backfill_history",
    "fetch_manifold",
    "fetch_polymarket",
    "fetch_kalshi",
    "fetch_espn",
]
