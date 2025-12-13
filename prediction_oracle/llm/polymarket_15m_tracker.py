"""
Polymarket 15M Crypto Tracker - Real-time price monitoring.

Tracks the 4 crypto 15-minute up/down markets:
- BTC Up or Down - 15 minute
- ETH Up or Down - 15 minute  
- SOL Up or Down - 15 minute
- XRP Up or Down - 15 minute

Scrapes market data from Polymarket event pages since the 15M markets
aren't exposed through standard APIs.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Literal, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]

# Event slug patterns (timestamp changes with each 15-min window)
EVENT_SLUG_PATTERNS = {
    "BTC": "btc-updown-15m",
    "ETH": "eth-updown-15m",
    "SOL": "sol-updown-15m",
    "XRP": "xrp-updown-15m",
}


@dataclass
class Market15MData:
    """Data for a 15M crypto market."""

    symbol: CryptoSymbol
    condition_id: str
    question: str
    clob_token_ids: list[str]  # [Yes token, No token]
    yes_price: float  # 0-1 probability of UP
    no_price: float  # 0-1 probability of DOWN
    active: bool
    end_time: Optional[datetime] = None
    volume: float = 0.0
    liquidity: float = 0.0
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def market_direction(self) -> str:
        """Market consensus direction."""
        return "UP" if self.yes_price > 0.5 else "DOWN"

    @property
    def confidence(self) -> float:
        """Market confidence in direction."""
        return max(self.yes_price, self.no_price)

    @property
    def spread(self) -> float:
        """Implied bid-ask spread."""
        return abs(self.yes_price + self.no_price - 1.0)


@dataclass
class PriceUpdate:
    """Price update event."""

    symbol: CryptoSymbol
    old_yes: float
    new_yes: float
    change_pct: float
    direction: str  # "UP" or "DOWN"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Polymarket15MTracker:
    """
    Real-time tracker for Polymarket 15M crypto markets.

    Scrapes event pages to get market data since 15M markets
    aren't in the standard CLOB/Gamma APIs.
    """

    POLYMARKET_BASE = "https://polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(
        self,
        poll_interval: float = 2.0,
        timeout: float = 10.0,
    ):
        """
        Initialize tracker.

        Args:
            poll_interval: Seconds between price fetches
            timeout: HTTP request timeout
        """
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Market data
        self.markets: dict[CryptoSymbol, Market15MData] = {}
        self.price_history: dict[CryptoSymbol, list[tuple[datetime, float]]] = {
            "BTC": [],
            "ETH": [],
            "SOL": [],
            "XRP": [],
        }

        # Event slugs (discovered dynamically)
        self.event_slugs: dict[CryptoSymbol, str] = {}

        # Callbacks
        self._on_update: list[Callable[[Market15MData], None]] = []
        self._on_price_change: list[Callable[[PriceUpdate], None]] = []

        # State
        self._running = False

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args):
        self._running = False
        if self._client:
            await self._client.aclose()

    async def discover_current_markets(self) -> dict[CryptoSymbol, str]:
        """
        Discover the current 15M market event slugs.

        The slugs include a timestamp that changes every 15 minutes.
        We need to find the current active markets.

        Returns:
            Dict of symbol -> full event slug
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async with.")

        discovered = {}

        # Fetch the crypto 15M page to find current event slugs
        try:
            resp = await self._client.get(f"{self.POLYMARKET_BASE}/crypto/15M")
            if resp.status_code != 200:
                logger.error(f"Failed to fetch 15M page: {resp.status_code}")
                return discovered

            text = resp.text

            # Find event slugs in the page
            # Pattern: xxx-updown-15m-TIMESTAMP (case insensitive)
            for symbol, pattern in EVENT_SLUG_PATTERNS.items():
                # Direct match for the slug pattern
                regex = rf'({pattern.lower()}-\d+)'
                match = re.search(regex, text.lower())
                if match:
                    slug = match.group(1)
                    discovered[symbol] = slug
                    logger.info(f"Discovered {symbol} 15M market: {slug}")

            self.event_slugs = discovered
            return discovered

        except Exception as e:
            logger.error(f"Failed to discover markets: {e}")
            return discovered

    async def fetch_market_data(self, symbol: CryptoSymbol) -> Optional[Market15MData]:
        """
        Fetch current market data for a symbol by scraping the event page.

        Args:
            symbol: Crypto symbol (BTC, ETH, SOL, XRP)

        Returns:
            Market15MData or None
        """
        if not self._client:
            raise RuntimeError("Client not initialized")

        slug = self.event_slugs.get(symbol)
        if not slug:
            # Try to discover
            await self.discover_current_markets()
            slug = self.event_slugs.get(symbol)
            if not slug:
                logger.warning(f"No event slug found for {symbol}")
                return None

        try:
            url = f"{self.POLYMARKET_BASE}/event/{slug}"
            resp = await self._client.get(url)

            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {symbol} event: {resp.status_code}")
                return None

            # Extract __NEXT_DATA__
            match = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                resp.text,
                re.DOTALL,
            )
            if not match:
                logger.warning(f"No __NEXT_DATA__ found for {symbol}")
                return None

            data = json.loads(match.group(1))
            queries = (
                data.get("props", {})
                .get("pageProps", {})
                .get("dehydratedState", {})
                .get("queries", [])
            )

            # Find the event query
            for q in queries:
                query_key = q.get("queryKey", [])
                if len(query_key) >= 2 and "updown-15m" in str(query_key[1]):
                    event_data = q.get("state", {}).get("data", {})
                    markets = event_data.get("markets", [])

                    if markets:
                        m = markets[0]
                        prices = m.get("outcomePrices", ["0.5", "0.5"])

                        market_data = Market15MData(
                            symbol=symbol,
                            condition_id=m.get("conditionId", ""),
                            question=m.get("question", ""),
                            clob_token_ids=m.get("clobTokenIds", []),
                            yes_price=float(prices[0]) if prices else 0.5,
                            no_price=float(prices[1]) if len(prices) > 1 else 0.5,
                            active=m.get("active", False),
                            volume=float(m.get("volume", 0)),
                            liquidity=float(m.get("liquidity", 0)),
                        )

                        # Track price change
                        old_data = self.markets.get(symbol)
                        if old_data and abs(market_data.yes_price - old_data.yes_price) > 0.001:
                            change_pct = (
                                (market_data.yes_price - old_data.yes_price)
                                / old_data.yes_price
                                * 100
                                if old_data.yes_price > 0
                                else 0
                            )
                            update = PriceUpdate(
                                symbol=symbol,
                                old_yes=old_data.yes_price,
                                new_yes=market_data.yes_price,
                                change_pct=change_pct,
                                direction="UP" if change_pct > 0 else "DOWN",
                            )
                            for cb in self._on_price_change:
                                try:
                                    cb(update)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")

                        # Update stored data
                        self.markets[symbol] = market_data

                        # Add to history
                        self.price_history[symbol].append(
                            (datetime.now(timezone.utc), market_data.yes_price)
                        )
                        # Keep last 5 minutes
                        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
                        self.price_history[symbol] = [
                            (t, p) for t, p in self.price_history[symbol] if t > cutoff
                        ]

                        # Notify callbacks
                        for cb in self._on_update:
                            try:
                                cb(market_data)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                        return market_data

            return None

        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return None

    async def fetch_all_markets(self) -> dict[CryptoSymbol, Market15MData]:
        """Fetch data for all 4 crypto 15M markets."""
        if not self.event_slugs:
            await self.discover_current_markets()

        results = {}
        for symbol in ["BTC", "ETH", "SOL", "XRP"]:
            data = await self.fetch_market_data(symbol)
            if data:
                results[symbol] = data

        return results

    async def poll_prices_from_clob(self) -> dict[CryptoSymbol, float]:
        """
        Get real-time prices from CLOB API using token IDs.

        This is faster than scraping but requires token IDs to be known.
        """
        if not self._client:
            raise RuntimeError("Client not initialized")

        prices = {}

        for symbol, market in self.markets.items():
            if not market.clob_token_ids:
                continue

            try:
                # Get orderbook for the Yes token
                token_id = market.clob_token_ids[0]  # Yes token
                resp = await self._client.get(
                    f"{self.CLOB_API}/book",
                    params={"token_id": token_id},
                )

                if resp.status_code == 200:
                    book = resp.json()
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])

                    if bids and asks:
                        best_bid = float(bids[0]["price"])
                        best_ask = float(asks[0]["price"])
                        mid_price = (best_bid + best_ask) / 2
                        prices[symbol] = mid_price
                    elif bids:
                        prices[symbol] = float(bids[0]["price"])
                    elif asks:
                        prices[symbol] = float(asks[0]["price"])

            except Exception as e:
                logger.debug(f"CLOB price fetch failed for {symbol}: {e}")

        return prices

    def on_update(self, callback: Callable[[Market15MData], None]):
        """Register callback for market updates."""
        self._on_update.append(callback)

    def on_price_change(self, callback: Callable[[PriceUpdate], None]):
        """Register callback for price changes."""
        self._on_price_change.append(callback)

    async def run_continuous(self, duration_seconds: Optional[int] = None):
        """
        Run continuous polling loop.

        Args:
            duration_seconds: How long to run (None = forever)
        """
        self._running = True
        start = datetime.now(timezone.utc)

        # Initial discovery
        await self.discover_current_markets()

        while self._running:
            try:
                # Fetch all markets
                await self.fetch_all_markets()

                # Log current state
                for symbol, market in self.markets.items():
                    logger.info(
                        f"{symbol}: {market.yes_price:.1%} UP | {market.no_price:.1%} DOWN"
                    )

                # Check duration
                if duration_seconds:
                    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                    if elapsed >= duration_seconds:
                        break

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(self.poll_interval)

        self._running = False

    def get_momentum(self, symbol: CryptoSymbol, seconds: int = 30) -> float:
        """Get price momentum over last N seconds."""
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return 0.0

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        recent = [(t, p) for t, p in history if t > cutoff]

        if len(recent) < 2:
            return 0.0

        first = recent[0][1]
        last = recent[-1][1]
        return (last - first) / first * 100 if first > 0 else 0.0

    def get_volatility(self, symbol: CryptoSymbol, seconds: int = 60) -> float:
        """Get price volatility over last N seconds."""
        history = self.price_history.get(symbol, [])
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        recent = [p for t, p in history if t > cutoff]

        if len(recent) < 2:
            return 0.0

        import numpy as np

        return float(np.std(recent) / np.mean(recent) * 100) if np.mean(recent) > 0 else 0.0

    def get_summary(self) -> dict:
        """Get summary of all markets."""
        return {
            symbol: {
                "direction": m.market_direction,
                "confidence": f"{m.confidence:.1%}",
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "question": m.question[:50],
                "momentum_30s": f"{self.get_momentum(symbol, 30):.2f}%",
            }
            for symbol, m in self.markets.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / CLI
# ─────────────────────────────────────────────────────────────────────────────


async def demo():
    """Demo the 15M tracker."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("POLYMARKET 15M CRYPTO TRACKER")
    print("=" * 60)

    async with Polymarket15MTracker(poll_interval=2.0) as tracker:
        # Discover markets
        slugs = await tracker.discover_current_markets()
        print(f"\nDiscovered {len(slugs)} markets:")
        for symbol, slug in slugs.items():
            print(f"  {symbol}: {slug}")

        # Fetch current data
        print("\n" + "-" * 60)
        print("CURRENT PRICES")
        print("-" * 60)

        markets = await tracker.fetch_all_markets()
        for symbol, market in markets.items():
            print(f"\n{symbol}: {market.question}")
            print(f"  UP:   {market.yes_price:.1%}")
            print(f"  DOWN: {market.no_price:.1%}")
            print(f"  Market says: {market.market_direction} ({market.confidence:.1%} confidence)")

        # Try CLOB prices
        print("\n" + "-" * 60)
        print("CLOB ORDERBOOK PRICES")
        print("-" * 60)

        clob_prices = await tracker.poll_prices_from_clob()
        for symbol, price in clob_prices.items():
            print(f"  {symbol}: {price:.3f}")

        print("\n✅ Tracker working!")


if __name__ == "__main__":
    asyncio.run(demo())
