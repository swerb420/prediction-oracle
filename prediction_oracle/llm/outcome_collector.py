"""
Outcome Collector - Tracks 15M markets and records actual outcomes.

NO FAKE DATA. This collects real outcomes from Polymarket:
1. Discovers current/recent 15M markets
2. Polls for market resolution
3. Records actual UP/DOWN outcome + crypto price movement
4. Updates predictions with actual outcomes

This builds the labeled dataset for ML training.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

from real_data_store import (
    get_store, MarketSnapshot, MarketOutcome, RealDataStore
)

logger = logging.getLogger(__name__)

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

# Polymarket APIs
POLYMARKET_BASE = "https://polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"


class OutcomeCollector:
    """
    Collects real market outcomes from Polymarket 15M markets.
    
    Usage:
        collector = OutcomeCollector()
        await collector.collect_cycle()  # One cycle of collection
        await collector.run_continuous()  # Run forever, collecting data
    """
    
    def __init__(self, store: Optional[RealDataStore] = None):
        self.store = store or get_store()
        self.client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
        
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def _ensure_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 15M Market Discovery - Improved approach
    # ─────────────────────────────────────────────────────────────────────────
    
    async def discover_15m_markets(self) -> dict[str, str]:
        """
        Discover current 15M market event slugs from the crypto/15M page.
        
        Returns dict of symbol -> event_slug
        """
        await self._ensure_client()
        
        try:
            resp = await self.client.get(
                f"{POLYMARKET_BASE}/crypto/15M",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            text = resp.text
            
            discovered = {}
            
            # Find event slugs in the page for each symbol
            patterns = {
                "BTC": r'btc-updown-15m-\d+',
                "ETH": r'eth-updown-15m-\d+',
                "SOL": r'sol-updown-15m-\d+',
                "XRP": r'xrp-updown-15m-\d+',
            }
            
            for symbol, pattern in patterns.items():
                match = re.search(pattern, text.lower())
                if match:
                    slug = match.group(0)
                    discovered[symbol] = slug
                    logger.debug(f"Discovered {symbol} 15M market: {slug}")
            
            logger.info(f"Discovered {len(discovered)} 15M markets")
            return discovered
            
        except Exception as e:
            logger.error(f"Failed to discover markets: {e}")
            return {}
    
    async def fetch_market_from_page(self, symbol: str, slug: str) -> Optional[dict]:
        """
        Fetch market data by scraping the event page.
        """
        await self._ensure_client()
        
        try:
            url = f"{POLYMARKET_BASE}/event/{slug}"
            resp = await self.client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            
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
                if len(query_key) >= 2 and "updown-15m" in str(query_key[1]).lower():
                    event_data = q.get("state", {}).get("data", {})
                    markets = event_data.get("markets", [])
                    
                    if markets:
                        m = markets[0]
                        prices = m.get("outcomePrices", ["0.5", "0.5"])
                        
                        yes_price = float(prices[0]) if prices else 0.5
                        no_price = float(prices[1]) if len(prices) > 1 else 0.5
                        
                        return {
                            "symbol": symbol,
                            "slug": slug,
                            "condition_id": m.get("conditionId", ""),
                            "question": m.get("question", ""),
                            "clob_token_ids": m.get("clobTokenIds", []),
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "active": m.get("active", False),
                            "volume": float(m.get("volume", 0)),
                            "liquidity": float(m.get("liquidity", 0)),
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching market {slug}: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Price Fetching
    # ─────────────────────────────────────────────────────────────────────────
    
    async def get_clob_prices(self, token_id: str) -> dict:
        """Get current CLOB orderbook prices."""
        await self._ensure_client()
        
        try:
            resp = await self.client.get(f"{CLOB_BASE}/book?token_id={token_id}")
            resp.raise_for_status()
            data = resp.json()
            
            # Best bid/ask
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            
            return {
                "yes_price": (best_bid + best_ask) / 2,
                "bid": best_bid,
                "ask": best_ask,
                "spread": best_ask - best_bid,
            }
        except Exception as e:
            logger.error(f"Failed to get CLOB prices: {e}")
            return {"yes_price": 0.5, "bid": 0.5, "ask": 0.5, "spread": 0.0}
    
    async def get_crypto_price(self, symbol: str) -> float:
        """Get current crypto price from Binance."""
        await self._ensure_client()
        
        try:
            pair = f"{symbol.upper()}USDT"
            resp = await self.client.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
            )
            resp.raise_for_status()
            return float(resp.json()["price"])
        except Exception as e:
            logger.error(f"Failed to get {symbol} price: {e}")
            return 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Snapshot Collection
    # ─────────────────────────────────────────────────────────────────────────
    
    async def collect_snapshot(self, symbol: str, slug: str) -> Optional[MarketSnapshot]:
        """Collect a snapshot of current market state."""
        market = await self.fetch_market_from_page(symbol, slug)
        if not market:
            logger.warning(f"No market data found for {symbol}")
            return None
        
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=market.get("question", ""),
            yes_price=market["yes_price"],
            no_price=market["no_price"],
            market_direction="UP" if market["yes_price"] > 0.5 else "DOWN",
            condition_id=market.get("condition_id", ""),
            event_slug=slug,
            volume=market.get("volume", 0),
            liquidity=market.get("liquidity", 0),
        )
        
        # Save to store
        is_new = self.store.save_snapshot(snapshot)
        if is_new:
            logger.info(f"Saved snapshot: {symbol} YES={market['yes_price']:.3f}")
        
        return snapshot
    
    async def collect_all_snapshots(self) -> list[MarketSnapshot]:
        """Collect snapshots for all symbols."""
        # First discover current market slugs
        slugs = await self.discover_15m_markets()
        
        if not slugs:
            logger.warning("No 15M markets discovered")
            return []
        
        snapshots = []
        
        for symbol in SYMBOLS:
            slug = slugs.get(symbol)
            if not slug:
                logger.warning(f"No slug found for {symbol}")
                continue
            
            snapshot = await self.collect_snapshot(symbol, slug)
            if snapshot:
                snapshots.append(snapshot)
        
        return snapshots
    
    # ─────────────────────────────────────────────────────────────────────────
    # Outcome Collection
    # ─────────────────────────────────────────────────────────────────────────
    
    async def check_resolved_markets(self) -> list[MarketOutcome]:
        """
        Check for resolved markets and record outcomes.
        
        Looks at recent snapshots and checks if their markets have resolved.
        """
        await self._ensure_client()
        outcomes = []
        
        # Get recent snapshots
        snapshots = self.store.get_snapshots(hours=2)
        
        # Group by event_slug
        slugs_seen = set()
        
        for snap in snapshots:
            slug = snap.get("event_slug", "")
            if not slug or slug in slugs_seen:
                continue
            slugs_seen.add(slug)
            
            # Check if this market resolved
            outcome = await self.check_market_resolution(slug, snap)
            if outcome:
                outcomes.append(outcome)
        
        return outcomes
    
    async def check_market_resolution(
        self, 
        event_slug: str, 
        snapshot: dict
    ) -> Optional[MarketOutcome]:
        """Check if a specific market has resolved."""
        await self._ensure_client()
        
        try:
            # Get market from Gamma API
            resp = await self.client.get(
                f"{GAMMA_BASE}/events",
                params={"slug": event_slug}
            )
            
            if resp.status_code != 200:
                return None
            
            events = resp.json()
            if not events:
                return None
            
            event = events[0]
            markets = event.get("markets", [])
            if not markets:
                return None
            
            market = markets[0]
            
            # Check if resolved
            if not market.get("resolutionSource"):
                # Not resolved yet
                return None
            
            # Get outcome
            outcome_str = market.get("outcome", "").upper()
            if "YES" in outcome_str or "UP" in outcome_str:
                actual = "UP"
            elif "NO" in outcome_str or "DOWN" in outcome_str:
                actual = "DOWN"
            else:
                logger.warning(f"Unknown outcome: {outcome_str}")
                return None
            
            # Create outcome record
            symbol = snapshot.get("symbol", "BTC")
            
            outcome = MarketOutcome(
                symbol=symbol,
                event_slug=event_slug,
                start_time=snapshot.get("timestamp", ""),
                end_time=datetime.now(timezone.utc).isoformat(),
                actual_outcome=actual,
            )
            
            # Save
            self.store.save_outcome(outcome)
            
            # Update any predictions for this market
            self.store.update_prediction_outcome(event_slug, actual)
            
            logger.info(f"Recorded outcome: {symbol} {event_slug} -> {actual}")
            return outcome
            
        except Exception as e:
            logger.error(f"Error checking resolution for {event_slug}: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Continuous Collection
    # ─────────────────────────────────────────────────────────────────────────
    
    async def collect_cycle(self):
        """Run one collection cycle."""
        logger.info("Starting collection cycle")
        
        # Collect current snapshots
        snapshots = await self.collect_all_snapshots()
        logger.info(f"Collected {len(snapshots)} snapshots")
        
        # Check for resolved markets
        outcomes = await self.check_resolved_markets()
        logger.info(f"Recorded {len(outcomes)} outcomes")
        
        return {
            "snapshots": len(snapshots),
            "outcomes": len(outcomes),
            "store_summary": self.store.get_summary(),
        }
    
    async def run_continuous(self, interval_seconds: int = 60):
        """Run continuous collection."""
        logger.info(f"Starting continuous collection (interval: {interval_seconds}s)")
        
        while True:
            try:
                result = await self.collect_cycle()
                logger.info(f"Cycle complete: {result}")
                
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            
            await asyncio.sleep(interval_seconds)


async def main():
    """Test the collector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    async with OutcomeCollector() as collector:
        result = await collector.collect_cycle()
        print("\nCollection Result:")
        print(f"  Snapshots: {result['snapshots']}")
        print(f"  Outcomes: {result['outcomes']}")
        print(f"\nStore Summary:")
        summary = result['store_summary']
        print(f"  Total snapshots: {summary['snapshots']}")
        print(f"  Total outcomes: {summary['outcomes']}")
        print(f"  Labeled examples: {summary['labeled_examples']}")


if __name__ == "__main__":
    asyncio.run(main())
