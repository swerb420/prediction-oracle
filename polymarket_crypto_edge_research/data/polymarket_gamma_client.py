"""
Polymarket Gamma API client.
Fetches market metadata, conditions, and outcomes.
"""

import asyncio
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logging_utils import get_logger
from core.time_utils import now_utc, parse_iso_datetime
from .rate_limiter import AdaptiveRateLimiter
from .schemas import PolymarketMarket, PolymarketOutcome

logger = get_logger(__name__)


class PolymarketGammaClient:
    """
    Client for Polymarket Gamma API.
    Fetches market metadata, categories, and resolution status.
    """
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.polymarket_gamma_url
        self._client = httpx.AsyncClient(timeout=30.0)
        self._rate_limiter = AdaptiveRateLimiter(
            requests_per_minute=60,
            requests_per_day=5000
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False
    ) -> list[PolymarketMarket]:
        """
        Fetch markets from Gamma API.
        
        Args:
            limit: Max markets to return
            offset: Pagination offset
            active: Include active markets
            closed: Include closed markets
            order: Sort field (volume24hr, liquidity, endDate)
            ascending: Sort order
            
        Returns:
            List of PolymarketMarket objects
        """
        await self._rate_limiter.acquire()
        
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower()
        }
        
        resp = await self._client.get(
            f"{self.base_url}/markets",
            params=params
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        markets = []
        for m in resp.json():
            markets.append(self._parse_market(m))
        
        logger.debug(f"Fetched {len(markets)} markets from Gamma API")
        return markets
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_market(self, market_id: str) -> PolymarketMarket | None:
        """Fetch a single market by ID."""
        await self._rate_limiter.acquire()
        
        resp = await self._client.get(
            f"{self.base_url}/markets/{market_id}"
        )
        
        if resp.status_code == 404:
            return None
        
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        return self._parse_market(resp.json())
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def search_markets(
        self,
        query: str,
        limit: int = 50
    ) -> list[PolymarketMarket]:
        """Search markets by text query."""
        await self._rate_limiter.acquire()
        
        resp = await self._client.get(
            f"{self.base_url}/markets",
            params={"_q": query, "limit": limit}
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        markets = []
        for m in resp.json():
            markets.append(self._parse_market(m))
        
        return markets
    
    async def get_markets_by_category(
        self,
        category: str,
        limit: int = 100
    ) -> list[PolymarketMarket]:
        """Get markets in a specific category."""
        await self._rate_limiter.acquire()
        
        resp = await self._client.get(
            f"{self.base_url}/markets",
            params={"category": category, "limit": limit, "active": "true"}
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        return [self._parse_market(m) for m in resp.json()]
    
    async def get_crypto_markets(self, limit: int = 100) -> list[PolymarketMarket]:
        """Get crypto-related markets (BTC, ETH, SOL)."""
        keywords = ["Bitcoin", "BTC", "Ethereum", "ETH", "Solana", "SOL", "crypto"]
        
        all_markets = []
        seen_ids = set()
        
        for keyword in keywords:
            try:
                markets = await self.search_markets(keyword, limit=50)
                for m in markets:
                    if m.market_id not in seen_ids:
                        seen_ids.add(m.market_id)
                        all_markets.append(m)
            except Exception as e:
                logger.warning(f"Search for '{keyword}' failed: {e}")
        
        # Sort by volume
        all_markets.sort(key=lambda m: m.volume_24h, reverse=True)
        return all_markets[:limit]
    
    async def get_resolving_soon(
        self,
        within_minutes: int = 60,
        limit: int = 100
    ) -> list[PolymarketMarket]:
        """Get markets resolving within specified time window."""
        # Fetch active markets ordered by end date
        all_markets = await self.get_markets(
            limit=limit,
            active=True,
            order="endDate",
            ascending=True
        )
        
        # Filter by resolution time
        now = now_utc()
        resolving = []
        
        for market in all_markets:
            if market.end_date:
                mins_until = (market.end_date - now).total_seconds() / 60
                if 0 < mins_until <= within_minutes:
                    resolving.append(market)
        
        return resolving
    
    async def get_all_markets_paginated(
        self,
        max_markets: int = 1000
    ) -> list[PolymarketMarket]:
        """Fetch all markets with pagination."""
        all_markets = []
        offset = 0
        batch_size = 100
        
        while len(all_markets) < max_markets:
            batch = await self.get_markets(
                limit=batch_size,
                offset=offset,
                active=True
            )
            
            if not batch:
                break
            
            all_markets.extend(batch)
            offset += batch_size
            
            # Small delay between batches
            await asyncio.sleep(0.2)
        
        return all_markets[:max_markets]
    
    def _parse_market(self, data: dict[str, Any]) -> PolymarketMarket:
        """Parse API response into PolymarketMarket."""
        # Parse outcomes
        outcomes = []
        if "outcomes" in data:
            for i, name in enumerate(data.get("outcomes", [])):
                price = 0.5  # Default
                if "outcomePrices" in data:
                    prices = data["outcomePrices"]
                    if isinstance(prices, str):
                        import json
                        prices = json.loads(prices)
                    if i < len(prices):
                        price = float(prices[i])
                
                outcomes.append(PolymarketOutcome(
                    outcome_id=f"{data.get('id', '')}_{i}",
                    name=name,
                    price=price
                ))
        
        # Parse end date
        end_date = None
        if data.get("endDate"):
            try:
                end_date = parse_iso_datetime(data["endDate"])
            except Exception:
                pass
        
        # Parse resolution date
        resolution_date = None
        if data.get("resolutionDate"):
            try:
                resolution_date = parse_iso_datetime(data["resolutionDate"])
            except Exception:
                pass
        
        return PolymarketMarket(
            market_id=data.get("id", ""),
            condition_id=data.get("conditionId", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            end_date=end_date,
            resolution_date=resolution_date,
            outcomes=outcomes,
            volume_24h=float(data.get("volume24hr", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            is_active=data.get("active", True)
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


# Singleton instance
_gamma_client: PolymarketGammaClient | None = None


def get_gamma_client() -> PolymarketGammaClient:
    """Get or create global Gamma client."""
    global _gamma_client
    if _gamma_client is None:
        _gamma_client = PolymarketGammaClient()
    return _gamma_client
