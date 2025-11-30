"""
Real Polymarket client using their public APIs.
No API key needed for market data!
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from . import Market, OrderRequest, OrderResult, OrderStatus, Outcome, Position, Venue
from .base_client import BaseMarketClient

logger = logging.getLogger(__name__)


class RealPolymarketClient(BaseMarketClient):
    """
    Client for Polymarket using their public APIs.
    Market data is completely free - no API key needed!
    """
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "PredictionOracle/2.0"}
        )
    
    async def list_markets(
        self,
        category: str | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[Market]:
        """Fetch real markets from Polymarket."""
        if self.mock_mode:
            return self._get_mock_markets(limit or 10)
        
        markets = []
        
        # Try multiple endpoints for better coverage
        # Request more to filter for quality
        request_limit = min((limit or 50) * 3, 200)
        endpoints = [
            (f"{self.GAMMA_API}/markets", {"closed": "false", "active": "true", "limit": request_limit}),
        ]
        
        for url, params in endpoints:
            try:
                resp = await self.client.get(url, params=params)
                
                if resp.status_code != 200:
                    logger.debug(f"{url} returned {resp.status_code}")
                    continue
                
                data = resp.json()
                
                # Handle different response formats
                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict) and "data" in data:
                    items = data["data"]
                elif isinstance(data, dict) and "markets" in data:
                    items = data["markets"]
                
                for item in items:
                    try:
                        # Some endpoints nest markets in events
                        if "markets" in item:
                            for sub_market in item["markets"]:
                                market = self._parse_market(sub_market)
                                if market and market.market_id not in [m.market_id for m in markets]:
                                    markets.append(market)
                        else:
                            market = self._parse_market(item)
                            if market and market.market_id not in [m.market_id for m in markets]:
                                markets.append(market)
                    except Exception as e:
                        logger.debug(f"Failed to parse: {e}")
                        continue
                
            except Exception as e:
                logger.debug(f"Endpoint {url} error: {e}")
                continue
        
        # Apply filters
        filtered = []
        for market in markets:
            if min_volume and (market.volume_24h or 0) < min_volume:
                continue
            if category and category.lower() not in market.category.lower():
                continue
            filtered.append(market)
        
        if filtered:
            logger.info(f"Fetched {len(filtered)} real Polymarket markets")
            return filtered[:limit] if limit else filtered
        else:
            logger.warning("No real markets, falling back to mock")
            return self._get_mock_markets(limit or 10)
    
    def _parse_market(self, data: dict) -> Market | None:
        """Parse Polymarket API response into Market object."""
        try:
            # Skip markets without proper prices - these are inactive
            outcome_prices = data.get("outcomePrices", "")
            if not outcome_prices or outcome_prices == "[]":
                return None
            
            # Parse prices
            if isinstance(outcome_prices, str) and outcome_prices:
                # Format: '["0.45", "0.55"]' or similar
                cleaned = outcome_prices.strip("[]").replace('"', '').replace("'", "")
                prices = [float(p.strip()) for p in cleaned.split(",") if p.strip()]
            elif isinstance(outcome_prices, list):
                prices = [float(p) for p in outcome_prices]
            else:
                return None  # Skip markets without prices
            
            # Skip if prices are all defaults (no trading)
            if len(prices) < 2 or (prices[0] == 0.5 and prices[1] == 0.5):
                # Check volume - skip if no volume
                if not data.get("volume24hr") or float(data.get("volume24hr", 0)) < 100:
                    return None
            
            # Get outcomes
            outcomes_data = data.get("outcomes", ["Yes", "No"])
            if isinstance(outcomes_data, str):
                outcomes_data = outcomes_data.strip("[]").replace('"', '').split(",")
            
            outcomes = []
            for i, label in enumerate(outcomes_data):
                price = prices[i] if i < len(prices) else 0.5
                # Ensure price is valid
                price = max(0.01, min(0.99, price))
                
                outcomes.append(Outcome(
                    id=f"{data.get('id', 'unknown')}_{i}",
                    label=label.strip() if isinstance(label, str) else str(label),
                    price=price,
                    volume_24h=float(data.get("volume24hr", 0) or 0) / len(outcomes_data),
                    liquidity=float(data.get("liquidity", 0) or 0) / len(outcomes_data),
                ))
            
            # Parse close time
            end_date = data.get("endDate") or data.get("end_date_iso")
            if end_date:
                try:
                    if isinstance(end_date, str):
                        close_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    else:
                        close_time = datetime.now(timezone.utc).replace(hour=23, minute=59)
                except Exception:
                    close_time = datetime.now(timezone.utc).replace(hour=23, minute=59)
            else:
                close_time = datetime.now(timezone.utc).replace(hour=23, minute=59)
            
            return Market(
                venue=Venue.POLYMARKET,
                market_id=str(data.get("id", data.get("condition_id", "unknown"))),
                question=data.get("question", "Unknown market"),
                rules=data.get("description", "")[:500] if data.get("description") else "",
                category=data.get("category", "Other") or "Other",
                close_time=close_time,
                outcomes=outcomes,
                volume_24h=float(data.get("volume24hr", 0) or 0),
                tags=data.get("tags", []) if isinstance(data.get("tags"), list) else [],
            )
            
        except Exception as e:
            logger.debug(f"Market parse error: {e}")
            return None
    
    async def get_market(self, market_id: str) -> Market | None:
        """Get a specific market by ID."""
        if self.mock_mode:
            return self._get_mock_markets(1)[0]
        
        try:
            resp = await self.client.get(f"{self.GAMMA_API}/markets/{market_id}")
            if resp.status_code == 200:
                return self._parse_market(resp.json())
        except Exception as e:
            logger.warning(f"Failed to get market {market_id}: {e}")
        
        return None
    
    async def get_orderbook(self, token_id: str) -> dict | None:
        """Get order book for a token (outcome)."""
        try:
            resp = await self.client.get(
                f"{self.CLOB_API}/book",
                params={"token_id": token_id}
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Orderbook error: {e}")
        return None
    
    async def get_positions(self) -> list[Position]:
        """Get positions - requires authentication."""
        return []  # Would need API key for this
    
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place order - requires authentication."""
        return OrderResult(
            order_id="NOT_IMPLEMENTED",
            status=OrderStatus.FAILED,
            message="Real trading requires API authentication"
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order - requires authentication."""
        return False
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
    
    def _get_mock_markets(self, count: int) -> list[Market]:
        """Generate mock markets for testing."""
        now = datetime.now(timezone.utc)
        
        return [
            Market(
                venue=Venue.POLYMARKET,
                market_id=f"POLY-MOCK-{i}",
                question=f"Will crypto event {i} occur?",
                rules=f"Market resolves YES if event {i} occurs.",
                category="Crypto",
                close_time=now.replace(hour=23, minute=59),
                outcomes=[
                    Outcome(
                        id=f"YES-{i}",
                        label="Yes",
                        price=min(0.90, 0.25 + (i % 8) * 0.1),
                        volume_24h=5000.0 + (i * 500),
                        liquidity=2000.0,
                    ),
                    Outcome(
                        id=f"NO-{i}",
                        label="No",
                        price=max(0.10, 0.75 - (i % 8) * 0.1),
                        volume_24h=5000.0 + (i * 500),
                        liquidity=2000.0,
                    ),
                ],
                volume_24h=10000.0 + (i * 1000),
                tags=["mock"],
            )
            for i in range(count)
        ]
