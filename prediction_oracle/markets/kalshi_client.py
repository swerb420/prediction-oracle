"""Kalshi market client."""

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import settings
from . import Market, OrderRequest, OrderResult, OrderStatus, Outcome, Position, Venue
from .base_client import BaseMarketClient

logger = logging.getLogger(__name__)


class KalshiClient(BaseMarketClient):
    """
    Client for Kalshi prediction market API.
    
    Implements authentication, market data fetching, and order placement.
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize Kalshi client.
        
        Args:
            mock_mode: If True, return mock data instead of real API calls
        """
        self.mock_mode = mock_mode
        self.base_url = settings.kalshi_base_url
        self.api_key = settings.kalshi_api_key
        self.api_secret = settings.kalshi_api_secret
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        self._token: str | None = None

    def _sign_request(self, method: str, path: str, body: str = "") -> dict[str, str]:
        """Generate authentication headers for Kalshi API."""
        timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        message = f"{timestamp}{method}{path}{body}"
        
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        
        return {
            "Authorization": f"HMAC {self.api_key}:{signature}",
            "X-Kalshi-Timestamp": timestamp,
        }

    async def _ensure_authenticated(self) -> None:
        """Ensure we have a valid authentication token."""
        if self._token:
            return
            
        # In mock mode, skip authentication
        if self.mock_mode:
            self._token = "mock_token"
            return
            
        # TODO: Implement actual Kalshi authentication flow
        logger.warning("Kalshi authentication not yet implemented")
        self._token = "placeholder"

    async def list_markets(
        self,
        category: str | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[Market]:
        """List active markets from Kalshi."""
        if self.mock_mode:
            return self._get_mock_markets(limit or 10)
        
        await self._ensure_authenticated()
        
        # TODO: Implement actual Kalshi API call
        logger.warning("Kalshi list_markets not fully implemented, returning empty list")
        return []

    async def get_market(self, market_id: str) -> Market | None:
        """Get details for a specific Kalshi market."""
        if self.mock_mode:
            markets = self._get_mock_markets(1)
            return markets[0] if markets else None
        
        await self._ensure_authenticated()
        
        # TODO: Implement actual Kalshi API call
        logger.warning(f"Kalshi get_market not fully implemented for {market_id}")
        return None

    async def get_positions(self) -> list[Position]:
        """Get current positions from Kalshi."""
        if self.mock_mode:
            return []
        
        await self._ensure_authenticated()
        
        # TODO: Implement actual Kalshi API call
        logger.warning("Kalshi get_positions not fully implemented")
        return []

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order on Kalshi."""
        if self.mock_mode:
            return OrderResult(
                order_id=f"MOCK_KALSHI_{datetime.now().timestamp()}",
                status=OrderStatus.FILLED,
                filled_size=request.size_usd,
                avg_fill_price=request.limit_price or 0.5,
                message="Mock order filled",
            )
        
        await self._ensure_authenticated()
        
        # TODO: Implement actual Kalshi order placement
        logger.warning(f"Kalshi place_order not fully implemented for {request}")
        return OrderResult(
            order_id="placeholder",
            status=OrderStatus.PENDING,
            message="Not implemented",
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on Kalshi."""
        if self.mock_mode:
            return True
        
        await self._ensure_authenticated()
        
        # TODO: Implement actual Kalshi order cancellation
        logger.warning(f"Kalshi cancel_order not fully implemented for {order_id}")
        return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    def _get_mock_markets(self, count: int) -> list[Market]:
        """Generate mock Kalshi markets for testing."""
        now = datetime.now(timezone.utc)
        
        mock_markets = [
            Market(
                venue=Venue.KALSHI,
                market_id=f"KALSHI-MARKET-{i}",
                question=f"Will event {i} happen?",
                rules=f"Market resolves YES if event {i} occurs by close time.",
                category="Politics" if i % 2 == 0 else "Economics",
                close_time=now.replace(hour=23, minute=59),
                outcomes=[
                    Outcome(
                        id=f"YES-{i}",
                        label="Yes",
                        price=min(0.95, 0.30 + (i % 10) * 0.05),  # Keep between 0.30-0.75
                        volume_24h=1000.0 + (i * 100),
                        liquidity=500.0,
                    ),
                    Outcome(
                        id=f"NO-{i}",
                        label="No",
                        price=max(0.05, 0.70 - (i % 10) * 0.05),  # Complementary
                        volume_24h=1000.0 + (i * 100),
                        liquidity=500.0,
                    ),
                ],
                volume_24h=2000.0 + (i * 200),
                tags=["mock", "test"],
            )
            for i in range(count)
        ]
        
        return mock_markets
