"""Market router for unified access to multiple venues."""

import logging
from typing import Any

from . import Venue
from .base_client import BaseMarketClient
from .real_polymarket import RealPolymarketClient

logger = logging.getLogger(__name__)


class MarketRouter:
    """
    Routes market operations to the appropriate venue clients.
    
    ONLY uses venues with REAL market data available.
    - Polymarket: Free public API, no credentials needed ✅
    - Kalshi: DISABLED (requires API credentials we don't have)
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize market router with REAL clients only.
        
        Args:
            mock_mode: IGNORED - we always use real data
        """
        self._clients: dict[Venue, BaseMarketClient] = {}
        
        # ONLY Polymarket - it's free and has real data!
        self._clients[Venue.POLYMARKET] = RealPolymarketClient(mock_mode=False)
        
        # Kalshi disabled - we don't have API credentials
        # To enable Kalshi, add KALSHI_API_KEY and KALSHI_API_SECRET to .env
        
        logger.info("MarketRouter initialized - Polymarket ONLY (real data)")

    def get_client(self, venue: Venue) -> BaseMarketClient:
        """Get the client for a specific venue."""
        client = self._clients.get(venue)
        if not client:
            logger.warning(f"No client configured for venue: {venue} - skipping")
            return None
        return client
    
    def get_active_venues(self) -> list[Venue]:
        """Get list of venues with active clients."""
        return list(self._clients.keys())

    async def close_all(self) -> None:
        """Close all venue clients."""
        for venue, client in self._clients.items():
            try:
                await client.close()
                logger.info(f"Closed {venue} client")
            except Exception as e:
                logger.error(f"Error closing {venue} client: {e}")
