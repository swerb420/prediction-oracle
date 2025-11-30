"""Market router for unified access to multiple venues."""

import logging
from typing import Any

from . import Venue
from .base_client import BaseMarketClient
from .kalshi_client import KalshiClient
from .polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)


class MarketRouter:
    """
    Routes market operations to the appropriate venue client.
    
    Provides a unified interface for interacting with multiple prediction markets.
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize market router.
        
        Args:
            mock_mode: If True, all clients will use mock data
        """
        self.mock_mode = mock_mode
        self._clients: dict[Venue, BaseMarketClient] = {}
        
        # Initialize clients
        self._clients[Venue.KALSHI] = KalshiClient(mock_mode=mock_mode)
        self._clients[Venue.POLYMARKET] = PolymarketClient(mock_mode=mock_mode)
        
        logger.info(f"MarketRouter initialized (mock_mode={mock_mode})")

    def get_client(self, venue: Venue) -> BaseMarketClient:
        """
        Get the client for a specific venue.
        
        Args:
            venue: The venue to get the client for
            
        Returns:
            The market client for that venue
            
        Raises:
            ValueError: If venue is not supported
        """
        client = self._clients.get(venue)
        if not client:
            raise ValueError(f"No client configured for venue: {venue}")
        return client

    async def close_all(self) -> None:
        """Close all venue clients."""
        for venue, client in self._clients.items():
            try:
                await client.close()
                logger.info(f"Closed {venue} client")
            except Exception as e:
                logger.error(f"Error closing {venue} client: {e}")
