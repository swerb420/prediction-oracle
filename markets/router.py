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

    def __init__(
        self,
        mock_mode: bool = False,
        extra_clients: dict[Venue, BaseMarketClient] | None = None,
    ):
        """
        Initialize market router.

        Args:
            mock_mode: If True, all clients will use mock data.
            extra_clients: Optional mapping of additional venue clients to register.
        """
        self.mock_mode = mock_mode
        self._clients: dict[Venue, BaseMarketClient] = {}

        # Initialize core clients
        self.register_client(Venue.KALSHI, KalshiClient(mock_mode=mock_mode))
        self.register_client(Venue.POLYMARKET, PolymarketClient(mock_mode=mock_mode))

        # Allow callers to bolt on custom venues (e.g., RainBet or others)
        if extra_clients:
            for venue, client in extra_clients.items():
                self.register_client(venue, client)

        logger.info(
            "MarketRouter initialized (mock_mode=%s, venues=%s)",
            mock_mode,
            list(self._clients.keys()),
        )

    def register_client(self, venue: Venue, client: BaseMarketClient) -> None:
        """Register a new venue client or replace an existing one.

        This makes it easy to plug in additional markets without modifying the
        router internals. Duplicate registrations overwrite the previous client
        and emit a debug log entry.
        """

        if not isinstance(client, BaseMarketClient):
            raise TypeError("client must implement BaseMarketClient")

        if venue in self._clients:
            logger.debug("Replacing existing client for venue %s", venue)

        self._clients[venue] = client
        logger.debug("Registered client for venue %s", venue)

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

    @property
    def supported_venues(self) -> list[Venue]:
        """Return the list of currently registered venues."""

        return list(self._clients.keys())

    async def close_all(self) -> None:
        """Close all venue clients."""
        for venue, client in self._clients.items():
            try:
                await client.close()
                logger.info(f"Closed {venue} client")
            except Exception as e:
                logger.error(f"Error closing {venue} client: {e}")
