"""Base client interface for prediction market venues."""

from abc import ABC, abstractmethod
from typing import Any

from . import Market, OrderRequest, OrderResult, Position


class BaseMarketClient(ABC):
    """
    Abstract base class for prediction market venue clients.
    
    All venue clients must implement this interface.
    """

    @abstractmethod
    async def list_markets(
        self,
        category: str | None = None,
        min_volume: float | None = None,
        limit: int | None = None,
    ) -> list[Market]:
        """
        List active markets on this venue.
        
        Args:
            category: Filter by category
            min_volume: Minimum 24h volume in USD
            limit: Maximum number of markets to return
            
        Returns:
            List of Market objects
        """
        pass

    @abstractmethod
    async def get_market(self, market_id: str) -> Market | None:
        """
        Get details for a specific market.
        
        Args:
            market_id: Market identifier
            
        Returns:
            Market object or None if not found
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get current open positions.
        
        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place an order.
        
        Args:
            request: Order request details
            
        Returns:
            Order result
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if successfully cancelled
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close client connections and cleanup resources."""
        pass
