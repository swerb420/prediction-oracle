"""Bankroll tracking and management."""

import logging
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BankrollState(BaseModel):
    """Current bankroll state."""

    total: float
    available: float
    allocated: float
    daily_pnl: float
    total_pnl: float


class BankrollManager:
    """
    Manages trading bankroll and tracks P&L.
    
    Tracks available capital, allocated positions, and performance metrics.
    """

    def __init__(self, initial_bankroll: float):
        """
        Initialize bankroll manager.
        
        Args:
            initial_bankroll: Starting bankroll in USD
        """
        self.initial = initial_bankroll
        self.current = initial_bankroll
        self.allocated = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        logger.info(f"BankrollManager initialized with ${initial_bankroll:.2f}")

    def get_state(self) -> BankrollState:
        """Get current bankroll state."""
        return BankrollState(
            total=self.current,
            available=self.current - self.allocated,
            allocated=self.allocated,
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl,
        )

    def allocate(self, amount: float) -> bool:
        """
        Allocate capital for a position.
        
        Args:
            amount: Amount to allocate
            
        Returns:
            True if allocation successful
        """
        available = self.current - self.allocated
        if amount > available:
            logger.warning(
                f"Insufficient funds: ${amount:.2f} requested, ${available:.2f} available"
            )
            return False
        
        self.allocated += amount
        logger.debug(f"Allocated ${amount:.2f}, total allocated: ${self.allocated:.2f}")
        return True

    def release(self, amount: float, pnl: float = 0.0) -> None:
        """
        Release allocated capital (position closed).
        
        Args:
            amount: Amount to release
            pnl: Profit/loss on the position
        """
        self.allocated = max(0, self.allocated - amount)
        self.current += pnl
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        logger.info(
            f"Released ${amount:.2f} with PnL ${pnl:+.2f}, "
            f"current bankroll: ${self.current:.2f}"
        )

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L counter (call at start of each day)."""
        self.daily_pnl = 0.0
        logger.debug("Daily PnL reset")

    def get_available(self) -> float:
        """Get available (unallocated) capital."""
        return self.current - self.allocated
