"""Risk limits and position management."""

import logging
from collections import defaultdict

from ..strategies import TradeDecision
from .bankroll import BankrollManager

logger = logging.getLogger(__name__)


class RejectionReason:
    """Reasons for rejecting a trade."""

    OVER_EXPOSURE = "OVER_EXPOSURE"
    DAILY_DD = "DAILY_DD"
    POSITION_TOO_LARGE = "POSITION_TOO_LARGE"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    MAX_POSITIONS = "MAX_POSITIONS"
    CORRELATED = "CORRELATED"


class RiskManager:
    """
    Enforces risk limits on trading decisions.
    
    Validates trades against bankroll, exposure, and correlation limits.
    """

    def __init__(self, config: dict, bankroll: BankrollManager):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration
            bankroll: Bankroll manager
        """
        self.config = config
        self.bankroll = bankroll
        
        # Track positions by venue
        self.venue_exposure: dict[str, float] = defaultdict(float)
        self.position_count = 0
        
        logger.info("RiskManager initialized")

    def validate_decisions(
        self,
        decisions: list[TradeDecision],
    ) -> list[tuple[TradeDecision, str | None]]:
        """
        Validate trade decisions against risk limits.
        
        Args:
            decisions: List of proposed trade decisions
            
        Returns:
            List of (decision, rejection_reason) tuples.
            rejection_reason is None if approved.
        """
        max_position_size_pct = self.config.get("max_position_size_pct", 0.01)
        max_venue_exposure_pct = self.config.get("max_venue_exposure_pct", 0.20)
        max_daily_drawdown_pct = self.config.get("max_daily_drawdown_pct", 0.05)
        max_open_positions = self.config.get("max_open_positions", 20)
        
        bankroll_state = self.bankroll.get_state()
        results = []
        
        # Check daily drawdown first
        if bankroll_state.daily_pnl < -bankroll_state.total * max_daily_drawdown_pct:
            logger.warning(
                f"Daily drawdown limit breached: "
                f"${bankroll_state.daily_pnl:.2f} / ${bankroll_state.total:.2f}"
            )
            return [(d, RejectionReason.DAILY_DD) for d in decisions]
        
        for decision in decisions:
            # Check max positions
            if self.position_count >= max_open_positions:
                results.append((decision, RejectionReason.MAX_POSITIONS))
                continue
            
            # Check position size
            max_position_size = bankroll_state.total * max_position_size_pct
            if decision.size_usd > max_position_size:
                results.append((decision, RejectionReason.POSITION_TOO_LARGE))
                continue
            
            # Check available funds
            if decision.size_usd > bankroll_state.available:
                results.append((decision, RejectionReason.INSUFFICIENT_FUNDS))
                continue
            
            # Check venue exposure
            venue_key = decision.venue.value
            current_venue_exposure = self.venue_exposure[venue_key]
            new_venue_exposure = current_venue_exposure + decision.size_usd
            max_venue_exposure = bankroll_state.total * max_venue_exposure_pct
            
            if new_venue_exposure > max_venue_exposure:
                results.append((decision, RejectionReason.OVER_EXPOSURE))
                continue
            
            # Approved
            results.append((decision, None))
        
        approved_count = sum(1 for _, reason in results if reason is None)
        logger.info(
            f"Risk validation: {approved_count}/{len(decisions)} decisions approved"
        )
        
        return results

    def record_position_opened(self, decision: TradeDecision) -> None:
        """Record that a position was opened."""
        venue_key = decision.venue.value
        self.venue_exposure[venue_key] += decision.size_usd
        self.position_count += 1
        
        logger.debug(
            f"Position opened: {venue_key} exposure now ${self.venue_exposure[venue_key]:.2f}"
        )

    def record_position_closed(self, decision: TradeDecision) -> None:
        """Record that a position was closed."""
        venue_key = decision.venue.value
        self.venue_exposure[venue_key] = max(
            0, self.venue_exposure[venue_key] - decision.size_usd
        )
        self.position_count = max(0, self.position_count - 1)
        
        logger.debug(
            f"Position closed: {venue_key} exposure now ${self.venue_exposure[venue_key]:.2f}"
        )
