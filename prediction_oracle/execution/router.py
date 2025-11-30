"""Execution router for placing and tracking orders."""

import logging
from typing import Literal

from ..markets import OrderRequest, OrderResult, OrderSide, OrderType
from ..markets.router import MarketRouter
from ..strategies import TradeDecision

logger = logging.getLogger(__name__)


class ExecutionRouter:
    """
    Routes trade decisions to market venues for execution.
    
    Handles research, paper, and live trading modes.
    """

    def __init__(
        self,
        market_router: MarketRouter,
        mode: Literal["research", "paper", "live"] = "research",
    ):
        """
        Initialize execution router.
        
        Args:
            market_router: Market router for venue access
            mode: Execution mode (research/paper/live)
        """
        self.market_router = market_router
        self.mode = mode
        
        # Paper trading simulation
        self.paper_fills: dict[str, OrderResult] = {}
        
        logger.info(f"ExecutionRouter initialized in {mode} mode")

    async def execute_decisions(
        self,
        decisions: list[TradeDecision],
    ) -> list[tuple[TradeDecision, OrderResult | None]]:
        """
        Execute approved trade decisions.
        
        Args:
            decisions: List of trade decisions to execute
            
        Returns:
            List of (decision, order_result) tuples
        """
        results = []
        
        for decision in decisions:
            if self.mode == "research":
                result = self._execute_research(decision)
            elif self.mode == "paper":
                result = await self._execute_paper(decision)
            else:  # live
                result = await self._execute_live(decision)
            
            results.append((decision, result))
        
        logger.info(
            f"Executed {len(decisions)} decisions in {self.mode} mode"
        )
        return results

    def _execute_research(self, decision: TradeDecision) -> None:
        """Log decision without executing (research mode)."""
        logger.info(
            f"[RESEARCH] {decision.strategy_name}: {decision.direction} "
            f"{decision.outcome_id} on {decision.market_id} "
            f"(${decision.size_usd:.2f}, edge: {decision.edge:+.3f})"
        )
        return None

    async def _execute_paper(self, decision: TradeDecision) -> OrderResult:
        """Simulate execution (paper trading mode)."""
        # Simulate fill at current price (simplified)
        fill_price = decision.implied_p
        
        order_result = OrderResult(
            order_id=f"PAPER_{decision.market_id}_{decision.outcome_id}",
            status="FILLED",
            filled_size=decision.size_usd,
            avg_fill_price=fill_price,
            message=f"Paper trade: {decision.rationale}",
        )
        
        self.paper_fills[order_result.order_id] = order_result
        
        logger.info(
            f"[PAPER] {decision.strategy_name}: {decision.direction} "
            f"{decision.outcome_id} @ ${fill_price:.3f} "
            f"(${decision.size_usd:.2f})"
        )
        
        return order_result

    async def _execute_live(self, decision: TradeDecision) -> OrderResult:
        """Execute real order (live trading mode)."""
        client = self.market_router.get_client(decision.venue)
        
        # Build order request
        order_request = OrderRequest(
            venue=decision.venue,
            market_id=decision.market_id,
            outcome_id=decision.outcome_id,
            side=OrderSide.BUY if decision.direction == "BUY" else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            size_usd=decision.size_usd,
            limit_price=decision.implied_p,  # Could adjust for execution
        )
        
        try:
            result = await client.place_order(order_request)
            
            logger.info(
                f"[LIVE] {decision.strategy_name}: Order {result.order_id} "
                f"{result.status} ({decision.market_id})"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"[LIVE] Failed to execute {decision.market_id}: {e}"
            )
            return OrderResult(
                order_id="",
                status="FAILED",
                message=str(e),
            )
