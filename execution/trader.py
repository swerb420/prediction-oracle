"""Execution layer for coordinated order placement with risk controls."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Iterable

from ..markets import OrderRequest, OrderResult, OrderSide, OrderStatus
from ..markets.router import MarketRouter

logger = logging.getLogger(__name__)


class ExecutionTrader:
    """Place paired orders with price caps and portfolio-level limits.

    The trader coordinates multi-leg orders (e.g., buy/sell on opposing
    outcomes) while respecting exposure limits and providing execution
    diagnostics such as slippage and post-trade PnL.
    """

    def __init__(
        self,
        market_router: MarketRouter,
        max_market_exposure: float,
        max_outcome_exposure: float,
        max_daily_exposure: float,
    ) -> None:
        self.market_router = market_router
        self.max_market_exposure = max_market_exposure
        self.max_outcome_exposure = max_outcome_exposure
        self.max_daily_exposure = max_daily_exposure

        self.market_exposure: dict[str, float] = {}
        self.outcome_exposure: dict[tuple[str, str], float] = {}
        self.daily_exposure: dict[str, float] = {}

        logger.info(
            "ExecutionTrader initialized (market=%.2f, outcome=%.2f, daily=%.2f)",
            max_market_exposure,
            max_outcome_exposure,
            max_daily_exposure,
        )

    async def place_paired_orders(
        self,
        order_a: OrderRequest,
        order_b: OrderRequest,
        price_caps: tuple[float | None, float | None] = (None, None),
        max_retries: int = 2,
        size_tightening: float = 0.5,
    ) -> list[tuple[OrderRequest, OrderResult]]:
        """Place two related orders with retries and price caps.

        Args:
            order_a: First order request.
            order_b: Second order request.
            price_caps: Optional per-leg price caps. For BUY orders the price
                will not exceed the cap; for SELL orders the price will not be
                lower than the cap.
            max_retries: Maximum number of retries if depth is insufficient.
            size_tightening: Multiplier applied to remaining size on each retry.

        Returns:
            List of (order_request, order_result) pairs including retries.
        """

        capped_orders = [
            self._apply_price_cap(deepcopy(order_a), price_caps[0]),
            self._apply_price_cap(deepcopy(order_b), price_caps[1]),
        ]

        if not self._check_limits(capped_orders):
            logger.warning("Paired orders blocked by portfolio limits")
            return []

        results: list[tuple[OrderRequest, OrderResult]] = []
        pending_orders = capped_orders

        for attempt in range(max_retries + 1):
            next_round: list[OrderRequest] = []

            for order in pending_orders:
                client = self.market_router.get_client(order.venue)
                result = await client.place_order(order)
                results.append((order, result))
                self._record_fill(order, result)
                self._log_execution(order, result, attempt)

                remaining_size = max(order.size_usd - result.filled_size, 0.0)
                if remaining_size > 0 and attempt < max_retries:
                    tightened_size = max(remaining_size * size_tightening, 0.0)
                    if tightened_size >= 1.0:
                        retry_order = deepcopy(order)
                        retry_order.size_usd = tightened_size
                        next_round.append(retry_order)
                        logger.info(
                            "Retrying %s/%s for remaining $%.2f (tightened to $%.2f)",
                            order.market_id,
                            order.outcome_id,
                            remaining_size,
                            tightened_size,
                        )

            if not next_round:
                break

            if not self._check_limits(next_round):
                logger.warning("Retries cancelled due to portfolio limits")
                break

            pending_orders = next_round

        return results

    def _apply_price_cap(
        self, order: OrderRequest, price_cap: float | None
    ) -> OrderRequest:
        if price_cap is None or order.limit_price is None:
            return order

        if order.side == OrderSide.BUY and order.limit_price > price_cap:
            logger.debug(
                "Adjusting BUY price for %s to cap %.4f (was %.4f)",
                order.market_id,
                price_cap,
                order.limit_price,
            )
            order.limit_price = price_cap
        elif order.side == OrderSide.SELL and order.limit_price < price_cap:
            logger.debug(
                "Adjusting SELL price for %s to floor %.4f (was %.4f)",
                order.market_id,
                price_cap,
                order.limit_price,
            )
            order.limit_price = price_cap

        return order

    def _check_limits(self, orders: Iterable[OrderRequest]) -> bool:
        """Ensure portfolio limits would not be breached by the orders."""

        day_key = datetime.utcnow().date().isoformat()
        current_daily = self.daily_exposure.get(day_key, 0.0)

        for order in orders:
            market_key = order.market_id
            outcome_key = (order.market_id, order.outcome_id)

            projected_market = self.market_exposure.get(market_key, 0.0) + order.size_usd
            projected_outcome = self.outcome_exposure.get(outcome_key, 0.0) + order.size_usd
            projected_daily = current_daily + order.size_usd

            if projected_market > self.max_market_exposure:
                logger.warning(
                    "Market exposure limit hit for %s: %.2f > %.2f",
                    market_key,
                    projected_market,
                    self.max_market_exposure,
                )
                return False

            if projected_outcome > self.max_outcome_exposure:
                logger.warning(
                    "Outcome exposure limit hit for %s/%s: %.2f > %.2f",
                    order.market_id,
                    order.outcome_id,
                    projected_outcome,
                    self.max_outcome_exposure,
                )
                return False

            if projected_daily > self.max_daily_exposure:
                logger.warning(
                    "Daily exposure limit hit for %s: %.2f > %.2f",
                    day_key,
                    projected_daily,
                    self.max_daily_exposure,
                )
                return False

        return True

    def _record_fill(self, order: OrderRequest, result: OrderResult) -> None:
        """Update exposure trackers based on fills."""

        filled = result.filled_size
        if filled <= 0:
            return

        market_key = order.market_id
        outcome_key = (order.market_id, order.outcome_id)
        day_key = datetime.utcnow().date().isoformat()

        self.market_exposure[market_key] = self.market_exposure.get(market_key, 0.0) + filled
        self.outcome_exposure[outcome_key] = self.outcome_exposure.get(outcome_key, 0.0) + filled
        self.daily_exposure[day_key] = self.daily_exposure.get(day_key, 0.0) + filled

    def _log_execution(
        self, order: OrderRequest, result: OrderResult, attempt: int
    ) -> None:
        """Emit detailed execution logs including slippage and PnL."""

        filled = result.filled_size
        if filled <= 0:
            logger.info(
                "Attempt %d: %s %s %s %s failed (%s)",
                attempt,
                order.side,
                order.market_id,
                order.outcome_id,
                f"${order.size_usd:.2f}",
                result.status,
            )
            return

        slippage = self._calculate_slippage(order, result)
        pnl = self._estimate_pnl(order, result)

        logger.info(
            "Attempt %d: %s %s/%s filled $%.2f @ %.4f (slippage: %.4f, PnL: %.2f)",
            attempt,
            order.side,
            order.market_id,
            order.outcome_id,
            filled,
            result.avg_fill_price or 0.0,
            slippage,
            pnl,
        )

        if result.status == OrderStatus.PARTIALLY_FILLED:
            logger.info(
                "Partial fill for %s/%s: filled $%.2f of $%.2f",
                order.market_id,
                order.outcome_id,
                filled,
                order.size_usd,
            )

    def _calculate_slippage(self, order: OrderRequest, result: OrderResult) -> float:
        if order.limit_price is None or result.avg_fill_price is None:
            return 0.0

        price_diff = result.avg_fill_price - order.limit_price
        return price_diff if order.side == OrderSide.BUY else -price_diff

    def _estimate_pnl(self, order: OrderRequest, result: OrderResult) -> float:
        """Estimate immediate PnL relative to the submitted limit price."""

        if order.limit_price is None or result.avg_fill_price is None:
            return 0.0

        price_delta = order.limit_price - result.avg_fill_price
        signed_delta = price_delta if order.side == OrderSide.BUY else -price_delta
        return signed_delta * result.filled_size
