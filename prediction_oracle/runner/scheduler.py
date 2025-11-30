"""Main event loop and scheduler for the prediction oracle."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import yaml

from ..execution import ExecutionRouter
from ..llm import LLMOracle
from ..llm.enhanced_oracle import EnhancedOracle
from ..markets import Venue
from ..markets.router import MarketRouter
from ..risk import BankrollManager, RiskManager
from ..storage import Trade, create_tables, get_session
from ..strategies import ConservativeStrategy, LongshotStrategy
from ..strategies.base_strategy import EnhancedStrategy
from ..strategies.enhanced_conservative import EnhancedConservativeStrategy
from ..strategies.enhanced_longshot import EnhancedLongshotStrategy
from ..config import settings

logger = logging.getLogger(__name__)


class OracleScheduler:
    """
    Main scheduler that orchestrates the entire trading system.
    
    Coordinates market data fetching, LLM analysis, strategy execution,
    risk management, and order placement.
    """

    def __init__(
        self,
        config_path: str,
        mode: str = "research",
        mock_mode: bool = True,
    ):
        """
        Initialize oracle scheduler.
        
        Args:
            config_path: Path to YAML configuration file
            mode: Trading mode (research/paper/live)
            mock_mode: Use mock data for testing
        """
        self.mode = mode
        self.mock_mode = mock_mode
        
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"OracleScheduler initialized in {mode} mode (mock={mock_mode})")

    async def initialize(self):
        """Initialize all components."""
        # Database
        await create_tables()
        
        # Markets
        self.market_router = MarketRouter(mock_mode=self.mock_mode)
        
        # LLM Oracle
        if settings.enable_enhanced_strategies:
            self.oracle = EnhancedOracle(self.config)
        else:
            self.oracle = LLMOracle(self.config)
        
        # Strategies (use enhanced if enabled)
        self.strategies = []
        
        # Risk management first (strategies need it)
        self.bankroll = BankrollManager(settings.initial_bankroll)
        self.risk_manager = RiskManager(
            self.config.get("risk", {}),
            self.bankroll,
        )
        
        # Initialize strategies
        if self.config.get("strategies", {}).get("conservative", {}).get("enabled"):
            if settings.enable_enhanced_strategies:
                logger.info("Using EnhancedConservativeStrategy")
                self.strategies.append(
                    EnhancedConservativeStrategy(
                        self.config,
                        self.bankroll,
                        oracle=self.oracle if isinstance(self.oracle, EnhancedOracle) else None,
                    )
                )
            else:
                logger.info("Using basic ConservativeStrategy")
                self.strategies.append(
                    ConservativeStrategy(self.config["strategies"]["conservative"])
                )
        
        if self.config.get("strategies", {}).get("longshot", {}).get("enabled"):
            if settings.enable_enhanced_strategies:
                logger.info("Using EnhancedLongshotStrategy")
                self.strategies.append(
                    EnhancedLongshotStrategy(
                        self.config,
                        self.bankroll,
                        oracle=self.oracle if isinstance(self.oracle, EnhancedOracle) else None,
                    )
                )
            else:
                logger.info("Using basic LongshotStrategy")
                self.strategies.append(
                    LongshotStrategy(self.config["strategies"]["longshot"])
                )
        
        # Execution
        self.executor = ExecutionRouter(self.market_router, mode=self.mode)
        
        logger.info(
            f"Initialized with {len(self.strategies)} strategies, "
            f"bankroll ${self.bankroll.current:.2f}"
        )

    async def run_once(self):
        """Run one iteration of the main loop."""
        logger.info("=" * 80)
        logger.info(f"Starting scan iteration at {datetime.utcnow().isoformat()}")
        
        # 1. Fetch markets from all venues
        all_markets = []
        
        markets_config = self.config.get("markets", {})
        
        if markets_config.get("kalshi", {}).get("enabled", True):
            try:
                kalshi_client = self.market_router.get_client(Venue.KALSHI)
                kalshi_markets = await kalshi_client.list_markets(limit=20)
                all_markets.extend(kalshi_markets)
                logger.info(f"Fetched {len(kalshi_markets)} Kalshi markets")
            except Exception as e:
                logger.error(f"Error fetching Kalshi markets: {e}")
        
        if markets_config.get("polymarket", {}).get("enabled", True):
            try:
                poly_client = self.market_router.get_client(Venue.POLYMARKET)
                poly_markets = await poly_client.list_markets(limit=20)
                all_markets.extend(poly_markets)
                logger.info(f"Fetched {len(poly_markets)} Polymarket markets")
            except Exception as e:
                logger.error(f"Error fetching Polymarket markets: {e}")
        
        if not all_markets:
            logger.warning("No markets fetched, skipping iteration")
            return
        
        # 2. Run strategies
        all_decisions = []
        
        for strategy in self.strategies:
            # Filter markets for this strategy
            candidate_markets = await strategy.select_markets(all_markets)
            
            if not candidate_markets:
                logger.info(f"{strategy.name}: No candidate markets")
                continue
            
            decisions: list = []

            # Get evaluations from appropriate oracle source
            if isinstance(strategy, EnhancedStrategy):
                logger.info(
                    f"{strategy.name}: Running enhanced evaluation for "
                    f"{len(candidate_markets)} markets..."
                )
                decisions = await strategy.evaluate(candidate_markets, None)
            else:
                logger.info(
                    f"{strategy.name}: Analyzing {len(candidate_markets)} markets..."
                )

                oracle_results = await self.oracle.evaluate_markets(
                    candidate_markets,
                    model_group=strategy.name,
                )

                decisions = await strategy.evaluate(candidate_markets, oracle_results)
            all_decisions.extend(decisions)
            
            logger.info(f"{strategy.name}: Generated {len(decisions)} decisions")
        
        if not all_decisions:
            logger.info("No trade decisions generated")
            return
        
        # 3. Risk management
        validated = self.risk_manager.validate_decisions(all_decisions)
        
        approved_decisions = [d for d, reason in validated if reason is None]
        rejected_decisions = [(d, reason) for d, reason in validated if reason is not None]
        
        logger.info(
            f"Risk validation: {len(approved_decisions)} approved, "
            f"{len(rejected_decisions)} rejected"
        )
        
        for decision, reason in rejected_decisions:
            logger.debug(
                f"Rejected {decision.market_id} ({decision.strategy_name}): {reason}"
            )
        
        # 4. Execute approved decisions
        if approved_decisions:
            results = await self.executor.execute_decisions(approved_decisions)
            
            # Log to database
            await self._log_trades(results)
            
            # Update risk manager
            for decision, order_result in results:
                if order_result and order_result.status in ["FILLED", "PENDING"]:
                    self.risk_manager.record_position_opened(decision)
                    self.bankroll.allocate(decision.size_usd)
        
        logger.info("Scan iteration completed")

    async def main_loop(self):
        """Main event loop."""
        from ..config import settings
        
        await self.initialize()
        
        scan_interval = settings.scan_interval_seconds
        
        logger.info(f"Starting main loop (scan every {scan_interval}s)")
        
        try:
            while True:
                try:
                    await self.run_once()
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                
                logger.info(f"Sleeping for {scan_interval}s...")
                await asyncio.sleep(scan_interval)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        await self.market_router.close_all()
        await self.oracle.close()
        logger.info("Cleanup completed")

    async def _log_trades(self, results):
        """Log trades to database."""
        async with get_session() as session:
            for decision, order_result in results:
                trade = Trade(
                    venue=decision.venue.value,
                    market_id=decision.market_id,
                    outcome_id=decision.outcome_id,
                    strategy=decision.strategy_name,
                    mode=self.mode,
                    direction=decision.direction,
                    size_usd=decision.size_usd,
                    p_true=decision.p_true,
                    implied_p=decision.implied_p,
                    edge=decision.edge,
                    entry_price=order_result.avg_fill_price if order_result else None,
                    order_id=order_result.order_id if order_result else None,
                    rationale=decision.rationale,
                )
                session.add(trade)
