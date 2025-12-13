"""
Live orchestrator for real-time trading.
Coordinates data ingestion, prediction, and execution.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from core.config import get_settings
from core.logging_utils import get_logger

logger = get_logger(__name__)


class OrchestratorConfig(BaseModel):
    """Configuration for live orchestrator."""
    
    # Trading mode
    mode: str = "paper"  # "paper" or "live"
    
    # Symbols to trade
    crypto_symbols: list[str] = Field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    
    # Timing
    prediction_interval_minutes: int = 15
    regime_update_interval_minutes: int = 30
    cluster_update_interval_minutes: int = 60
    
    # Model paths
    model_registry_path: Path = Path("models")
    
    # Risk params
    max_daily_trades: int = 50
    max_drawdown_pct: float = 0.10
    
    # Polymarket
    poll_polymarket: bool = True
    polymarket_poll_interval_seconds: int = 30
    
    # Output
    log_trades: bool = True
    trades_log_path: Path = Path("trades")


class LiveOrchestrator:
    """
    Orchestrates live trading operations.
    
    Components:
    - Data ingestion (crypto prices, Polymarket data)
    - Feature computation
    - Model prediction
    - Grok regime analysis
    - Signal generation
    - Risk management
    - Trade execution (paper or live)
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._running = False
        self._tasks: list[asyncio.Task] = []
        
        # Components (lazily initialized)
        self._paper_trader = None
        self._model_registry = None
        self._grok_client = None
        self._regime_classifier = None
        self._direction_policy = None
        self._risk_manager = None
        
        # State
        self._last_prediction_time: dict[str, datetime] = {}
        self._last_regime_update: datetime | None = None
        self._current_regimes: dict[str, Any] = {}
        self._current_prices: dict[str, float] = {}
    
    async def start(self) -> None:
        """Start the live orchestrator."""
        logger.info(f"Starting orchestrator in {self.config.mode} mode")
        
        self._running = True
        
        # Initialize components
        await self._initialize_components()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._prediction_loop()),
            asyncio.create_task(self._regime_update_loop()),
            asyncio.create_task(self._position_monitor_loop()),
        ]
        
        if self.config.poll_polymarket:
            self._tasks.append(
                asyncio.create_task(self._polymarket_loop())
            )
        
        logger.info("Orchestrator started")
        
        # Wait for tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Orchestrator stopped")
    
    async def _initialize_components(self) -> None:
        """Initialize trading components."""
        from exec.paper_trader import PaperTrader
        from ml.model_registry import ModelRegistry
        from llm.grok_client import create_grok_client
        from llm.regime_classifier import RegimeClassifier
        from strategy.policy_15m_direction import DirectionPolicy
        from strategy.risk_manager import RiskManager, RiskConfig
        
        # Paper trader
        trades_file = self.config.trades_log_path / "paper_trades.json" if self.config.log_trades else None
        self._paper_trader = PaperTrader(
            initial_capital=10000.0,
            trades_file=trades_file
        )
        
        # Model registry
        self._model_registry = ModelRegistry(self.config.model_registry_path)
        
        # Grok client and regime classifier
        try:
            self._grok_client = create_grok_client()
            self._regime_classifier = RegimeClassifier(self._grok_client)
        except Exception as e:
            logger.warning(f"Grok client initialization failed: {e}")
            self._regime_classifier = None
        
        # Policies
        self._direction_policy = DirectionPolicy()
        
        # Risk manager
        risk_config = RiskConfig(
            max_trades_per_day=self.config.max_daily_trades,
            max_total_drawdown_pct=self.config.max_drawdown_pct
        )
        self._risk_manager = RiskManager(risk_config)
        
        logger.info("Components initialized")
    
    async def _prediction_loop(self) -> None:
        """Main prediction loop - runs every 15 minutes."""
        interval = self.config.prediction_interval_minutes * 60
        
        while self._running:
            try:
                await self._run_predictions()
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _regime_update_loop(self) -> None:
        """Regime update loop - runs every 30 minutes."""
        interval = self.config.regime_update_interval_minutes * 60
        
        while self._running:
            try:
                await self._update_regimes()
            except Exception as e:
                logger.error(f"Regime update error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _position_monitor_loop(self) -> None:
        """Monitor open positions - runs every minute."""
        interval = 60
        
        while self._running:
            try:
                await self._monitor_positions()
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _polymarket_loop(self) -> None:
        """Poll Polymarket for scalping opportunities."""
        interval = self.config.polymarket_poll_interval_seconds
        
        while self._running:
            try:
                await self._poll_polymarket()
            except Exception as e:
                logger.error(f"Polymarket loop error: {e}")
            
            await asyncio.sleep(interval)
    
    async def _run_predictions(self) -> None:
        """Run predictions for all symbols."""
        from core.time_utils import now_utc
        
        logger.info("Running predictions...")
        
        for symbol in self.config.crypto_symbols:
            try:
                # Get current data
                price_data = await self._get_price_data(symbol)
                if not price_data:
                    continue
                
                self._current_prices[symbol] = price_data.get("close", 0)
                
                # Build features
                features = await self._build_features(symbol, price_data)
                
                # Get ML prediction
                ml_prediction = await self._get_ml_prediction(symbol, features)
                
                # Get regime
                regime = self._current_regimes.get(symbol, {})
                
                # Generate signal
                signal = self._direction_policy.generate_signal(
                    symbol=symbol,
                    ml_prediction=ml_prediction,
                    regime_classification=regime,
                    current_price=self._current_prices.get(symbol),
                    indicators=features
                )
                
                # Execute if tradeable
                if signal.is_tradeable:
                    await self._execute_signal(signal)
                
                self._last_prediction_time[symbol] = now_utc()
                
            except Exception as e:
                logger.error(f"Prediction error for {symbol}: {e}")
        
        # Update positions with current prices
        if self._paper_trader:
            self._paper_trader.update_prices(self._current_prices)
    
    async def _update_regimes(self) -> None:
        """Update regime classifications for all symbols."""
        from core.time_utils import now_utc
        
        if self._regime_classifier is None:
            return
        
        logger.info("Updating regimes...")
        
        for symbol in self.config.crypto_symbols:
            try:
                price_data = await self._get_price_data(symbol)
                indicators = await self._get_indicators(symbol)
                
                classification = await self._regime_classifier.classify(
                    symbol=symbol,
                    price_data=price_data,
                    indicators=indicators
                )
                
                self._current_regimes[symbol] = {
                    "regime": classification.regime.value,
                    "regime_confidence": classification.regime_confidence,
                    "sentiment_score": classification.sentiment_score,
                    "is_volatile": classification.is_volatile,
                    "avoid_trading": classification.avoid_trading
                }
                
                logger.info(
                    f"Regime for {symbol}: {classification.regime.value} "
                    f"(conf={classification.regime_confidence:.2f})"
                )
                
            except Exception as e:
                logger.error(f"Regime update error for {symbol}: {e}")
        
        self._last_regime_update = now_utc()
    
    async def _monitor_positions(self) -> None:
        """Monitor open positions for stop-loss/take-profit."""
        if self._paper_trader is None:
            return
        
        open_trades = self._paper_trader.get_open_trades()
        
        for trade in open_trades:
            current_price = self._current_prices.get(trade.symbol)
            if current_price is None:
                continue
            
            # Check stop loss
            if trade.direction == "long":
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - current_price) / trade.entry_price
            
            # Stop loss at -5%
            if pnl_pct < -0.05:
                logger.info(f"Stop loss triggered for {trade.trade_id}")
                self._paper_trader.close_trade(trade.trade_id, current_price)
            
            # Take profit at +10%
            elif pnl_pct > 0.10:
                logger.info(f"Take profit triggered for {trade.trade_id}")
                self._paper_trader.close_trade(trade.trade_id, current_price)
    
    async def _poll_polymarket(self) -> None:
        """Poll Polymarket for opportunities."""
        # This would integrate with Polymarket API
        # For now, just a placeholder
        pass
    
    async def _get_price_data(self, symbol: str) -> dict[str, Any]:
        """Get current price data for symbol."""
        # This would integrate with data ingestion
        # Placeholder implementation
        return {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000.0,
            "current_price": 100.5,
            "high_24h": 102.0,
            "low_24h": 98.0,
            "change_24h": 0.5,
            "volume_24h": 5000000.0
        }
    
    async def _get_indicators(self, symbol: str) -> dict[str, float]:
        """Get technical indicators for symbol."""
        # Placeholder
        return {
            "rsi_14": 55.0,
            "macd_histogram": 0.01,
            "bb_percent": 0.6,
            "atr_percent": 0.02
        }
    
    async def _build_features(
        self,
        symbol: str,
        price_data: dict[str, Any]
    ) -> dict[str, float]:
        """Build features for ML model."""
        # Placeholder - would use feature builders
        return {
            "rsi_14": 55.0,
            "macd_histogram": 0.01,
            "momentum_5": 0.02,
            "volatility_20": 0.015
        }
    
    async def _get_ml_prediction(
        self,
        symbol: str,
        features: dict[str, float]
    ) -> dict[str, Any]:
        """Get ML model prediction."""
        if self._model_registry is None:
            return {"prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5}
        
        try:
            result = self._model_registry.load_champion(symbol)
            if result is None:
                return {"prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5}
            
            model, calibrator = result
            prediction = model.predict_single(features, symbol)
            
            # Apply calibration
            if calibrator:
                import numpy as np
                calibrated = calibrator.calibrate(np.array([prediction.probability_up]))[0]
                prediction.probability_up = float(calibrated)
                prediction.probability_down = 1 - float(calibrated)
            
            return {
                "prob_up": prediction.probability_up,
                "prob_down": prediction.probability_down,
                "confidence": prediction.confidence
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {"prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5}
    
    async def _execute_signal(self, signal) -> None:
        """Execute a trading signal."""
        if self._paper_trader is None:
            return
        
        if self.config.mode == "paper":
            current_price = self._current_prices.get(signal.symbol, 100.0)
            size_usd = signal.recommended_size * 1000  # Scale to dollars
            
            if signal.direction == "long":
                self._paper_trader.open_trade(
                    symbol=signal.symbol,
                    direction="long",
                    size_usd=size_usd,
                    entry_price=current_price,
                    strategy="15m_direction",
                    confidence=signal.combined_confidence,
                    regime=signal.regime
                )
            elif signal.direction == "short":
                self._paper_trader.open_trade(
                    symbol=signal.symbol,
                    direction="short",
                    size_usd=size_usd,
                    entry_price=current_price,
                    strategy="15m_direction",
                    confidence=signal.combined_confidence,
                    regime=signal.regime
                )
        
        elif self.config.mode == "live":
            # Live execution would go here
            logger.warning("Live trading not implemented")
    
    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        from core.time_utils import now_utc
        
        status = {
            "running": self._running,
            "mode": self.config.mode,
            "current_time": now_utc().isoformat(),
            "symbols": self.config.crypto_symbols,
            "current_prices": self._current_prices,
            "current_regimes": {
                k: v.get("regime", "unknown")
                for k, v in self._current_regimes.items()
            }
        }
        
        if self._paper_trader:
            status["paper_trading"] = self._paper_trader.get_stats()
        
        return status
    
    def print_status(self) -> None:
        """Print current status."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        status = self.get_status()
        
        console.print(f"\n[bold]Orchestrator Status[/bold]")
        console.print(f"Mode: {status['mode']}")
        console.print(f"Running: {status['running']}")
        console.print(f"Time: {status['current_time']}")
        
        # Prices and regimes
        table = Table(title="Current State")
        table.add_column("Symbol")
        table.add_column("Price")
        table.add_column("Regime")
        
        for symbol in self.config.crypto_symbols:
            price = status["current_prices"].get(symbol, "N/A")
            regime = status["current_regimes"].get(symbol, "unknown")
            table.add_row(
                symbol,
                f"${price:,.2f}" if isinstance(price, (int, float)) else str(price),
                regime
            )
        
        console.print(table)
        
        # Paper trading stats
        if "paper_trading" in status:
            pt = status["paper_trading"]
            console.print(f"\n[bold]Paper Trading[/bold]")
            console.print(f"Trades: {pt.get('total_trades', 0)}")
            console.print(f"Win Rate: {pt.get('win_rate', 0):.1%}")
            console.print(f"P&L: ${pt.get('total_pnl', 0):+,.2f}")


async def run_orchestrator(config: OrchestratorConfig | None = None) -> None:
    """Run the live orchestrator."""
    config = config or OrchestratorConfig()
    orchestrator = LiveOrchestrator(config)
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        await orchestrator.stop()
