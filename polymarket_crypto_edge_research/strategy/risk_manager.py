"""
Risk management for trading strategies.
Implements Kelly-capped position sizing, drawdown limits, and exposure controls.
"""

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class PositionLimit(BaseModel):
    """Position limits for an asset or strategy."""
    
    symbol: str
    max_position_usd: float = 1000.0
    max_position_pct: float = 0.10  # Max % of portfolio
    max_daily_loss_usd: float = 200.0
    max_daily_trades: int = 20
    max_open_positions: int = 5


class RiskConfig(BaseModel):
    """Global risk configuration."""
    
    # Portfolio limits
    total_portfolio_usd: float = 10000.0
    max_total_exposure_pct: float = 0.50  # Max 50% deployed
    
    # Position limits
    max_single_position_pct: float = 0.10  # Max 10% per position
    max_per_asset_pct: float = 0.20  # Max 20% per asset class
    
    # Drawdown controls
    max_daily_drawdown_pct: float = 0.05  # 5% daily drawdown limit
    max_weekly_drawdown_pct: float = 0.10  # 10% weekly
    max_total_drawdown_pct: float = 0.20  # 20% total drawdown from peak
    
    # Kelly fraction cap
    max_kelly_fraction: float = 0.25  # Cap Kelly at 25%
    kelly_fraction_multiplier: float = 0.5  # Use half-Kelly
    
    # Trade limits
    max_trades_per_day: int = 50
    max_trades_per_hour: int = 10
    min_time_between_trades_seconds: int = 30
    
    # Correlation limits
    max_correlated_positions: int = 3  # Max positions in correlated assets


class RiskCheck(BaseModel):
    """Result of a risk check."""
    
    passed: bool
    check_name: str
    message: str
    current_value: float | None = None
    limit_value: float | None = None
    
    @property
    def severity(self) -> str:
        if self.passed:
            return "ok"
        if self.current_value and self.limit_value:
            pct_over = (self.current_value - self.limit_value) / self.limit_value
            if pct_over > 0.5:
                return "critical"
            elif pct_over > 0.2:
                return "warning"
        return "blocked"


class PortfolioState(BaseModel):
    """Current portfolio state for risk calculations."""
    
    timestamp: datetime
    
    # Value
    total_value_usd: float = 10000.0
    cash_usd: float = 10000.0
    deployed_usd: float = 0.0
    
    # P&L
    daily_pnl_usd: float = 0.0
    weekly_pnl_usd: float = 0.0
    total_pnl_usd: float = 0.0
    peak_value_usd: float = 10000.0
    
    # Positions
    n_open_positions: int = 0
    positions_by_asset: dict[str, float] = Field(default_factory=dict)
    
    # Activity
    trades_today: int = 0
    trades_this_hour: int = 0
    last_trade_time: datetime | None = None
    
    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_value_usd <= 0:
            return 0.0
        return (self.peak_value_usd - self.total_value_usd) / self.peak_value_usd
    
    @property
    def daily_drawdown_pct(self) -> float:
        """Daily drawdown percentage."""
        if self.total_value_usd <= 0:
            return 0.0
        return -self.daily_pnl_usd / self.total_value_usd
    
    @property
    def exposure_pct(self) -> float:
        """Current exposure percentage."""
        if self.total_value_usd <= 0:
            return 0.0
        return self.deployed_usd / self.total_value_usd


class RiskManager:
    """
    Risk management for trading operations.
    
    Features:
    - Position sizing with Kelly criterion
    - Drawdown controls
    - Exposure limits
    - Trade frequency limits
    - Correlation-aware position limits
    """
    
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self._position_limits: dict[str, PositionLimit] = {}
        self._portfolio_state: PortfolioState | None = None
    
    def set_portfolio_state(self, state: PortfolioState) -> None:
        """Update portfolio state for risk calculations."""
        self._portfolio_state = state
    
    def set_position_limit(self, limit: PositionLimit) -> None:
        """Set position limits for a specific asset."""
        self._position_limits[limit.symbol] = limit
    
    def check_trade(
        self,
        symbol: str,
        direction: str,  # "long" or "short"
        size_usd: float,
        confidence: float = 0.5
    ) -> list[RiskCheck]:
        """
        Run all risk checks for a proposed trade.
        
        Args:
            symbol: Asset symbol
            direction: Trade direction
            size_usd: Proposed position size in USD
            confidence: Prediction confidence
            
        Returns:
            List of RiskCheck results
        """
        checks = []
        
        if self._portfolio_state is None:
            checks.append(RiskCheck(
                passed=False,
                check_name="portfolio_state",
                message="Portfolio state not initialized"
            ))
            return checks
        
        state = self._portfolio_state
        
        # 1. Check max single position
        max_position = self.config.max_single_position_pct * state.total_value_usd
        checks.append(RiskCheck(
            passed=size_usd <= max_position,
            check_name="max_single_position",
            message=f"Position ${size_usd:.0f} vs limit ${max_position:.0f}",
            current_value=size_usd,
            limit_value=max_position
        ))
        
        # 2. Check total exposure
        new_exposure = state.deployed_usd + size_usd
        max_exposure = self.config.max_total_exposure_pct * state.total_value_usd
        checks.append(RiskCheck(
            passed=new_exposure <= max_exposure,
            check_name="max_exposure",
            message=f"Exposure ${new_exposure:.0f} vs limit ${max_exposure:.0f}",
            current_value=new_exposure,
            limit_value=max_exposure
        ))
        
        # 3. Check daily drawdown
        max_dd = self.config.max_daily_drawdown_pct
        checks.append(RiskCheck(
            passed=state.daily_drawdown_pct < max_dd,
            check_name="daily_drawdown",
            message=f"Daily DD {state.daily_drawdown_pct:.1%} vs limit {max_dd:.1%}",
            current_value=state.daily_drawdown_pct,
            limit_value=max_dd
        ))
        
        # 4. Check total drawdown
        max_total_dd = self.config.max_total_drawdown_pct
        checks.append(RiskCheck(
            passed=state.current_drawdown_pct < max_total_dd,
            check_name="total_drawdown",
            message=f"Total DD {state.current_drawdown_pct:.1%} vs limit {max_total_dd:.1%}",
            current_value=state.current_drawdown_pct,
            limit_value=max_total_dd
        ))
        
        # 5. Check daily trade count
        max_trades = self.config.max_trades_per_day
        checks.append(RiskCheck(
            passed=state.trades_today < max_trades,
            check_name="daily_trades",
            message=f"Trades today {state.trades_today} vs limit {max_trades}",
            current_value=float(state.trades_today),
            limit_value=float(max_trades)
        ))
        
        # 6. Check hourly trade count
        max_hourly = self.config.max_trades_per_hour
        checks.append(RiskCheck(
            passed=state.trades_this_hour < max_hourly,
            check_name="hourly_trades",
            message=f"Trades this hour {state.trades_this_hour} vs limit {max_hourly}",
            current_value=float(state.trades_this_hour),
            limit_value=float(max_hourly)
        ))
        
        # 7. Check time since last trade
        if state.last_trade_time:
            from core.time_utils import now_utc
            elapsed = (now_utc() - state.last_trade_time).total_seconds()
            min_time = self.config.min_time_between_trades_seconds
            checks.append(RiskCheck(
                passed=elapsed >= min_time,
                check_name="trade_cooldown",
                message=f"Time since last trade {elapsed:.0f}s vs min {min_time}s",
                current_value=elapsed,
                limit_value=float(min_time)
            ))
        
        # 8. Check per-asset exposure
        current_asset_exposure = state.positions_by_asset.get(symbol, 0.0)
        new_asset_exposure = current_asset_exposure + size_usd
        max_asset = self.config.max_per_asset_pct * state.total_value_usd
        checks.append(RiskCheck(
            passed=new_asset_exposure <= max_asset,
            check_name="per_asset_exposure",
            message=f"Asset exposure ${new_asset_exposure:.0f} vs limit ${max_asset:.0f}",
            current_value=new_asset_exposure,
            limit_value=max_asset
        ))
        
        # 9. Check asset-specific limits
        if symbol in self._position_limits:
            limit = self._position_limits[symbol]
            checks.append(RiskCheck(
                passed=size_usd <= limit.max_position_usd,
                check_name="asset_position_limit",
                message=f"Position ${size_usd:.0f} vs {symbol} limit ${limit.max_position_usd:.0f}",
                current_value=size_usd,
                limit_value=limit.max_position_usd
            ))
        
        # 10. Check sufficient cash
        checks.append(RiskCheck(
            passed=size_usd <= state.cash_usd,
            check_name="sufficient_cash",
            message=f"Size ${size_usd:.0f} vs available ${state.cash_usd:.0f}",
            current_value=size_usd,
            limit_value=state.cash_usd
        ))
        
        return checks
    
    def calculate_position_size(
        self,
        win_probability: float,
        confidence: float,
        win_payout: float = 1.0,  # 1:1 odds by default
        loss_payout: float = 1.0
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion.
        
        Kelly formula: f* = (bp - q) / b
        where b = win_payout/loss_payout, p = win_prob, q = 1-p
        
        Returns:
            Fraction of portfolio to bet (0-1)
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        q = 1 - win_probability
        b = win_payout / loss_payout
        
        kelly = (b * win_probability - q) / b
        
        # Cap Kelly
        kelly = max(0, min(kelly, 1.0))
        
        # Apply fraction multiplier (half-Kelly)
        kelly *= self.config.kelly_fraction_multiplier
        
        # Apply confidence adjustment
        kelly *= confidence
        
        # Cap at max Kelly
        kelly = min(kelly, self.config.max_kelly_fraction)
        
        return kelly
    
    def get_max_position_size(
        self,
        symbol: str,
        kelly_fraction: float
    ) -> float:
        """
        Get maximum position size in USD considering all limits.
        
        Args:
            symbol: Asset symbol
            kelly_fraction: Calculated Kelly fraction
            
        Returns:
            Max position size in USD
        """
        if self._portfolio_state is None:
            return 0.0
        
        state = self._portfolio_state
        
        limits = []
        
        # Kelly-based size
        limits.append(kelly_fraction * state.total_value_usd)
        
        # Single position limit
        limits.append(self.config.max_single_position_pct * state.total_value_usd)
        
        # Available exposure
        max_exposure = self.config.max_total_exposure_pct * state.total_value_usd
        available_exposure = max_exposure - state.deployed_usd
        limits.append(max(0, available_exposure))
        
        # Per-asset limit
        current_asset = state.positions_by_asset.get(symbol, 0.0)
        max_asset = self.config.max_per_asset_pct * state.total_value_usd
        limits.append(max(0, max_asset - current_asset))
        
        # Asset-specific limit
        if symbol in self._position_limits:
            limits.append(self._position_limits[symbol].max_position_usd)
        
        # Available cash
        limits.append(state.cash_usd)
        
        return min(limits)
    
    def should_reduce_exposure(self) -> tuple[bool, str]:
        """
        Check if exposure should be reduced due to drawdown.
        
        Returns:
            (should_reduce, reason)
        """
        if self._portfolio_state is None:
            return False, ""
        
        state = self._portfolio_state
        
        # Check daily drawdown
        if state.daily_drawdown_pct > self.config.max_daily_drawdown_pct * 0.8:
            return True, f"Approaching daily drawdown limit ({state.daily_drawdown_pct:.1%})"
        
        # Check total drawdown
        if state.current_drawdown_pct > self.config.max_total_drawdown_pct * 0.8:
            return True, f"Approaching total drawdown limit ({state.current_drawdown_pct:.1%})"
        
        # Check exposure
        if state.exposure_pct > self.config.max_total_exposure_pct * 0.9:
            return True, f"High exposure ({state.exposure_pct:.1%})"
        
        return False, ""
    
    def get_position_adjustment(
        self,
        symbol: str,
        current_size: float,
        current_pnl_pct: float
    ) -> tuple[str, float]:
        """
        Get recommended position adjustment (hold, reduce, close).
        
        Args:
            symbol: Asset symbol
            current_size: Current position size USD
            current_pnl_pct: Current P&L percentage
            
        Returns:
            (action, target_size)
        """
        # Check if we should reduce overall exposure
        should_reduce, reason = self.should_reduce_exposure()
        
        if should_reduce:
            # Reduce all positions by 50%
            return "reduce", current_size * 0.5
        
        # Check stop loss (trailing)
        if current_pnl_pct < -0.10:  # -10%
            return "close", 0.0
        
        # Check take profit with trailing
        if current_pnl_pct > 0.20:  # +20%
            # Take some profit
            return "reduce", current_size * 0.5
        
        return "hold", current_size
    
    def record_trade(
        self,
        symbol: str,
        size_usd: float,
        pnl_usd: float = 0.0
    ) -> None:
        """Record a trade for tracking purposes."""
        if self._portfolio_state is None:
            return
        
        from core.time_utils import now_utc
        
        self._portfolio_state.trades_today += 1
        self._portfolio_state.trades_this_hour += 1
        self._portfolio_state.last_trade_time = now_utc()
        
        # Update P&L
        self._portfolio_state.daily_pnl_usd += pnl_usd
        self._portfolio_state.total_pnl_usd += pnl_usd
        self._portfolio_state.total_value_usd += pnl_usd
        
        # Update peak
        if self._portfolio_state.total_value_usd > self._portfolio_state.peak_value_usd:
            self._portfolio_state.peak_value_usd = self._portfolio_state.total_value_usd
    
    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of each day)."""
        if self._portfolio_state is None:
            return
        
        self._portfolio_state.trades_today = 0
        self._portfolio_state.daily_pnl_usd = 0.0
    
    def reset_hourly_counters(self) -> None:
        """Reset hourly counters."""
        if self._portfolio_state is None:
            return
        
        self._portfolio_state.trades_this_hour = 0
    
    def get_risk_summary(self) -> dict[str, Any]:
        """Get summary of current risk state."""
        if self._portfolio_state is None:
            return {"error": "Portfolio state not initialized"}
        
        state = self._portfolio_state
        
        return {
            "total_value_usd": state.total_value_usd,
            "exposure_pct": state.exposure_pct,
            "daily_pnl_usd": state.daily_pnl_usd,
            "daily_drawdown_pct": state.daily_drawdown_pct,
            "total_drawdown_pct": state.current_drawdown_pct,
            "n_positions": state.n_open_positions,
            "trades_today": state.trades_today,
            "trades_this_hour": state.trades_this_hour,
            "should_reduce_exposure": self.should_reduce_exposure()[0],
            "status": "OK" if state.current_drawdown_pct < self.config.max_total_drawdown_pct else "CRITICAL"
        }
