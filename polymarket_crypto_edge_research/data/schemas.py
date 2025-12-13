"""
Pydantic schemas for all data structures.
Strict validation with type safety.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Candle(BaseModel):
    """OHLCV candle data."""
    
    symbol: str
    timestamp: datetime
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: float = Field(ge=0)
    trades: int = Field(default=0, ge=0)
    
    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v
    
    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def body_size(self) -> float:
        """Absolute size of candle body."""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open


class OrderBookLevel(BaseModel):
    """Single level in order book."""
    
    price: float = Field(ge=0)
    size: float = Field(ge=0)
    
    @property
    def notional(self) -> float:
        """Dollar notional value."""
        return self.price * self.size


class OrderBook(BaseModel):
    """Order book snapshot."""
    
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    
    @property
    def best_bid(self) -> float | None:
        """Best bid price."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> float | None:
        """Best ask price."""
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> float | None:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> float | None:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> float | None:
        """Spread in basis points."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None
    
    def depth_at_price(self, price: float, side: Literal["bid", "ask"]) -> float:
        """Get cumulative depth at a price level."""
        levels = self.bids if side == "bid" else self.asks
        cumulative = 0.0
        for level in levels:
            if side == "bid" and level.price >= price:
                cumulative += level.size
            elif side == "ask" and level.price <= price:
                cumulative += level.size
        return cumulative


class Trade(BaseModel):
    """Trade/execution data."""
    
    symbol: str
    timestamp: datetime
    price: float = Field(ge=0)
    size: float = Field(ge=0)
    side: Literal["buy", "sell"]
    trade_id: str | None = None


class PolymarketOutcome(BaseModel):
    """Polymarket outcome within a market."""
    
    outcome_id: str
    name: str
    price: float = Field(ge=0, le=1)
    
    @property
    def implied_prob(self) -> float:
        """Implied probability from price."""
        return self.price


class PolymarketMarket(BaseModel):
    """Polymarket market data."""
    
    market_id: str
    condition_id: str
    question: str
    description: str = ""
    category: str = ""
    end_date: datetime | None = None
    resolution_date: datetime | None = None
    outcomes: list[PolymarketOutcome] = Field(default_factory=list)
    volume_24h: float = Field(default=0, ge=0)
    liquidity: float = Field(default=0, ge=0)
    is_active: bool = True
    
    @property
    def outcome_sum(self) -> float:
        """Sum of all outcome prices (should be ~1.0)."""
        return sum(o.price for o in self.outcomes)
    
    @property
    def has_arb_opportunity(self) -> bool:
        """Check if outcome prices don't sum to 1.0 (potential arb)."""
        return abs(self.outcome_sum - 1.0) > 0.02
    
    @property
    def minutes_until_resolution(self) -> float | None:
        """Minutes until market resolution."""
        if self.end_date:
            from core.time_utils import now_utc
            delta = self.end_date - now_utc()
            return delta.total_seconds() / 60
        return None
    
    @property
    def is_last_hour(self) -> bool:
        """Check if market resolves within 60 minutes."""
        mins = self.minutes_until_resolution
        return mins is not None and 0 < mins <= 60
    
    @property
    def is_last_10_min(self) -> bool:
        """Check if market resolves within 10 minutes."""
        mins = self.minutes_until_resolution
        return mins is not None and 0 < mins <= 10


class PolymarketTrade(BaseModel):
    """Trade on Polymarket."""
    
    market_id: str
    outcome_id: str
    timestamp: datetime
    price: float = Field(ge=0, le=1)
    size: float = Field(ge=0)
    side: Literal["buy", "sell"]
    maker: str | None = None
    taker: str | None = None


class GrokRegimeOutput(BaseModel):
    """Structured output from Grok regime classifier."""
    
    timestamp: datetime
    regime_label: Literal["risk_on", "risk_off", "choppy", "unknown"] = "unknown"
    sentiment_btc: float = Field(default=0.0, ge=-1, le=1)
    sentiment_eth: float = Field(default=0.0, ge=-1, le=1)
    sentiment_sol: float = Field(default=0.0, ge=-1, le=1)
    event_risk: float = Field(default=0.0, ge=0, le=1)
    reasoning: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    
    @property
    def is_risk_on(self) -> bool:
        return self.regime_label == "risk_on"
    
    @property
    def is_risk_off(self) -> bool:
        return self.regime_label == "risk_off"
    
    @property
    def avg_sentiment(self) -> float:
        """Average sentiment across all cryptos."""
        return (self.sentiment_btc + self.sentiment_eth + self.sentiment_sol) / 3


class PriceData(BaseModel):
    """Current price snapshot."""
    
    symbol: str
    timestamp: datetime
    price: float = Field(ge=0)
    bid: float | None = None
    ask: float | None = None
    volume_24h: float = Field(default=0, ge=0)
    change_24h_pct: float = 0.0
    
    @property
    def spread(self) -> float | None:
        if self.bid and self.ask:
            return self.ask - self.bid
        return None


class FeatureRow(BaseModel):
    """Row of features for ML model."""
    
    timestamp: datetime
    symbol: str
    features: dict[str, float]
    label: float | None = None
    
    def to_array(self, feature_names: list[str]) -> list[float]:
        """Convert to array in specified order."""
        return [self.features.get(name, 0.0) for name in feature_names]


class PredictionOutput(BaseModel):
    """ML model prediction output."""
    
    timestamp: datetime
    symbol: str
    model_name: str
    prediction: float
    confidence: float = Field(ge=0, le=1)
    direction: Literal["up", "flat", "down"]
    features_used: list[str] = Field(default_factory=list)


class TradeSignal(BaseModel):
    """Trading signal from strategy."""
    
    timestamp: datetime
    symbol: str
    market_id: str | None = None
    signal: Literal["long", "short", "hold"]
    confidence: float = Field(ge=0, le=1)
    edge: float = 0.0
    size_fraction: float = Field(default=0.0, ge=0, le=1)
    source: str = ""
    metadata: dict = Field(default_factory=dict)
