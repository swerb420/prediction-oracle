"""Market data models and types."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Venue(str, Enum):
    """Supported prediction market venues."""

    KALSHI = "KALSHI"
    POLYMARKET = "POLYMARKET"


class Outcome(BaseModel):
    """A single outcome within a market."""

    id: str = Field(..., description="Unique outcome identifier")
    label: str = Field(..., description="Human-readable label")
    price: float = Field(..., ge=0.0, le=1.0, description="Current price (0-1 probability)")
    volume_24h: float | None = Field(None, description="24h volume in USD")
    liquidity: float | None = Field(None, description="Available liquidity in USD")
    

class Market(BaseModel):
    """A prediction market with its outcomes."""

    venue: Venue
    market_id: str = Field(..., description="Venue-specific market identifier")
    question: str = Field(..., description="Market question")
    rules: str = Field(default="", description="Resolution rules")
    category: str = Field(default="", description="Market category/topic")
    close_time: datetime = Field(..., description="Market close/resolution time")
    outcomes: list[Outcome] = Field(..., description="Available outcomes")
    volume_24h: float | None = Field(None, description="Total 24h volume")
    tags: list[str] = Field(default_factory=list, description="Market tags")
    

class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderRequest(BaseModel):
    """Request to place an order."""

    venue: Venue
    market_id: str
    outcome_id: str
    side: OrderSide
    order_type: OrderType
    size_usd: float = Field(..., gt=0)
    limit_price: float | None = Field(None, ge=0.0, le=1.0)


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class OrderResult(BaseModel):
    """Result of placing an order."""

    order_id: str
    status: OrderStatus
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    message: str = ""


class Position(BaseModel):
    """Current position in a market."""

    venue: Venue
    market_id: str
    outcome_id: str
    size: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
