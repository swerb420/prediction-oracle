"""
Feature builder for Polymarket microstructure.
Computes orderbook, flow, and momentum features for market scalping.
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel

from core.logging_utils import get_logger
from data.schemas import OrderBook, PolymarketMarket, PolymarketTrade

logger = get_logger(__name__)


class MarketFeatures(BaseModel):
    """Features for a single Polymarket market."""
    
    market_id: str
    timestamp: datetime
    
    # Price features
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    
    # Orderbook features
    bid_depth_5: float  # Total bid depth in top 5 levels
    ask_depth_5: float
    bid_depth_10: float
    ask_depth_10: float
    depth_imbalance: float  # (bid - ask) / (bid + ask)
    
    # Orderbook shape
    bid_slope: float  # Price decay rate in bids
    ask_slope: float
    book_skew: float  # Asymmetry in depth
    
    # Trade flow features
    buy_volume_10m: float
    sell_volume_10m: float
    net_flow_10m: float
    trade_count_10m: int
    avg_trade_size: float
    vwap_10m: float
    
    # Momentum features
    price_change_5m: float
    price_change_10m: float
    price_change_30m: float
    price_volatility: float
    price_trend: float  # -1 to 1
    
    # Market metadata
    volume_24h: float
    liquidity: float
    minutes_to_resolution: float
    outcome_sum: float  # Sum of outcome prices (should be ~1.0)
    
    # Derived features
    urgency_score: float  # Higher closer to resolution
    arb_potential: float  # Deviation from sum=1
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dict for ML model."""
        return {
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
            "bid_depth_5": self.bid_depth_5,
            "ask_depth_5": self.ask_depth_5,
            "depth_imbalance": self.depth_imbalance,
            "bid_slope": self.bid_slope,
            "ask_slope": self.ask_slope,
            "book_skew": self.book_skew,
            "buy_volume_10m": self.buy_volume_10m,
            "sell_volume_10m": self.sell_volume_10m,
            "net_flow_10m": self.net_flow_10m,
            "trade_count_10m": float(self.trade_count_10m),
            "avg_trade_size": self.avg_trade_size,
            "vwap_10m": self.vwap_10m,
            "price_change_5m": self.price_change_5m,
            "price_change_10m": self.price_change_10m,
            "price_change_30m": self.price_change_30m,
            "price_volatility": self.price_volatility,
            "price_trend": self.price_trend,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "minutes_to_resolution": self.minutes_to_resolution,
            "outcome_sum": self.outcome_sum,
            "urgency_score": self.urgency_score,
            "arb_potential": self.arb_potential,
        }


class MarketFeatureBuilder:
    """
    Builds microstructure features for Polymarket markets.
    """
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get list of feature names."""
        return [
            "mid_price", "spread_bps",
            "bid_depth_5", "ask_depth_5", "depth_imbalance",
            "bid_slope", "ask_slope", "book_skew",
            "buy_volume_10m", "sell_volume_10m", "net_flow_10m",
            "trade_count_10m", "avg_trade_size", "vwap_10m",
            "price_change_5m", "price_change_10m", "price_change_30m",
            "price_volatility", "price_trend",
            "volume_24h", "liquidity", "minutes_to_resolution",
            "outcome_sum", "urgency_score", "arb_potential",
        ]
    
    def build(
        self,
        market: PolymarketMarket,
        orderbook: OrderBook | None = None,
        recent_trades: list[PolymarketTrade] | None = None,
        price_history: list[dict] | None = None
    ) -> MarketFeatures:
        """
        Build features for a market.
        
        Args:
            market: Market metadata
            orderbook: Current orderbook snapshot
            recent_trades: Recent trades for flow analysis
            price_history: Historical price data for momentum
            
        Returns:
            MarketFeatures
        """
        from core.time_utils import now_utc
        
        timestamp = now_utc()
        
        # Default values
        mid_price = 0.5
        best_bid = 0.0
        best_ask = 1.0
        spread_bps = 0.0
        
        bid_depth_5 = 0.0
        ask_depth_5 = 0.0
        bid_depth_10 = 0.0
        ask_depth_10 = 0.0
        depth_imbalance = 0.0
        
        bid_slope = 0.0
        ask_slope = 0.0
        book_skew = 0.0
        
        # Orderbook features
        if orderbook and orderbook.bids and orderbook.asks:
            best_bid = orderbook.best_bid or 0.0
            best_ask = orderbook.best_ask or 1.0
            mid_price = orderbook.mid_price or 0.5
            spread_bps = orderbook.spread_bps or 0.0
            
            # Depth calculation
            bid_depth_5 = sum(l.size for l in orderbook.bids[:5])
            ask_depth_5 = sum(l.size for l in orderbook.asks[:5])
            bid_depth_10 = sum(l.size for l in orderbook.bids[:10])
            ask_depth_10 = sum(l.size for l in orderbook.asks[:10])
            
            total_depth = bid_depth_5 + ask_depth_5
            depth_imbalance = (bid_depth_5 - ask_depth_5) / total_depth if total_depth > 0 else 0
            
            # Book shape (slope = how fast depth decays with price)
            bid_slope = self._calculate_slope(orderbook.bids[:10], "bid")
            ask_slope = self._calculate_slope(orderbook.asks[:10], "ask")
            
            # Skew
            book_skew = (bid_slope - ask_slope) / (abs(bid_slope) + abs(ask_slope) + 1e-6)
        
        # Trade flow features
        buy_volume_10m = 0.0
        sell_volume_10m = 0.0
        net_flow_10m = 0.0
        trade_count_10m = 0
        avg_trade_size = 0.0
        vwap_10m = mid_price
        
        if recent_trades:
            buy_trades = [t for t in recent_trades if t.side == "buy"]
            sell_trades = [t for t in recent_trades if t.side == "sell"]
            
            buy_volume_10m = sum(t.size for t in buy_trades)
            sell_volume_10m = sum(t.size for t in sell_trades)
            net_flow_10m = buy_volume_10m - sell_volume_10m
            trade_count_10m = len(recent_trades)
            
            total_volume = buy_volume_10m + sell_volume_10m
            if total_volume > 0:
                avg_trade_size = total_volume / trade_count_10m
                vwap_10m = sum(t.price * t.size for t in recent_trades) / total_volume
        
        # Price momentum
        price_change_5m = 0.0
        price_change_10m = 0.0
        price_change_30m = 0.0
        price_volatility = 0.0
        price_trend = 0.0
        
        if price_history and len(price_history) >= 2:
            prices = [h.get("price", mid_price) for h in price_history]
            
            if len(prices) >= 5:
                price_change_5m = (prices[-1] / prices[-5]) - 1 if prices[-5] > 0 else 0
            if len(prices) >= 10:
                price_change_10m = (prices[-1] / prices[-10]) - 1 if prices[-10] > 0 else 0
            if len(prices) >= 30:
                price_change_30m = (prices[-1] / prices[-30]) - 1 if prices[-30] > 0 else 0
            
            if len(prices) >= 10:
                returns = np.diff(prices[-10:]) / np.array(prices[-10:-1])
                price_volatility = float(np.std(returns)) if len(returns) > 0 else 0
            
            # Trend direction
            if price_change_10m > 0.02:
                price_trend = min(1.0, price_change_10m * 10)
            elif price_change_10m < -0.02:
                price_trend = max(-1.0, price_change_10m * 10)
        
        # Market metadata
        minutes_to_resolution = market.minutes_until_resolution or 1440  # Default 24h
        outcome_sum = market.outcome_sum
        
        # Derived features
        urgency_score = self._calculate_urgency(minutes_to_resolution)
        arb_potential = abs(outcome_sum - 1.0)
        
        return MarketFeatures(
            market_id=market.market_id,
            timestamp=timestamp,
            mid_price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            bid_depth_10=bid_depth_10,
            ask_depth_10=ask_depth_10,
            depth_imbalance=depth_imbalance,
            bid_slope=bid_slope,
            ask_slope=ask_slope,
            book_skew=book_skew,
            buy_volume_10m=buy_volume_10m,
            sell_volume_10m=sell_volume_10m,
            net_flow_10m=net_flow_10m,
            trade_count_10m=trade_count_10m,
            avg_trade_size=avg_trade_size,
            vwap_10m=vwap_10m,
            price_change_5m=price_change_5m,
            price_change_10m=price_change_10m,
            price_change_30m=price_change_30m,
            price_volatility=price_volatility,
            price_trend=price_trend,
            volume_24h=market.volume_24h,
            liquidity=market.liquidity,
            minutes_to_resolution=minutes_to_resolution,
            outcome_sum=outcome_sum,
            urgency_score=urgency_score,
            arb_potential=arb_potential,
        )
    
    def _calculate_slope(self, levels: list, side: str) -> float:
        """Calculate depth decay slope."""
        if len(levels) < 2:
            return 0.0
        
        cumulative_depth = []
        prices = []
        
        cum = 0.0
        for level in levels:
            cum += level.size
            cumulative_depth.append(cum)
            prices.append(level.price)
        
        if len(prices) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.array(prices)
        y = np.array(cumulative_depth)
        
        if np.std(x) == 0:
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize
        return float(slope / (cumulative_depth[-1] + 1e-6))
    
    def _calculate_urgency(self, minutes_to_resolution: float) -> float:
        """
        Calculate urgency score based on time to resolution.
        Higher score = closer to resolution = more urgent.
        """
        if minutes_to_resolution <= 0:
            return 1.0
        elif minutes_to_resolution <= 5:
            return 0.95
        elif minutes_to_resolution <= 10:
            return 0.85
        elif minutes_to_resolution <= 30:
            return 0.7
        elif minutes_to_resolution <= 60:
            return 0.5
        elif minutes_to_resolution <= 180:
            return 0.3
        elif minutes_to_resolution <= 720:
            return 0.15
        else:
            return 0.05


def build_market_features(
    market: PolymarketMarket,
    orderbook: OrderBook | None = None,
    recent_trades: list[PolymarketTrade] | None = None,
    price_history: list[dict] | None = None
) -> MarketFeatures:
    """Convenience function to build market features."""
    builder = MarketFeatureBuilder()
    return builder.build(market, orderbook, recent_trades, price_history)
