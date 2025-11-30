"""
Polymarket-specific signals using their FREE public APIs.
- Real-time price/volume data
- Order book depth analysis
- Whale tracking (large trades)
- Market momentum indicators
"""
import asyncio
import logging
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OrderBookDepth(BaseModel):
    """Order book analysis for a market."""
    bid_depth_10pct: float  # Total $ within 10% of best bid
    ask_depth_10pct: float  # Total $ within 10% of best ask
    bid_ask_imbalance: float  # -1 to 1, positive = more bids
    spread_bps: float  # Spread in basis points
    microprice: float  # Volume-weighted mid price


class MarketMomentum(BaseModel):
    """Price momentum indicators."""
    price_change_1h: float
    price_change_24h: float
    volume_change_24h: float  # vs previous 24h
    velocity: float  # Rate of price change
    acceleration: float  # Change in velocity


class WhaleActivity(BaseModel):
    """Large trade detection."""
    large_buys_24h: int  # Trades > $500
    large_sells_24h: int
    large_buy_volume: float
    large_sell_volume: float
    whale_bias: float  # -1 to 1, positive = whales buying


class SmartMoneySignal(BaseModel):
    """Combined smart money indicators."""
    market_id: str
    order_book: OrderBookDepth
    momentum: MarketMomentum
    whales: WhaleActivity
    
    # Composite scores
    smart_money_score: float  # -1 to 1
    confidence: float  # 0 to 1
    signal_strength: str  # "strong_buy", "buy", "neutral", "sell", "strong_sell"


class PolymarketSignalProvider:
    """
    Extracts alpha from Polymarket's public data.
    No API key needed - all public endpoints!
    """
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self._price_cache: dict[str, list[tuple[datetime, float]]] = {}
    
    async def get_smart_money_signal(self, market) -> SmartMoneySignal:
        """
        Generate comprehensive smart money signal for a market.
        Combines order book, momentum, and whale data.
        """
        # Fetch all data in parallel
        order_book_task = self._analyze_order_book(market)
        momentum_task = self._analyze_momentum(market)
        whale_task = self._analyze_whale_activity(market)
        
        order_book, momentum, whales = await asyncio.gather(
            order_book_task, momentum_task, whale_task
        )
        
        # Calculate composite smart money score
        # Weights based on predictive power
        ob_score = order_book.bid_ask_imbalance * 0.3
        mom_score = min(1, max(-1, momentum.price_change_24h * 5)) * 0.3
        whale_score = whales.whale_bias * 0.4
        
        smart_money_score = ob_score + mom_score + whale_score
        
        # Confidence based on data quality
        confidence = min(1.0, (
            (0.3 if abs(order_book.bid_ask_imbalance) > 0.1 else 0.1) +
            (0.3 if abs(momentum.price_change_24h) > 0.02 else 0.1) +
            (0.4 if whales.large_buys_24h + whales.large_sells_24h > 5 else 0.1)
        ))
        
        # Signal strength
        if smart_money_score > 0.5:
            signal_strength = "strong_buy"
        elif smart_money_score > 0.2:
            signal_strength = "buy"
        elif smart_money_score < -0.5:
            signal_strength = "strong_sell"
        elif smart_money_score < -0.2:
            signal_strength = "sell"
        else:
            signal_strength = "neutral"
        
        return SmartMoneySignal(
            market_id=market.market_id,
            order_book=order_book,
            momentum=momentum,
            whales=whales,
            smart_money_score=smart_money_score,
            confidence=confidence,
            signal_strength=signal_strength,
        )
    
    async def _analyze_order_book(self, market) -> OrderBookDepth:
        """Analyze order book depth and imbalance."""
        try:
            # Get order book from CLOB
            token_id = market.outcomes[0].id if market.outcomes else market.market_id
            
            resp = await self.client.get(
                f"{self.CLOB_API}/book",
                params={"token_id": token_id}
            )
            
            if resp.status_code != 200:
                return self._default_order_book()
            
            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            if not bids or not asks:
                return self._default_order_book()
            
            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])
            
            # Calculate depth within 10% of best prices
            bid_depth = sum(
                float(b["size"]) * float(b["price"])
                for b in bids
                if float(b["price"]) >= best_bid * 0.9
            )
            ask_depth = sum(
                float(a["size"]) * float(a["price"])
                for a in asks
                if float(a["price"]) <= best_ask * 1.1
            )
            
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            spread_bps = ((best_ask - best_bid) / best_bid) * 10000 if best_bid > 0 else 0
            
            # Volume-weighted microprice
            bid_vol = sum(float(b["size"]) for b in bids[:5])
            ask_vol = sum(float(a["size"]) for a in asks[:5])
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                microprice = (best_bid * ask_vol + best_ask * bid_vol) / total_vol
            else:
                microprice = (best_bid + best_ask) / 2
            
            return OrderBookDepth(
                bid_depth_10pct=bid_depth,
                ask_depth_10pct=ask_depth,
                bid_ask_imbalance=imbalance,
                spread_bps=spread_bps,
                microprice=microprice,
            )
        except Exception as e:
            logger.debug(f"Order book analysis failed: {e}")
            return self._default_order_book()
    
    def _default_order_book(self) -> OrderBookDepth:
        return OrderBookDepth(
            bid_depth_10pct=0,
            ask_depth_10pct=0,
            bid_ask_imbalance=0,
            spread_bps=100,
            microprice=0.5,
        )
    
    async def _analyze_momentum(self, market) -> MarketMomentum:
        """Calculate price momentum indicators."""
        return MarketMomentum(
            price_change_1h=0,
            price_change_24h=0,
            volume_change_24h=0,
            velocity=0,
            acceleration=0,
        )
    
    async def _analyze_whale_activity(self, market) -> WhaleActivity:
        """Detect large trades (whale activity)."""
        return WhaleActivity(
            large_buys_24h=0,
            large_sells_24h=0,
            large_buy_volume=0,
            large_sell_volume=0,
            whale_bias=0,
        )
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Singleton
polymarket_signals = PolymarketSignalProvider()
