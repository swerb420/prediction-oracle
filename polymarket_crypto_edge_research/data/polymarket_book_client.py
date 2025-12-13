"""
Polymarket CLOB (Order Book) client.
Fetches real-time orderbook depth and simulates fills.
"""

import asyncio
import json
from datetime import datetime
from typing import Callable

import httpx
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logging_utils import get_logger
from core.time_utils import now_utc
from .rate_limiter import AdaptiveRateLimiter
from .schemas import OrderBook, OrderBookLevel

logger = get_logger(__name__)


class PolymarketBookClient:
    """
    Client for Polymarket CLOB (Central Limit Order Book).
    Fetches orderbook snapshots and simulates trade execution.
    """
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.polymarket_clob_url
        self._client = httpx.AsyncClient(timeout=30.0)
        self._rate_limiter = AdaptiveRateLimiter(
            requests_per_minute=120,
            requests_per_day=10000
        )
        self._ws_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._running = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_orderbook(
        self,
        token_id: str,
        depth: int = 20
    ) -> OrderBook:
        """
        Fetch orderbook snapshot for a token.
        
        Args:
            token_id: Token/outcome ID
            depth: Number of levels per side
            
        Returns:
            OrderBook object
        """
        await self._rate_limiter.acquire()
        
        resp = await self._client.get(
            f"{self.base_url}/book",
            params={"token_id": token_id}
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        data = resp.json()
        
        bids = []
        asks = []
        
        for bid in data.get("bids", [])[:depth]:
            bids.append(OrderBookLevel(
                price=float(bid.get("price", 0)),
                size=float(bid.get("size", 0))
            ))
        
        for ask in data.get("asks", [])[:depth]:
            asks.append(OrderBookLevel(
                price=float(ask.get("price", 0)),
                size=float(ask.get("size", 0))
            ))
        
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return OrderBook(
            symbol=token_id,
            timestamp=now_utc(),
            bids=bids,
            asks=asks
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_market_info(self, token_id: str) -> dict:
        """Get market/token metadata."""
        await self._rate_limiter.acquire()
        
        resp = await self._client.get(
            f"{self.base_url}/markets/{token_id}"
        )
        
        if resp.status_code == 404:
            return {}
        
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        return resp.json()
    
    async def get_mid_price(self, token_id: str) -> float | None:
        """Get mid price for a token."""
        book = await self.get_orderbook(token_id, depth=1)
        return book.mid_price
    
    async def get_spread(self, token_id: str) -> tuple[float | None, float | None]:
        """Get spread (absolute and bps) for a token."""
        book = await self.get_orderbook(token_id, depth=1)
        return book.spread, book.spread_bps
    
    def simulate_market_buy(
        self,
        orderbook: OrderBook,
        size: float
    ) -> tuple[float, float, float]:
        """
        Simulate a market buy order.
        
        Args:
            orderbook: Current orderbook
            size: Size to buy
            
        Returns:
            (avg_price, filled_size, slippage_pct)
        """
        if not orderbook.asks:
            return 0.0, 0.0, 0.0
        
        remaining = size
        total_cost = 0.0
        filled = 0.0
        
        for level in orderbook.asks:
            if remaining <= 0:
                break
            
            fill_size = min(remaining, level.size)
            total_cost += fill_size * level.price
            filled += fill_size
            remaining -= fill_size
        
        if filled == 0:
            return 0.0, 0.0, 0.0
        
        avg_price = total_cost / filled
        best_ask = orderbook.asks[0].price
        slippage = (avg_price - best_ask) / best_ask if best_ask > 0 else 0
        
        return avg_price, filled, slippage
    
    def simulate_market_sell(
        self,
        orderbook: OrderBook,
        size: float
    ) -> tuple[float, float, float]:
        """
        Simulate a market sell order.
        
        Args:
            orderbook: Current orderbook
            size: Size to sell
            
        Returns:
            (avg_price, filled_size, slippage_pct)
        """
        if not orderbook.bids:
            return 0.0, 0.0, 0.0
        
        remaining = size
        total_proceeds = 0.0
        filled = 0.0
        
        for level in orderbook.bids:
            if remaining <= 0:
                break
            
            fill_size = min(remaining, level.size)
            total_proceeds += fill_size * level.price
            filled += fill_size
            remaining -= fill_size
        
        if filled == 0:
            return 0.0, 0.0, 0.0
        
        avg_price = total_proceeds / filled
        best_bid = orderbook.bids[0].price
        slippage = (best_bid - avg_price) / best_bid if best_bid > 0 else 0
        
        return avg_price, filled, slippage
    
    def estimate_fill_probability(
        self,
        orderbook: OrderBook,
        side: str,
        price: float,
        size: float
    ) -> float:
        """
        Estimate probability of limit order fill.
        
        Args:
            orderbook: Current orderbook
            side: "buy" or "sell"
            price: Limit price
            size: Order size
            
        Returns:
            Estimated fill probability (0-1)
        """
        if side == "buy":
            # Buy limit order - need price to drop to our level
            if not orderbook.asks or price >= orderbook.asks[0].price:
                return 0.95  # Immediately fillable
            
            # Check depth at our price level
            depth_above = sum(
                level.size for level in orderbook.asks
                if level.price > price
            )
            
            # Heuristic: more depth above = lower fill probability
            fill_prob = max(0.1, 1.0 - (depth_above / (depth_above + size)))
            return fill_prob
            
        else:  # sell
            if not orderbook.bids or price <= orderbook.bids[0].price:
                return 0.95
            
            depth_below = sum(
                level.size for level in orderbook.bids
                if level.price < price
            )
            
            fill_prob = max(0.1, 1.0 - (depth_below / (depth_below + size)))
            return fill_prob
    
    def calculate_impact(
        self,
        orderbook: OrderBook,
        side: str,
        size: float
    ) -> dict:
        """
        Calculate market impact of a trade.
        
        Returns:
            Dict with impact metrics
        """
        if side == "buy":
            avg_price, filled, slippage = self.simulate_market_buy(orderbook, size)
            reference_price = orderbook.best_ask or 0
        else:
            avg_price, filled, slippage = self.simulate_market_sell(orderbook, size)
            reference_price = orderbook.best_bid or 0
        
        # Calculate depth consumed
        if side == "buy":
            total_ask_depth = sum(l.size for l in orderbook.asks)
            depth_pct = filled / total_ask_depth if total_ask_depth > 0 else 1
        else:
            total_bid_depth = sum(l.size for l in orderbook.bids)
            depth_pct = filled / total_bid_depth if total_bid_depth > 0 else 1
        
        return {
            "avg_price": avg_price,
            "filled_size": filled,
            "slippage_pct": slippage,
            "depth_consumed_pct": depth_pct,
            "unfilled_size": size - filled,
            "reference_price": reference_price
        }
    
    async def subscribe_orderbook(
        self,
        token_id: str,
        callback: Callable[[OrderBook], None]
    ) -> None:
        """
        Subscribe to real-time orderbook updates via WebSocket.
        
        Args:
            token_id: Token to subscribe to
            callback: Function to call with orderbook updates
        """
        ws_url = f"{self.base_url.replace('https', 'wss')}/ws"
        
        self._running = True
        
        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws_connections[token_id] = ws
                    
                    # Subscribe to orderbook channel
                    subscribe_msg = {
                        "type": "subscribe",
                        "channel": "book",
                        "market": token_id
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    logger.info(f"Subscribed to orderbook for {token_id}")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        data = json.loads(message)
                        
                        if data.get("type") == "book":
                            book = self._parse_ws_book(data, token_id)
                            callback(book)
                            
            except websockets.ConnectionClosed:
                logger.warning(f"Orderbook WS closed for {token_id}, reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Orderbook WS error: {e}")
                await asyncio.sleep(5)
    
    def _parse_ws_book(self, data: dict, token_id: str) -> OrderBook:
        """Parse WebSocket orderbook message."""
        bids = []
        asks = []
        
        for bid in data.get("bids", []):
            bids.append(OrderBookLevel(
                price=float(bid[0]),
                size=float(bid[1])
            ))
        
        for ask in data.get("asks", []):
            asks.append(OrderBookLevel(
                price=float(ask[0]),
                size=float(ask[1])
            ))
        
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return OrderBook(
            symbol=token_id,
            timestamp=now_utc(),
            bids=bids,
            asks=asks
        )
    
    async def close(self) -> None:
        """Close all connections."""
        self._running = False
        
        for token_id, ws in self._ws_connections.items():
            await ws.close()
        
        await self._client.aclose()


# Singleton instance
_book_client: PolymarketBookClient | None = None


def get_book_client() -> PolymarketBookClient:
    """Get or create global Book client."""
    global _book_client
    if _book_client is None:
        _book_client = PolymarketBookClient()
    return _book_client
