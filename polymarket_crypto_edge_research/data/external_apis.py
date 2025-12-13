"""
External API clients for market data.
CoinGecko (DEX prices/vol), CoinMarketCap (OI/funding), GeckoTerminal (DEX trends).
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from core.config import get_settings
from core.logging_utils import get_logger
from core.time_utils import now_utc

logger = get_logger(__name__)


class CoinGeckoData(BaseModel):
    """Data from CoinGecko API."""
    symbol: str
    price_usd: float
    market_cap: float = 0
    volume_24h: float = 0
    price_change_24h_pct: float = 0
    price_change_7d_pct: float = 0
    ath: float = 0
    ath_change_pct: float = 0
    timestamp: datetime = Field(default_factory=now_utc)


class DEXPoolData(BaseModel):
    """DEX pool/pair data."""
    pool_address: str
    dex_name: str
    base_token: str
    quote_token: str
    price_usd: float
    liquidity_usd: float
    volume_24h: float
    price_change_24h_pct: float = 0
    txns_24h: int = 0
    timestamp: datetime = Field(default_factory=now_utc)


class FundingOIData(BaseModel):
    """Funding and Open Interest data from CMC/aggregators."""
    symbol: str
    total_open_interest: float = 0
    oi_change_24h_pct: float = 0
    avg_funding_rate: float = 0
    long_short_ratio: float = 1.0
    liquidations_24h: float = 0
    timestamp: datetime = Field(default_factory=now_utc)


class ExternalAPIsClient:
    """
    Client for external market data APIs.
    All free tier, rate-limited appropriately.
    """
    
    # CoinGecko IDs
    CG_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "XRP": "ripple"
    }
    
    # Rate limits (requests per minute)
    RATE_LIMITS = {
        "coingecko": 30,  # Free tier ~30/min
        "cmc": 30,
        "geckoterminal": 30
    }
    
    def __init__(self):
        self._http = httpx.AsyncClient(timeout=30.0)
        self._last_calls: dict[str, datetime] = {}
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = 60  # 1 minute cache
    
    async def __aenter__(self) -> "ExternalAPIsClient":
        return self
    
    async def __aexit__(self, *args) -> None:
        await self._http.aclose()
    
    async def _rate_limit(self, api: str) -> None:
        """Enforce rate limiting."""
        limit = self.RATE_LIMITS.get(api, 30)
        min_interval = 60 / limit
        
        last = self._last_calls.get(api)
        if last:
            elapsed = (now_utc() - last).total_seconds()
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        
        self._last_calls[api] = now_utc()
    
    def _get_cache(self, key: str) -> Any | None:
        if key in self._cache:
            value, ts = self._cache[key]
            if (now_utc() - ts).total_seconds() < self._cache_ttl:
                return value
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (value, now_utc())
    
    # ==================== CoinGecko ====================
    
    async def get_coingecko_price(self, symbol: str) -> CoinGeckoData | None:
        """Get price data from CoinGecko."""
        cache_key = f"cg:{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        cg_id = self.CG_IDS.get(symbol.upper())
        if not cg_id:
            return None
        
        await self._rate_limit("coingecko")
        
        try:
            resp = await self._http.get(
                f"https://api.coingecko.com/api/v3/coins/{cg_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "false",
                    "developer_data": "false"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            market = data.get("market_data", {})
            
            result = CoinGeckoData(
                symbol=symbol.upper(),
                price_usd=market.get("current_price", {}).get("usd", 0),
                market_cap=market.get("market_cap", {}).get("usd", 0),
                volume_24h=market.get("total_volume", {}).get("usd", 0),
                price_change_24h_pct=market.get("price_change_percentage_24h", 0) or 0,
                price_change_7d_pct=market.get("price_change_percentage_7d", 0) or 0,
                ath=market.get("ath", {}).get("usd", 0),
                ath_change_pct=market.get("ath_change_percentage", {}).get("usd", 0) or 0
            )
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"CoinGecko error for {symbol}: {e}")
            return None
    
    async def get_all_coingecko_prices(
        self,
        symbols: list[str] = ["BTC", "ETH", "SOL", "XRP"]
    ) -> dict[str, CoinGeckoData]:
        """Get prices for multiple symbols."""
        # Use simple/price endpoint for efficiency
        cache_key = "cg:all"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        await self._rate_limit("coingecko")
        
        ids = [self.CG_IDS[s] for s in symbols if s in self.CG_IDS]
        
        try:
            resp = await self._http.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": ",".join(ids),
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true",
                    "include_market_cap": "true"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = {}
            for symbol in symbols:
                cg_id = self.CG_IDS.get(symbol)
                if cg_id and cg_id in data:
                    coin = data[cg_id]
                    results[symbol] = CoinGeckoData(
                        symbol=symbol,
                        price_usd=coin.get("usd", 0),
                        volume_24h=coin.get("usd_24h_vol", 0),
                        price_change_24h_pct=coin.get("usd_24h_change", 0) or 0,
                        market_cap=coin.get("usd_market_cap", 0)
                    )
            
            self._set_cache(cache_key, results)
            return results
            
        except Exception as e:
            logger.error(f"CoinGecko batch error: {e}")
            return {}
    
    # ==================== GeckoTerminal (DEX) ====================
    
    async def get_dex_pools(
        self,
        network: str = "eth",
        limit: int = 10
    ) -> list[DEXPoolData]:
        """Get top DEX pools from GeckoTerminal."""
        cache_key = f"gt:pools:{network}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        await self._rate_limit("geckoterminal")
        
        try:
            resp = await self._http.get(
                f"https://api.geckoterminal.com/api/v2/networks/{network}/trending_pools",
                params={"page": 1}
            )
            resp.raise_for_status()
            data = resp.json()
            
            pools = []
            for item in data.get("data", [])[:limit]:
                attrs = item.get("attributes", {})
                pools.append(DEXPoolData(
                    pool_address=attrs.get("address", ""),
                    dex_name=attrs.get("dex_id", ""),
                    base_token=attrs.get("base_token_symbol", ""),
                    quote_token=attrs.get("quote_token_symbol", ""),
                    price_usd=float(attrs.get("base_token_price_usd", 0) or 0),
                    liquidity_usd=float(attrs.get("reserve_in_usd", 0) or 0),
                    volume_24h=float(attrs.get("volume_usd", {}).get("h24", 0) or 0),
                    price_change_24h_pct=float(attrs.get("price_change_percentage", {}).get("h24", 0) or 0),
                    txns_24h=int(attrs.get("transactions", {}).get("h24", {}).get("total", 0) or 0)
                ))
            
            self._set_cache(cache_key, pools)
            return pools
            
        except Exception as e:
            logger.error(f"GeckoTerminal error: {e}")
            return []
    
    async def get_token_dex_data(
        self,
        symbol: str,
        network: str = "eth"
    ) -> list[DEXPoolData]:
        """Get DEX pools for a specific token."""
        # Map symbols to addresses (mainnet)
        token_addresses = {
            "eth": {
                "ETH": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
                "BTC": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
            },
            "solana": {
                "SOL": "So11111111111111111111111111111111111111112",
            }
        }
        
        address = token_addresses.get(network, {}).get(symbol)
        if not address:
            return []
        
        cache_key = f"gt:token:{network}:{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        await self._rate_limit("geckoterminal")
        
        try:
            resp = await self._http.get(
                f"https://api.geckoterminal.com/api/v2/networks/{network}/tokens/{address}/pools",
                params={"page": 1}
            )
            resp.raise_for_status()
            data = resp.json()
            
            pools = []
            for item in data.get("data", [])[:5]:
                attrs = item.get("attributes", {})
                pools.append(DEXPoolData(
                    pool_address=attrs.get("address", ""),
                    dex_name=attrs.get("dex_id", ""),
                    base_token=attrs.get("base_token_symbol", ""),
                    quote_token=attrs.get("quote_token_symbol", ""),
                    price_usd=float(attrs.get("base_token_price_usd", 0) or 0),
                    liquidity_usd=float(attrs.get("reserve_in_usd", 0) or 0),
                    volume_24h=float(attrs.get("volume_usd", {}).get("h24", 0) or 0),
                    price_change_24h_pct=float(attrs.get("price_change_percentage", {}).get("h24", 0) or 0)
                ))
            
            self._set_cache(cache_key, pools)
            return pools
            
        except Exception as e:
            logger.error(f"GeckoTerminal token error: {e}")
            return []
    
    # ==================== Funding/OI (aggregated) ====================
    
    async def get_funding_oi(self, symbol: str) -> FundingOIData | None:
        """
        Get funding and OI data.
        Uses CoinGlass-style free endpoints or CoinGecko derivatives.
        """
        cache_key = f"funding:{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        # Try CoinGecko derivatives endpoint
        await self._rate_limit("coingecko")
        
        try:
            resp = await self._http.get(
                "https://api.coingecko.com/api/v3/derivatives",
                params={"per_page": 100}
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Filter for symbol
            symbol_lower = symbol.lower()
            relevant = [
                d for d in data 
                if symbol_lower in d.get("symbol", "").lower()
            ]
            
            if relevant:
                total_oi = sum(d.get("open_interest", 0) or 0 for d in relevant)
                avg_funding = sum(
                    d.get("funding_rate", 0) or 0 for d in relevant
                ) / len(relevant) if relevant else 0
                
                result = FundingOIData(
                    symbol=symbol.upper(),
                    total_open_interest=total_oi,
                    avg_funding_rate=avg_funding
                )
                
                self._set_cache(cache_key, result)
                return result
            
        except Exception as e:
            logger.debug(f"Funding/OI error for {symbol}: {e}")
        
        return FundingOIData(symbol=symbol.upper())
    
    # ==================== Aggregated Market Data ====================
    
    async def get_market_overview(
        self,
        symbols: list[str] = ["BTC", "ETH", "SOL", "XRP"]
    ) -> dict[str, dict]:
        """Get comprehensive market overview for all symbols."""
        results = {}
        
        # Fetch all data in parallel
        cg_task = self.get_all_coingecko_prices(symbols)
        eth_pools_task = self.get_dex_pools("eth", limit=5)
        sol_pools_task = self.get_dex_pools("solana", limit=5)
        
        cg_data, eth_pools, sol_pools = await asyncio.gather(
            cg_task, eth_pools_task, sol_pools_task,
            return_exceptions=True
        )
        
        # Process CoinGecko data
        if isinstance(cg_data, dict):
            for symbol, data in cg_data.items():
                results[symbol] = {
                    "price": data.price_usd,
                    "volume_24h": data.volume_24h,
                    "change_24h_pct": data.price_change_24h_pct,
                    "market_cap": data.market_cap,
                    "source": "coingecko"
                }
        
        # Add DEX pool summaries
        dex_summary = {
            "eth_pools": len(eth_pools) if isinstance(eth_pools, list) else 0,
            "sol_pools": len(sol_pools) if isinstance(sol_pools, list) else 0,
            "total_eth_liquidity": sum(
                p.liquidity_usd for p in (eth_pools if isinstance(eth_pools, list) else [])
            ),
            "total_sol_liquidity": sum(
                p.liquidity_usd for p in (sol_pools if isinstance(sol_pools, list) else [])
            )
        }
        results["_dex_summary"] = dex_summary
        
        # Fetch funding for each
        funding_tasks = [self.get_funding_oi(s) for s in symbols]
        funding_results = await asyncio.gather(*funding_tasks, return_exceptions=True)
        
        for symbol, funding in zip(symbols, funding_results):
            if isinstance(funding, FundingOIData) and symbol in results:
                results[symbol]["open_interest"] = funding.total_open_interest
                results[symbol]["avg_funding"] = funding.avg_funding_rate
        
        return results


# Singleton
_external_apis: ExternalAPIsClient | None = None


async def get_external_apis() -> ExternalAPIsClient:
    """Get or create external APIs client."""
    global _external_apis
    if _external_apis is None:
        _external_apis = ExternalAPIsClient()
    return _external_apis
