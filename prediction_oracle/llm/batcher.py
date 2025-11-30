"""LLM batching, caching, and rate limiting."""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from .providers import LLMProvider, LLMQuery, LLMResponse

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for LLM responses."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove oldest item
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, calls_per_minute: int = 10):
        self.calls_per_minute = calls_per_minute
        self.tokens = calls_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Refill tokens based on time elapsed
            self.tokens = min(
                self.calls_per_minute,
                self.tokens + (elapsed * self.calls_per_minute / 60.0)
            )
            self.last_update = now
            
            # Wait if no tokens available
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) * 60.0 / self.calls_per_minute
                logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 1.0
            
            self.tokens -= 1.0


class LLMBatcher:
    """
    Batches LLM queries and manages caching and rate limiting.
    
    Optimizes LLM API usage by batching queries, caching results,
    and respecting rate limits.
    """

    def __init__(
        self,
        provider: LLMProvider,
        batch_size: int = 10,
        cache_ttl_seconds: int = 600,
        rate_limit_per_minute: int = 10,
    ):
        """
        Initialize LLM batcher.
        
        Args:
            provider: LLM provider to use
            batch_size: Maximum queries per batch
            cache_ttl_seconds: Time-to-live for cache entries
            rate_limit_per_minute: Maximum API calls per minute
        """
        self.provider = provider
        self.batch_size = batch_size
        self.cache_ttl_seconds = cache_ttl_seconds
        
        self.cache = LRUCache(capacity=1000)
        self.rate_limiter = RateLimiter(calls_per_minute=rate_limit_per_minute)
        self.cache_timestamps: dict[str, float] = {}

    def _cache_key(self, query: LLMQuery) -> str:
        """Generate cache key for a query."""
        content = f"{self.provider.name}:{query.prompt}:{query.temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self.cache_timestamps:
            return False
        
        age = time.time() - self.cache_timestamps[key]
        return age < self.cache_ttl_seconds

    async def batch_generate(
        self,
        queries: list[LLMQuery],
        use_cache: bool = True,
    ) -> list[LLMResponse]:
        """
        Generate responses for a batch of queries with caching.
        
        Args:
            queries: List of queries to process
            use_cache: Whether to use cached responses
            
        Returns:
            List of responses in the same order as queries
        """
        responses: list[LLMResponse | None] = [None] * len(queries)
        queries_to_run: list[tuple[int, LLMQuery]] = []
        
        # Check cache first
        if use_cache:
            for i, query in enumerate(queries):
                cache_key = self._cache_key(query)
                if self._is_cache_valid(cache_key):
                    cached = self.cache.get(cache_key)
                    if cached:
                        logger.debug(f"Cache hit for query {query.id}")
                        responses[i] = cached
                        continue
                
                queries_to_run.append((i, query))
        else:
            queries_to_run = list(enumerate(queries))
        
        if not queries_to_run:
            return [r for r in responses if r is not None]
        
        # Process queries in batches with rate limiting
        for batch_start in range(0, len(queries_to_run), self.batch_size):
            batch = queries_to_run[batch_start : batch_start + self.batch_size]
            
            # Wait for rate limiter
            await self.rate_limiter.acquire()
            
            # Extract just the queries
            batch_queries = [q for _, q in batch]
            
            logger.info(
                f"Calling {self.provider.name} with {len(batch_queries)} queries"
            )
            
            try:
                batch_responses = await self.provider.generate(batch_queries)
                
                # Store responses and update cache
                for (original_idx, query), response in zip(batch, batch_responses):
                    responses[original_idx] = response
                    
                    if use_cache and not response.error:
                        cache_key = self._cache_key(query)
                        self.cache.put(cache_key, response)
                        self.cache_timestamps[cache_key] = time.time()
                
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                # Fill with error responses
                for original_idx, query in batch:
                    responses[original_idx] = LLMResponse(
                        id=query.id,
                        raw_text="",
                        error=f"Batch error: {str(e)}",
                    )
        
        return [r for r in responses if r is not None]

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")
