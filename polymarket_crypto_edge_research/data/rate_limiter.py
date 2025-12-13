"""
Rate limiter with adaptive backoff and token bucket algorithm.
"""

import asyncio
import time
from dataclasses import dataclass, field

from core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Args:
        requests_per_minute: Maximum requests per minute
        requests_per_day: Maximum requests per day
        burst_size: Maximum burst size (defaults to rpm)
    """
    
    requests_per_minute: int = 60
    requests_per_day: int = 10000
    burst_size: int | None = None
    
    _minute_tokens: float = field(init=False, default=0)
    _day_tokens: float = field(init=False, default=0)
    _last_refill: float = field(init=False, default=0)
    _day_start: float = field(init=False, default=0)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    
    def __post_init__(self) -> None:
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute
        self._minute_tokens = float(self.burst_size)
        self._day_tokens = float(self.requests_per_day)
        self._last_refill = time.monotonic()
        self._day_start = time.time()
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        
        # Refill minute tokens
        refill_rate = self.requests_per_minute / 60.0  # tokens per second
        self._minute_tokens = min(
            self.burst_size,
            self._minute_tokens + elapsed * refill_rate
        )
        
        # Check if day rolled over
        current_time = time.time()
        if current_time - self._day_start >= 86400:
            self._day_tokens = float(self.requests_per_day)
            self._day_start = current_time
        
        self._last_refill = now
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
        """
        async with self._lock:
            total_wait = 0.0
            
            while True:
                self._refill_tokens()
                
                # Check daily limit
                if self._day_tokens < tokens:
                    logger.warning(f"Daily rate limit reached ({self.requests_per_day}/day)")
                    raise RateLimitExceeded("Daily rate limit exceeded")
                
                # Check minute limit
                if self._minute_tokens >= tokens:
                    self._minute_tokens -= tokens
                    self._day_tokens -= tokens
                    return total_wait
                
                # Calculate wait time
                tokens_needed = tokens - self._minute_tokens
                refill_rate = self.requests_per_minute / 60.0
                wait_time = tokens_needed / refill_rate + 0.1
                
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                total_wait += wait_time
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Returns:
            True if tokens acquired, False otherwise
        """
        self._refill_tokens()
        
        if self._minute_tokens >= tokens and self._day_tokens >= tokens:
            self._minute_tokens -= tokens
            self._day_tokens -= tokens
            return True
        return False
    
    @property
    def available_tokens(self) -> int:
        """Get currently available tokens."""
        self._refill_tokens()
        return int(min(self._minute_tokens, self._day_tokens))


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


@dataclass
class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts based on API responses.
    Backs off on 429s, speeds up on success.
    """
    
    min_rpm: int = 10
    max_rpm: int = 120
    backoff_factor: float = 0.5
    recovery_factor: float = 1.1
    
    _consecutive_success: int = field(init=False, default=0)
    _current_rpm: float = field(init=False, default=0)
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self._current_rpm = float(self.requests_per_minute)
    
    def on_success(self) -> None:
        """Call on successful API response."""
        self._consecutive_success += 1
        
        # Speed up after 10 consecutive successes
        if self._consecutive_success >= 10:
            new_rpm = min(self.max_rpm, self._current_rpm * self.recovery_factor)
            if new_rpm != self._current_rpm:
                logger.debug(f"Rate limit recovery: {self._current_rpm:.0f} -> {new_rpm:.0f} RPM")
                self._current_rpm = new_rpm
                self.requests_per_minute = int(new_rpm)
            self._consecutive_success = 0
    
    def on_rate_limit(self) -> None:
        """Call when rate limited (429 response)."""
        self._consecutive_success = 0
        new_rpm = max(self.min_rpm, self._current_rpm * self.backoff_factor)
        logger.warning(f"Rate limit hit, backing off: {self._current_rpm:.0f} -> {new_rpm:.0f} RPM")
        self._current_rpm = new_rpm
        self.requests_per_minute = int(new_rpm)
    
    def on_error(self) -> None:
        """Call on API error (not rate limit)."""
        self._consecutive_success = 0


class MultiEndpointRateLimiter:
    """
    Manages rate limits across multiple API endpoints.
    """
    
    def __init__(self) -> None:
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()
    
    def register(
        self,
        endpoint: str,
        requests_per_minute: int = 60,
        requests_per_day: int = 10000,
        adaptive: bool = False
    ) -> None:
        """Register a rate limiter for an endpoint."""
        if adaptive:
            self._limiters[endpoint] = AdaptiveRateLimiter(
                requests_per_minute=requests_per_minute,
                requests_per_day=requests_per_day
            )
        else:
            self._limiters[endpoint] = RateLimiter(
                requests_per_minute=requests_per_minute,
                requests_per_day=requests_per_day
            )
    
    async def acquire(self, endpoint: str, tokens: int = 1) -> float:
        """Acquire tokens for an endpoint."""
        if endpoint not in self._limiters:
            # Default limiter
            self._limiters[endpoint] = RateLimiter()
        
        return await self._limiters[endpoint].acquire(tokens)
    
    def get_limiter(self, endpoint: str) -> RateLimiter | None:
        """Get the rate limiter for an endpoint."""
        return self._limiters.get(endpoint)
