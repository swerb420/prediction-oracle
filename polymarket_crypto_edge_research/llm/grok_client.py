"""
Grok 4.1 API client for LLM-enhanced features.
Handles rate limiting, retries, and JSON response parsing.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from core.config import get_settings
from core.logging_utils import get_logger

logger = get_logger(__name__)


class GrokRequest(BaseModel):
    """Request to Grok API."""
    
    prompt: str
    system_prompt: str | None = None
    max_tokens: int = 500
    temperature: float = 0.1  # Low for structured output
    
    # Metadata
    request_type: str = "general"  # regime, sentiment, clustering
    symbol: str | None = None


class GrokResponse(BaseModel):
    """Response from Grok API."""
    
    # Raw response
    content: str
    
    # Parsed JSON (if applicable)
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    
    # Metrics
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    
    # Status
    success: bool = True
    error: str | None = None
    
    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


class GrokClient:
    """
    Async client for xAI Grok 4.1 API.
    
    Features:
    - Automatic JSON parsing
    - Rate limiting
    - Retry with exponential backoff
    - Token tracking for cost management
    """
    
    # API endpoints
    BASE_URL = "https://api.x.ai/v1"
    CHAT_ENDPOINT = "/chat/completions"
    
    # Models
    MODEL_FAST = "grok-3-fast"  # Lower cost, faster
    MODEL_DEFAULT = "grok-3"    # Higher quality
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL_FAST,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float = 30.0,
        max_tokens_per_minute: int = 50000
    ):
        self.api_key = api_key or get_settings().xai_api_key
        if not self.api_key:
            raise ValueError("XAI_API_KEY not set")
        
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.max_tokens_per_minute = max_tokens_per_minute
        
        # Rate limiting
        self._token_bucket = max_tokens_per_minute
        self._last_refill = time.time()
        self._lock = asyncio.Lock()
        
        # Stats
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0
    
    async def _refill_bucket(self) -> None:
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        refill = int(elapsed * self.max_tokens_per_minute / 60)
        
        if refill > 0:
            self._token_bucket = min(
                self._token_bucket + refill,
                self.max_tokens_per_minute
            )
            self._last_refill = now
    
    async def _wait_for_tokens(self, estimated_tokens: int) -> None:
        """Wait for rate limit bucket to have enough tokens."""
        async with self._lock:
            await self._refill_bucket()
            
            while self._token_bucket < estimated_tokens:
                wait_time = (estimated_tokens - self._token_bucket) * 60 / self.max_tokens_per_minute
                logger.debug(f"Rate limited, waiting {wait_time:.1f}s")
                await asyncio.sleep(min(wait_time, 5.0))
                await self._refill_bucket()
            
            self._token_bucket -= estimated_tokens
    
    async def complete(
        self,
        request: GrokRequest
    ) -> GrokResponse:
        """
        Send completion request to Grok API.
        
        Args:
            request: GrokRequest with prompt
            
        Returns:
            GrokResponse with parsed content
        """
        # Estimate tokens for rate limiting
        estimated_tokens = len(request.prompt) // 4 + request.max_tokens
        await self._wait_for_tokens(estimated_tokens)
        
        # Build messages
        messages = []
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        messages.append({
            "role": "user",
            "content": request.prompt
        })
        
        # Build request body
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        # Make request with retries
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.BASE_URL}{self.CHAT_ENDPOINT}",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=body
                    )
                    response.raise_for_status()
                    data = response.json()
                
                # Extract content
                content = data["choices"][0]["message"]["content"]
                
                # Extract token usage
                usage = data.get("usage", {})
                tokens_input = usage.get("prompt_tokens", 0)
                tokens_output = usage.get("completion_tokens", 0)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Update stats
                self._total_requests += 1
                self._total_tokens += tokens_input + tokens_output
                
                # Try to parse JSON
                parsed, parse_error = self._parse_json(content)
                
                return GrokResponse(
                    content=content,
                    parsed=parsed,
                    parse_error=parse_error,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    latency_ms=latency_ms,
                    success=True
                )
                
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                
                if e.response.status_code == 429:  # Rate limited
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s")
                    await asyncio.sleep(delay)
                elif e.response.status_code >= 500:  # Server error
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Server error, retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    break
                    
            except Exception as e:
                last_error = str(e)
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Request failed: {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
        
        # All retries failed
        self._total_errors += 1
        logger.error(f"Grok request failed after {self.max_retries} retries: {last_error}")
        
        return GrokResponse(
            content="",
            success=False,
            error=last_error,
            latency_ms=(time.time() - start_time) * 1000
        )
    
    def _parse_json(self, content: str) -> tuple[dict[str, Any] | None, str | None]:
        """
        Parse JSON from response content.
        Handles various formats (raw JSON, markdown code blocks, etc.)
        """
        # Try direct parse
        try:
            return json.loads(content), None
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1)), None
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in content
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0)), None
            except json.JSONDecodeError:
                pass
        
        return None, f"Could not parse JSON from response: {content[:100]}"
    
    async def batch_complete(
        self,
        requests: list[GrokRequest],
        concurrency: int = 3
    ) -> list[GrokResponse]:
        """
        Process multiple requests with controlled concurrency.
        
        Args:
            requests: List of requests
            concurrency: Max concurrent requests
            
        Returns:
            List of responses in same order
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_complete(request: GrokRequest) -> GrokResponse:
            async with semaphore:
                return await self.complete(request)
        
        return await asyncio.gather(
            *[limited_complete(r) for r in requests]
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(1, self._total_requests),
            "avg_tokens_per_request": self._total_tokens / max(1, self._total_requests)
        }


# System prompts for different use cases
SYSTEM_PROMPTS = {
    "regime": """You are a quantitative analyst specializing in crypto market regime classification.
Analyze the provided market data and classify the current regime.
Always respond with valid JSON only. No explanations outside the JSON.""",

    "sentiment": """You are a crypto market sentiment analyst.
Score the sentiment based on the provided context.
Always respond with valid JSON only. No explanations outside the JSON.""",

    "clustering": """You are a prediction market analyst.
Group semantically similar or duplicate markets together.
Always respond with valid JSON only. No explanations outside the JSON."""
}


def create_grok_client(
    model: str = GrokClient.MODEL_FAST,
    **kwargs
) -> GrokClient:
    """Factory function to create Grok client."""
    return GrokClient(model=model, **kwargs)


async def quick_complete(
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 500
) -> GrokResponse:
    """
    Quick one-off completion without managing client lifecycle.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        max_tokens: Max response tokens
        
    Returns:
        GrokResponse
    """
    client = create_grok_client()
    request = GrokRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens
    )
    return await client.complete(request)
