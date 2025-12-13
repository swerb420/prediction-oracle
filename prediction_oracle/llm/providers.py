"""LLM provider interfaces and implementations with rate limiting."""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)


class LLMQuery(BaseModel):
    """A query to send to an LLM."""

    id: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 8000  # Increased for longer JSON responses


class LLMResponse(BaseModel):
    """Response from an LLM."""

    id: str
    raw_text: str
    parsed_json: dict[str, Any] | list | None = None  # Can be dict OR list
    error: str | None = None
    model_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 10, requests_per_day: int = 1000):
        self.rpm = requests_per_minute
        self.rpd = requests_per_day
        self.minute_requests: list[float] = []
        self.day_requests: list[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until we can make a request."""
        async with self._lock:
            now = time.time()
            minute_ago = now - 60
            day_ago = now - 86400
            
            # Clean old entries
            self.minute_requests = [t for t in self.minute_requests if t > minute_ago]
            self.day_requests = [t for t in self.day_requests if t > day_ago]
            
            # Check RPM limit
            if len(self.minute_requests) >= self.rpm:
                wait_time = self.minute_requests[0] - minute_ago + 0.5
                logger.info(f"Rate limit: waiting {wait_time:.1f}s (RPM)")
                await asyncio.sleep(wait_time)
                self.minute_requests = [t for t in self.minute_requests if t > time.time() - 60]
            
            # Check RPD limit
            if len(self.day_requests) >= self.rpd:
                logger.warning(f"Daily rate limit reached ({self.rpd} requests)")
                raise Exception(f"Daily rate limit reached")
            
            # Record this request
            self.minute_requests.append(time.time())
            self.day_requests.append(time.time())


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, name: str, rpm: int = 10, rpd: int = 1000):
        self.name = name
        self.client = httpx.AsyncClient(timeout=60.0)
        self.max_retries = 3
        self.retry_delay = 2.0
        self.rate_limiter = RateLimiter(rpm, rpd)

    @abstractmethod
    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        """Generate responses for a batch of queries."""
        pass

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    def _repair_truncated_json(self, text: str) -> dict[str, Any] | list | None:
        """Attempt to repair truncated JSON by closing brackets."""
        # Count open vs close brackets
        open_brackets = text.count('[') - text.count(']')
        open_braces = text.count('{') - text.count('}')
        
        # Try to close unclosed structures
        repaired = text.rstrip()
        
        # Remove trailing comma if present
        if repaired.endswith(','):
            repaired = repaired[:-1]
        
        # Close any unclosed strings (rough heuristic)
        quote_count = repaired.count('"')
        if quote_count % 2 == 1:
            repaired += '"'
        
        # Close braces and brackets
        for _ in range(open_braces):
            repaired += '}'
        for _ in range(open_brackets):
            repaired += ']'
        
        try:
            result = json.loads(repaired)
            logger.info(f"Successfully repaired truncated JSON ({len(result)} items)")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"JSON repair failed: {e}")
            return None

    def _parse_json_response(self, text: str) -> dict[str, Any] | list | None:
        """Attempt to extract and parse JSON from response text."""
        try:
            result = json.loads(text)
            # Return both dicts and lists - let caller handle
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            logger.debug(f"Found ```json block: start={start}, end={end}")
            if end > start:
                content = text[start:end].strip()
                logger.debug(f"Code block content ({len(content)} chars): {content[:100]}...")
                try:
                    result = json.loads(content)
                    logger.debug(f"Parsed from code block: {type(result)}, len={len(result) if isinstance(result, list) else 1}")
                    return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Code block parse failed: {e}")
            else:
                # No closing ```, response was truncated - try to fix it
                content = text[start:].strip()
                logger.debug(f"Truncated response, attempting repair ({len(content)} chars)")
                result = self._repair_truncated_json(content)
                if result:
                    return result
        
        # Also try ```\n blocks (some models don't include json after ```)
        if "```" in text:
            # Find first code block
            start = text.find("```")
            if start != -1:
                # Skip the opening ```
                content_start = text.find("\n", start) + 1
                end = text.find("```", content_start)
                if end > content_start:
                    content = text[content_start:end].strip()
                    logger.debug(f"Generic code block content ({len(content)} chars)")
                    try:
                        result = json.loads(content)
                        logger.debug(f"Parsed from generic block: {type(result)}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.debug(f"Generic block parse failed: {e}")
        
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                content = text[start : end + 1]
                logger.debug(f"Trying {start_char}...{end_char} extract ({len(content)} chars)")
                try:
                    result = json.loads(content)
                    logger.debug(f"Parsed from {start_char}{end_char} extract: {type(result)}")
                    return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Extract parse failed: {e}")

        logger.warning(f"Failed to parse JSON from response (length {len(text)}): {text[:200]}...")
        return None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self):
        super().__init__("openai", rpm=60, rpd=10000)
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.model = settings.openai_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
            return [
                LLMResponse(id=q.id, raw_text="", error="API key not configured")
                for q in queries
            ]

        responses = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                result = await self._call_api(query)
                responses.append(result)
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                responses.append(LLMResponse(id=query.id, raw_text="", error=str(e)))

        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query.prompt}],
            "temperature": query.temperature,
            "max_tokens": query.max_tokens,
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed_json = self._parse_json_response(raw_text)

        usage = data.get("usage", {})

        return LLMResponse(
            id=query.id,
            raw_text=raw_text,
            parsed_json=parsed_json,
            model_name=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self):
        super().__init__("anthropic", rpm=60, rpd=10000)
        self.api_key = settings.anthropic_api_key
        self.base_url = settings.anthropic_base_url
        self.model = settings.anthropic_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        if not self.api_key:
            logger.warning("Anthropic API key not configured")
            return [
                LLMResponse(id=q.id, raw_text="", error="API key not configured")
                for q in queries
            ]

        responses = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                result = await self._call_api(query)
                responses.append(result)
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                responses.append(LLMResponse(id=query.id, raw_text="", error=str(e)))

        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": query.max_tokens,
            "messages": [{"role": "user", "content": query.prompt}],
        }

        response = await self.client.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        raw_text = data["content"][0]["text"]
        parsed_json = self._parse_json_response(raw_text)

        usage = data.get("usage", {})

        return LLMResponse(
            id=query.id,
            raw_text=raw_text,
            parsed_json=parsed_json,
            model_name=self.model,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )


class GrokProvider(LLMProvider):
    """xAI Grok API provider."""

    def __init__(self):
        super().__init__("grok", rpm=60, rpd=10000)
        self.api_key = settings.xai_api_key
        self.base_url = settings.xai_base_url
        self.model = settings.xai_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        if not self.api_key:
            logger.warning("xAI API key not configured")
            return [
                LLMResponse(id=q.id, raw_text="", error="API key not configured")
                for q in queries
            ]

        responses = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                result = await self._call_api(query)
                responses.append(result)
            except Exception as e:
                logger.error(f"xAI API error: {e}")
                responses.append(LLMResponse(id=query.id, raw_text="", error=str(e)))

        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query.prompt}],
            "temperature": query.temperature,
            "max_tokens": query.max_tokens,
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed_json = self._parse_json_response(raw_text)

        usage = data.get("usage", {})

        return LLMResponse(
            id=query.id,
            raw_text=raw_text,
            parsed_json=parsed_json,
            model_name=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class GroqProvider(LLMProvider):
    """Groq - FREE and very fast Llama inference!
    
    Rate Limits (Free Tier):
    - 30 RPM (requests per minute)
    - 1,000 RPD (requests per day) 
    - 12,000 TPM (tokens per minute)
    - 100,000 TPD (tokens per day)
    """

    def __init__(self):
        # Conservative limits: 20 RPM to stay safe, 900 RPD
        super().__init__("groq", rpm=20, rpd=900)
        self.api_key = settings.groq_api_key
        self.base_url = settings.groq_base_url
        self.model = settings.groq_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        if not self.api_key:
            logger.warning("Groq API key not configured")
            return [LLMResponse(id=q.id, raw_text="", error="API key not configured") for q in queries]

        responses = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                result = await self._call_api(query)
                responses.append(result)
                # Extra delay between requests for free tier
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.error(f"Groq API error: {e}")
                responses.append(LLMResponse(id=query.id, raw_text="", error=str(e)))
        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query.prompt}],
            "temperature": query.temperature,
            "max_tokens": query.max_tokens,
        }
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                
                if response.status_code == 429:
                    # Rate limited - check if it's a short wait or daily limit
                    retry_after = int(response.headers.get("retry-after", 60))
                    if retry_after > 120:  # More than 2 minutes = daily limit hit
                        logger.warning(f"Groq daily limit hit (retry_after={retry_after}s), failing fast")
                        raise Exception(f"Groq daily rate limit exceeded, retry in {retry_after}s")
                    logger.warning(f"Groq rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                raw_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return LLMResponse(
                    id=query.id,
                    raw_text=raw_text,
                    parsed_json=self._parse_json_response(raw_text),
                    model_name=self.model,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
        
        raise Exception("Max retries exceeded")


class GeminiProvider(LLMProvider):
    """Google Gemini - FREE tier available!
    
    Rate Limits (Free Tier for Gemini 2.0 Flash):
    - ~15 RPM (requests per minute)
    - Free input/output tokens
    """

    def __init__(self):
        # Conservative: 10 RPM to be safe
        super().__init__("gemini", rpm=10, rpd=5000)
        self.api_key = settings.google_api_key
        self.model = settings.google_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        if not self.api_key:
            logger.warning("Google API key not configured")
            return [LLMResponse(id=q.id, raw_text="", error="API key not configured") for q in queries]

        responses = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                result = await self._call_api(query)
                responses.append(result)
                # Gemini has stricter rate limits - add delay
                await asyncio.sleep(4.0)
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                responses.append(LLMResponse(id=query.id, raw_text="", error=str(e)))
        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": query.prompt}]}],
            "generationConfig": {"temperature": query.temperature, "maxOutputTokens": query.max_tokens}
        }
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(url, json=payload)
                
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Gemini rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Handle Gemini response format
                if "candidates" not in data or not data["candidates"]:
                    return LLMResponse(id=query.id, raw_text="", error="No response from Gemini")
                
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return LLMResponse(
                    id=query.id,
                    raw_text=raw_text,
                    parsed_json=self._parse_json_response(raw_text),
                    model_name=self.model,
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1) * 10)
                    continue
                raise
        
        raise Exception("Max retries exceeded")


class OpenRouterProvider(LLMProvider):
    """OpenRouter - Access to many free models!"""

    def __init__(self):
        super().__init__("openrouter", rpm=10, rpd=1000)
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
            return [LLMResponse(id=q.id, raw_text="", error="API key not configured") for q in queries]

        responses = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                result = await self._call_api(query)
                responses.append(result)
                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f"OpenRouter API error: {e}")
                responses.append(LLMResponse(id=query.id, raw_text="", error=str(e)))
        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": query.prompt}],
            "temperature": query.temperature,
            "max_tokens": query.max_tokens,
        }
        response = await self.client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            id=query.id,
            raw_text=raw_text,
            parsed_json=self._parse_json_response(raw_text),
            model_name=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def create_provider(provider_name: str) -> LLMProvider:
    """Factory function to create LLM providers."""
    providers = {
        "openai_gpt": OpenAIProvider,
        "claude_sonnet": AnthropicProvider,
        "grok": GrokProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
    }
    
    provider_class = providers.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    return provider_class()
