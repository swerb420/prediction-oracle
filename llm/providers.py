"""LLM provider interfaces and implementations."""

import json
import logging
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
    max_tokens: int = 2000


class LLMResponse(BaseModel):
    """Response from an LLM."""

    id: str
    raw_text: str
    parsed_json: dict[str, Any] | None = None
    error: str | None = None


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, name: str):
        self.name = name
        self.client = httpx.AsyncClient(timeout=60.0)

    @abstractmethod
    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        """
        Generate responses for a batch of queries.
        
        Args:
            queries: List of queries to process
            
        Returns:
            List of responses in the same order as queries
        """
        pass

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    def _parse_json_response(self, text: str) -> dict[str, Any] | None:
        """Attempt to extract and parse JSON from response text."""
        try:
            # Try direct JSON parse first
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON within markdown code blocks
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end > start:
                    try:
                        return json.loads(text[start:end].strip())
                    except json.JSONDecodeError:
                        pass
            
            # Try to find JSON array or object
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end > start:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        pass
        
        logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
        return None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-5, etc.)."""

    def __init__(self):
        super().__init__("openai_gpt")
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.model = settings.openai_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        """Generate responses using OpenAI API."""
        if not self.api_key:
            logger.warning("OpenAI API key not configured, returning empty responses")
            return [
                LLMResponse(id=q.id, raw_text="", error="API key not configured")
                for q in queries
            ]

        responses = []
        for query in queries:
            try:
                result = await self._call_api(query)
                responses.append(result)
            except Exception as e:
                logger.error(f"OpenAI API error for query {query.id}: {e}")
                responses.append(
                    LLMResponse(id=query.id, raw_text="", error=str(e))
                )
        
        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        """Make a single API call to OpenAI."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": query.prompt}
            ],
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
        
        return LLMResponse(
            id=query.id,
            raw_text=raw_text,
            parsed_json=parsed_json,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude)."""

    def __init__(self):
        super().__init__("claude_sonnet")
        self.api_key = settings.anthropic_api_key
        self.base_url = settings.anthropic_base_url
        self.model = settings.anthropic_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        """Generate responses using Anthropic API."""
        if not self.api_key:
            logger.warning("Anthropic API key not configured, returning empty responses")
            return [
                LLMResponse(id=q.id, raw_text="", error="API key not configured")
                for q in queries
            ]

        responses = []
        for query in queries:
            try:
                result = await self._call_api(query)
                responses.append(result)
            except Exception as e:
                logger.error(f"Anthropic API error for query {query.id}: {e}")
                responses.append(
                    LLMResponse(id=query.id, raw_text="", error=str(e))
                )
        
        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        """Make a single API call to Anthropic."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": query.prompt}
            ],
            "temperature": query.temperature,
            "max_tokens": query.max_tokens,
        }
        
        response = await self.client.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        
        data = response.json()
        raw_text = data["content"][0]["text"]
        parsed_json = self._parse_json_response(raw_text)
        
        return LLMResponse(
            id=query.id,
            raw_text=raw_text,
            parsed_json=parsed_json,
        )


class GrokProvider(LLMProvider):
    """xAI Grok API provider."""

    def __init__(self):
        super().__init__("grok")
        self.api_key = settings.xai_api_key
        self.base_url = settings.xai_base_url
        self.model = settings.xai_model

    async def generate(self, queries: list[LLMQuery]) -> list[LLMResponse]:
        """Generate responses using Grok API."""
        if not self.api_key:
            logger.warning("xAI API key not configured, returning empty responses")
            return [
                LLMResponse(id=q.id, raw_text="", error="API key not configured")
                for q in queries
            ]

        responses = []
        for query in queries:
            try:
                result = await self._call_api(query)
                responses.append(result)
            except Exception as e:
                logger.error(f"Grok API error for query {query.id}: {e}")
                responses.append(
                    LLMResponse(id=query.id, raw_text="", error=str(e))
                )
        
        return responses

    async def _call_api(self, query: LLMQuery) -> LLMResponse:
        """Make a single API call to Grok."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": query.prompt}
            ],
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
        
        return LLMResponse(
            id=query.id,
            raw_text=raw_text,
            parsed_json=parsed_json,
        )


def create_provider(provider_name: str) -> LLMProvider:
    """
    Factory function to create LLM providers.
    
    Args:
        provider_name: Name of the provider (openai_gpt, claude_sonnet, grok)
        
    Returns:
        LLM provider instance
    """
    providers = {
        "openai_gpt": OpenAIProvider,
        "claude_sonnet": AnthropicProvider,
        "grok": GrokProvider,
    }
    
    provider_class = providers.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return provider_class()
