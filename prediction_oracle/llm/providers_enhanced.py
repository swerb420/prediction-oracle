"""
Enhanced LLM providers with Grok reasoning and GPT-mini for cheap screening.
"""

import asyncio
import json
import time
import re
from datetime import datetime
from typing import Any
import httpx
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ReasoningStep(BaseModel):
    """A single reasoning step."""
    step_num: int
    thought: str
    confidence_delta: float = 0.0


class EnhancedLLMResponse(BaseModel):
    """Enhanced response with reasoning chain."""
    model: str
    market_id: str
    p_true: float
    confidence: float
    reasoning_steps: list[ReasoningStep]
    final_rationale: str
    tokens_used: int
    latency_ms: int
    rule_risks: list[str] = []
    cached: bool = False


class GrokReasoningProvider:
    """
    Grok provider with extended thinking/reasoning.
    Fast and smart - perfect for prediction markets.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "grok-2-1212",
        use_reasoning: bool = True,
        timeout: float = 60.0
    ):
        self.api_key = api_key
        self.model = model
        self.use_reasoning = use_reasoning
        self.base_url = "https://api.x.ai/v1"
        self.client = httpx.AsyncClient(timeout=timeout)
        self._semaphore = asyncio.Semaphore(3)
        
    async def evaluate_market(
        self,
        market_id: str,
        question: str,
        rules: str,
        current_price: float,
        context: str = "",
        news_context: str = ""
    ) -> EnhancedLLMResponse:
        """Evaluate a market with Grok's reasoning."""
        start_time = time.time()
        
        system_prompt = """You are an expert prediction market analyst. Use structured reasoning.

When analyzing:
1. Identify key outcome factors
2. Consider base rates and precedents
3. Account for information asymmetry vs market
4. Check rule interpretation carefully
5. Adjust for cognitive biases

Output JSON:
{
    "reasoning_steps": [
        {"step": 1, "thought": "...", "confidence_change": 0.0},
        {"step": 2, "thought": "...", "confidence_change": 0.05}
    ],
    "p_true": 0.XX,
    "confidence": 0.XX,
    "final_rationale": "One sentence",
    "rule_risks": ["any ambiguous rules"]
}"""

        user_prompt = f"""QUESTION: {question}

RULES: {rules}

CURRENT PRICE: {current_price:.1%}

{f"NEWS: {news_context}" if news_context else ""}
{f"CONTEXT: {context}" if context else ""}

Provide calibrated probability. Deviate from {current_price:.1%} only with good reason."""

        async with self._semaphore:
            try:
                resp = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1500
                    }
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                if resp.status_code != 200:
                    raise Exception(f"Grok API error: {resp.status_code}")
                
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                
                parsed = self._parse_response(content)
                
                return EnhancedLLMResponse(
                    model=self.model,
                    market_id=market_id,
                    p_true=parsed["p_true"],
                    confidence=parsed["confidence"],
                    reasoning_steps=[
                        ReasoningStep(
                            step_num=s.get("step", i+1),
                            thought=s.get("thought", ""),
                            confidence_delta=s.get("confidence_change", 0)
                        )
                        for i, s in enumerate(parsed.get("reasoning_steps", []))
                    ],
                    final_rationale=parsed.get("final_rationale", ""),
                    rule_risks=parsed.get("rule_risks", []),
                    tokens_used=tokens,
                    latency_ms=latency_ms
                )
                
            except Exception as e:
                logger.error(f"Grok error: {e}")
                return EnhancedLLMResponse(
                    model=self.model,
                    market_id=market_id,
                    p_true=current_price,
                    confidence=0.1,
                    reasoning_steps=[],
                    final_rationale=f"Error: {str(e)}",
                    tokens_used=0,
                    latency_ms=int((time.time() - start_time) * 1000)
                )
    
    def _parse_response(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            p_match = re.search(r'"p_true"\s*:\s*(0\.\d+)', content)
            conf_match = re.search(r'"confidence"\s*:\s*(0\.\d+)', content)
            
            return {
                "p_true": float(p_match.group(1)) if p_match else 0.5,
                "confidence": float(conf_match.group(1)) if conf_match else 0.3,
                "reasoning_steps": [],
                "final_rationale": "Parsed from unstructured"
            }

    async def close(self):
        await self.client.aclose()


class FastScreeningProvider:
    """
    Fast/cheap provider for high-volume screening.
    Uses GPT-4o-mini or similar for quick estimates.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.client = httpx.AsyncClient(timeout=timeout)
        self._semaphore = asyncio.Semaphore(10)
        
    async def quick_evaluate(
        self,
        market_id: str,
        question: str,
        current_price: float
    ) -> tuple[float, float, str]:
        """
        Quick probability estimate.
        Returns (p_true, confidence, rationale)
        """
        prompt = f"""Quick prediction analysis.

Question: {question}
Current price: {current_price:.0%}

Reply ONLY with JSON: {{"p": 0.XX, "c": 0.XX, "r": "reason"}}
p=probability, c=confidence, r=rationale"""

        async with self._semaphore:
            try:
                resp = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 100
                    }
                )
                
                if resp.status_code != 200:
                    return current_price, 0.1, "API error"
                
                content = resp.json()["choices"][0]["message"]["content"]
                
                # Clean and parse
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                data = json.loads(content)
                return float(data["p"]), float(data["c"]), str(data["r"])
                
            except Exception as e:
                logger.debug(f"Quick eval error: {e}")
                return current_price, 0.1, str(e)

    async def batch_screen(
        self,
        markets: list[dict],
        batch_size: int = 5
    ) -> list[tuple[str, float, float, str]]:
        """
        Screen multiple markets quickly.
        Returns list of (market_id, edge, confidence, rationale)
        """
        results = []
        
        for i in range(0, len(markets), batch_size):
            batch = markets[i:i+batch_size]
            tasks = [
                self.quick_evaluate(m["market_id"], m["question"], m["current_price"])
                for m in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for m, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    continue
                p_true, confidence, rationale = result
                edge = p_true - m["current_price"]
                results.append((m["market_id"], edge, confidence, rationale))
            
            # Small delay between batches
            if i + batch_size < len(markets):
                await asyncio.sleep(0.2)
        
        return results

    async def close(self):
        await self.client.aclose()


# Factory functions
def create_grok_provider(api_key: str) -> GrokReasoningProvider:
    return GrokReasoningProvider(api_key)

def create_fast_provider(api_key: str) -> FastScreeningProvider:
    return FastScreeningProvider(api_key)
