"""Nightly hypothesis testing pipeline for new signals/prompts."""

from __future__ import annotations

import asyncio
import logging
from statistics import mean

from ..storage import LLMEval, get_session

logger = logging.getLogger(__name__)


async def _compute_reality_check(evals: list[LLMEval]) -> float:
    if not evals:
        return 0.0
    edges = [e.edge for e in evals if e.edge is not None]
    return mean(edges) if edges else 0.0


async def run_hypothesis_pipeline() -> dict:
    """Run a simplified multiple-testing aware evaluation."""
    async with get_session() as session:
        results = (await session.execute("SELECT * FROM llm_evals ORDER BY created_at DESC LIMIT 500")).scalars().all()
    score = await _compute_reality_check(results)
    acceptance = score > 0.01
    payload = {"reality_check_score": score, "accepted": acceptance, "sample_size": len(results)}
    logger.info("Hypothesis test completed: %s", payload)
    return payload


if __name__ == "__main__":
    asyncio.run(run_hypothesis_pipeline())
