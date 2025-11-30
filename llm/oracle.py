"""LLM Oracle for market probability estimation and aggregation."""

import logging
import statistics
from typing import Any

import numpy as np
from pydantic import BaseModel

from ..markets import Market
from .batcher import LLMBatcher
from .prompts import build_probability_prompt
from .providers import LLMQuery, create_provider

logger = logging.getLogger(__name__)


class OutcomeEvaluation(BaseModel):
    """LLM evaluation for a single outcome."""

    market_id: str
    outcome_id: str
    p_true: float
    confidence: float
    rule_risks: list[str] = []
    notes: str = ""


class OracleResult(BaseModel):
    """Aggregated oracle result for a market outcome."""

    market_id: str
    outcome_id: str
    
    # Aggregated probabilities
    mean_p_true: float
    median_p_true: float
    std_p_true: float
    min_p_true: float
    max_p_true: float
    
    # Market comparison
    implied_p: float  # From market price
    edge: float  # mean_p_true - implied_p
    
    # Consensus metrics
    inter_model_disagreement: float  # Std dev of model estimates
    avg_confidence: float
    
    # Risk assessment
    rule_risks: list[str]
    models_used: list[str]
    
    # Individual evaluations
    evaluations: list[OutcomeEvaluation]


class LLMOracle:
    """
    Multi-model LLM oracle for prediction market analysis.
    
    Coordinates multiple LLM providers to analyze markets and aggregate
    their probability estimates into actionable signals.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize LLM Oracle.
        
        Args:
            config: Configuration dict with provider groups and settings
        """
        self.config = config
        self.batchers: dict[str, LLMBatcher] = {}
        
        # Initialize batchers for each provider
        for group_name, group_config in config.get("llm_groups", {}).items():
            for provider_name in group_config.get("providers", []):
                if provider_name not in self.batchers:
                    provider = create_provider(provider_name)
                    self.batchers[provider_name] = LLMBatcher(
                        provider=provider,
                        batch_size=group_config.get("batch_size", 10),
                        cache_ttl_seconds=group_config.get("cache_ttl_seconds", 600),
                        rate_limit_per_minute=10,
                    )
        
        logger.info(f"LLMOracle initialized with {len(self.batchers)} providers")

    async def evaluate_markets(
        self,
        markets: list[Market],
        model_group: str = "conservative",
    ) -> dict[str, list[OracleResult]]:
        """
        Evaluate markets using a group of LLM providers.
        
        Args:
            markets: Markets to evaluate
            model_group: Which provider group to use (from config)
            
        Returns:
            Dict mapping market_id to list of OracleResults (one per outcome)
        """
        group_config = self.config.get("llm_groups", {}).get(model_group, {})
        provider_names = group_config.get("providers", [])
        
        if not provider_names:
            logger.warning(f"No providers configured for group {model_group}")
            return {}
        
        logger.info(
            f"Evaluating {len(markets)} markets with {len(provider_names)} providers"
        )
        
        # Build prompt
        prompt = build_probability_prompt(markets)
        
        # Query all providers
        all_evaluations: dict[str, list[OutcomeEvaluation]] = {}
        
        for provider_name in provider_names:
            batcher = self.batchers.get(provider_name)
            if not batcher:
                logger.warning(f"Provider {provider_name} not initialized")
                continue
            
            # Create query
            query = LLMQuery(
                id=f"batch_{len(markets)}_markets",
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent probabilities
                max_tokens=3000,
            )
            
            try:
                responses = await batcher.batch_generate([query])
                
                if responses and responses[0].parsed_json:
                    evaluations = self._parse_evaluations(
                        responses[0].parsed_json,
                        provider_name,
                    )
                    
                    for eval in evaluations:
                        key = f"{eval.market_id}:{eval.outcome_id}"
                        if key not in all_evaluations:
                            all_evaluations[key] = []
                        all_evaluations[key].append(eval)
                else:
                    logger.warning(f"No valid JSON from {provider_name}")
                    
            except Exception as e:
                logger.error(f"Error evaluating with {provider_name}: {e}")
        
        # Aggregate results by market
        results: dict[str, list[OracleResult]] = {}
        
        for market in markets:
            market_results = []
            
            for outcome in market.outcomes:
                key = f"{market.market_id}:{outcome.id}"
                evals = all_evaluations.get(key, [])
                
                if evals:
                    oracle_result = self._aggregate_evaluations(
                        market_id=market.market_id,
                        outcome_id=outcome.id,
                        implied_p=outcome.price,
                        evaluations=evals,
                    )
                    market_results.append(oracle_result)
            
            if market_results:
                results[market.market_id] = market_results
        
        logger.info(f"Generated oracle results for {len(results)} markets")
        return results

    def _parse_evaluations(
        self,
        json_data: dict[str, Any] | list[Any],
        provider_name: str,
    ) -> list[OutcomeEvaluation]:
        """Parse JSON response into OutcomeEvaluation objects."""
        evaluations = []
        
        # Handle both dict and list responses
        items = json_data if isinstance(json_data, list) else [json_data]
        
        for item in items:
            try:
                eval = OutcomeEvaluation(
                    market_id=item.get("market_id", ""),
                    outcome_id=item.get("outcome_id", ""),
                    p_true=float(item.get("p_true", 0.5)),
                    confidence=float(item.get("confidence", 0.5)),
                    rule_risks=item.get("rule_risks", []),
                    notes=item.get("notes", ""),
                )
                evaluations.append(eval)
            except Exception as e:
                logger.warning(f"Failed to parse evaluation from {provider_name}: {e}")
        
        return evaluations

    def _aggregate_evaluations(
        self,
        market_id: str,
        outcome_id: str,
        implied_p: float,
        evaluations: list[OutcomeEvaluation],
    ) -> OracleResult:
        """Aggregate multiple model evaluations into a single result."""
        p_trues = [e.p_true for e in evaluations]
        confidences = [e.confidence for e in evaluations]
        all_risks = []
        for e in evaluations:
            all_risks.extend(e.rule_risks)
        
        # Calculate statistics
        mean_p = float(np.mean(p_trues))
        median_p = float(np.median(p_trues))
        std_p = float(np.std(p_trues)) if len(p_trues) > 1 else 0.0
        
        return OracleResult(
            market_id=market_id,
            outcome_id=outcome_id,
            mean_p_true=mean_p,
            median_p_true=median_p,
            std_p_true=std_p,
            min_p_true=min(p_trues),
            max_p_true=max(p_trues),
            implied_p=implied_p,
            edge=mean_p - implied_p,
            inter_model_disagreement=std_p,
            avg_confidence=float(np.mean(confidences)),
            rule_risks=list(set(all_risks)),  # Deduplicate
            models_used=[f"model_{i}" for i in range(len(evaluations))],
            evaluations=evaluations,
        )

    async def close(self) -> None:
        """Close all provider connections."""
        for batcher in self.batchers.values():
            await batcher.provider.close()
