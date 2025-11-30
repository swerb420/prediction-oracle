"""
Enhanced Oracle that integrates all signal sources.
"""
import asyncio
import json
import logging
from datetime import datetime
from hashlib import sha256

import numpy as np

from .oracle import OracleResult
from .enhanced_prompts import build_enhanced_probability_prompt, build_quick_filter_prompt
from .providers import LLMQuery, create_provider
from ..config import settings

logger = logging.getLogger(__name__)


class EnhancedOracle:
    """
    Oracle with integrated signal sources.
    Much smarter than base LLM-only approach.
    """
    
    def __init__(self, config: dict):
        """Initialize with config."""
        self.config = config
        self.providers = {}
        self.cache: dict[str, tuple[float, dict[str, list]] | None] = {}
        self.cache_ttl = config.get("llm_cache_ttl", 300)
        self.daily_spend = 0.0
        
        # Initialize providers
        for group_name, group_config in config.get("llm_groups", {}).items():
            for provider_name in group_config.get("providers", []):
                if provider_name not in self.providers:
                    try:
                        self.providers[provider_name] = create_provider(provider_name)
                    except Exception as e:
                        logger.warning(f"Failed to create provider {provider_name}: {e}")
        
        logger.info(f"EnhancedOracle initialized with {len(self.providers)} providers")
    
    async def evaluate_markets_enhanced(
        self,
        markets: list,
        model_group: str = "conservative",
    ) -> dict[str, list[OracleResult]]:
        """
        Enhanced evaluation with all signal sources.
        """
        if not markets:
            return {}
        
        logger.info(f"Enhanced evaluation of {len(markets)} markets")
        
        # Gather all signals in parallel
        news_signals = {}
        smart_money = {}
        social_buzz = {}
        
        if settings.enable_news_signals:
            try:
                from ..signals.news_signals import news_provider
                # Set API keys
                news_provider.set_api_keys(
                    newsapi_key=settings.newsapi_key,
                    gnews_key=settings.gnews_key
                )
                
                for market in markets:
                    query = self._extract_search_query(market.question)
                    try:
                        signal = await news_provider.get_news_signal(query, hours_back=48)
                        news_signals[market.market_id] = signal
                    except Exception as e:
                        logger.debug(f"News fetch failed for {market.market_id}: {e}")
            except Exception as e:
                logger.warning(f"News signals disabled: {e}")
        
        if settings.enable_smart_money_signals:
            try:
                from ..signals.polymarket_signals import polymarket_signals
                from ..markets import Venue
                
                for market in markets:
                    if market.venue == Venue.POLYMARKET:
                        try:
                            signal = await polymarket_signals.get_smart_money_signal(market)
                            smart_money[market.market_id] = signal
                        except Exception as e:
                            logger.debug(f"Smart money fetch failed for {market.market_id}: {e}")
            except Exception as e:
                logger.warning(f"Smart money signals disabled: {e}")
        
        if settings.enable_social_signals:
            try:
                from ..signals.social_signals import social_signals as social_provider
                
                for market in markets:
                    query = self._extract_search_query(market.question)
                    try:
                        signal = await social_provider.get_social_buzz(query)
                        social_buzz[market.market_id] = signal
                    except Exception as e:
                        logger.debug(f"Social fetch failed for {market.market_id}: {e}")
            except Exception as e:
                logger.warning(f"Social signals disabled: {e}")
        
        logger.info(
            f"Signals gathered - News: {len(news_signals)}, "
            f"SmartMoney: {len(smart_money)}, Social: {len(social_buzz)}"
        )
        
        # Build enhanced prompt
        prompt = build_enhanced_probability_prompt(
            markets,
            news_signals=news_signals if news_signals else None,
            smart_money=smart_money if smart_money else None,
            social_buzz=social_buzz if social_buzz else None,
        )

        prompt_key = sha256(prompt.encode()).hexdigest()
        cached = self.cache.get(prompt_key)
        now_ts = datetime.utcnow().timestamp()
        if cached and now_ts - cached[0] < self.cache_ttl:
            logger.info("Serving enhanced oracle results from cache")
            return cached[1]

        # Get providers for this group
        group_config = self.config.get("llm_groups", {}).get(model_group, {})
        provider_names = group_config.get("providers", ["openai_gpt"])
        
        all_results: dict[str, list] = {}

        # Query each provider
        for provider_name in provider_names:
            if self.daily_spend >= settings.llm_daily_budget:
                logger.warning("Daily LLM budget reached; skipping remaining providers")
                break
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            try:
                query = LLMQuery(
                    id=f"enhanced_{model_group}_{datetime.utcnow().timestamp()}",
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=3000,
                )

                responses = await provider.generate([query])
                
                if responses and responses[0].parsed_json:
                    parsed = self._parse_response(responses[0].parsed_json)
                    for item in parsed:
                        item["provider"] = provider_name
                        market_id = item.get("market_id")
                        if market_id not in all_results:
                            all_results[market_id] = []
                        all_results[market_id].append(item)

                # Track spend heuristically
                self.daily_spend += settings.max_cost_per_query

            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
        
        # Build oracle results
        final_results = {}
        
        for market in markets:
            market_results = all_results.get(market.market_id, [])
            if not market_results:
                continue
            
            # Convert to OracleResult format
            for outcome in market.outcomes:
                outcome_results = [
                    r for r in market_results 
                    if r.get("outcome_id") == outcome.id
                ]
                
                if outcome_results:
                    aggregated = self._aggregate_results(outcome_results, market, outcome)
                    
                    # BOOST: Adjust based on smart money signals
                    if market.market_id in smart_money:
                        sm = smart_money[market.market_id]
                        if abs(sm.smart_money_score) > settings.smart_money_min_signal:
                            shift = sm.smart_money_score * 0.1 * sm.confidence
                            aggregated.mean_p_true = max(0.01, min(0.99, aggregated.mean_p_true + shift))
                    
                    if market.market_id not in final_results:
                        final_results[market.market_id] = []
                    final_results[market.market_id].append(aggregated)

        self.cache[prompt_key] = (now_ts, final_results)

        return final_results
    
    async def quick_filter_markets(self, markets: list, top_n: int = 10) -> list[str]:
        """
        Use cheap LLM call to pre-filter markets.
        Returns list of market_ids worth deep analysis.
        """
        if len(markets) <= top_n:
            return [m.market_id for m in markets]
        
        prompt = build_quick_filter_prompt(markets)
        
        # Use first available provider
        provider = list(self.providers.values())[0] if self.providers else None
        if not provider:
            return [m.market_id for m in markets[:top_n]]
        
        try:
            query = LLMQuery(id="quick_filter", prompt=prompt, temperature=0.5, max_tokens=1000)
            responses = await provider.generate([query])
            
            if responses and responses[0].parsed_json:
                data = responses[0].parsed_json
                if isinstance(data, list):
                    return [item.get("market_id") for item in data[:top_n] if item.get("market_id")]
        except Exception as e:
            logger.warning(f"Quick filter failed: {e}")
        
        # Fallback
        return [m.market_id for m in markets[:top_n]]
    
    def _extract_search_query(self, question: str) -> str:
        """Extract key search terms from market question."""
        remove_phrases = [
            "will", "be", "the", "a", "an", "to", "in", "on", "at",
            "by", "for", "of", "?", "before", "after", "during",
        ]
        
        words = question.lower().split()
        keywords = [w for w in words if w not in remove_phrases and len(w) > 2]
        
        return " ".join(keywords[:5])
    
    def _parse_response(self, data) -> list[dict]:
        """Parse LLM response into list of results."""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    
    def _aggregate_results(self, results: list[dict], market, outcome) -> OracleResult:
        """Aggregate results from multiple models."""
        weights = []
        p_values = []
        confidences = []
        for r in results:
            p_values.append(r.get("p_true", 0.5))
            confidences.append(r.get("confidence", 0.5))

            provider = r.get("provider", "")
            # Penalize providers that disagree sharply with consensus
            disagreement_penalty = abs(r.get("p_true", 0.5) - outcome.price)
            weight = max(0.1, 1.0 - disagreement_penalty)
            # Lightly boost confidence when provider name is present
            if provider:
                weight *= 1.05
            weights.append(weight)

        mean_p = float(np.average(p_values, weights=weights)) if p_values else outcome.price
        std_p = float(np.std(p_values)) if len(p_values) > 1 else 0.0

        all_risks = []
        for r in results:
            all_risks.extend(r.get("rule_risks", []))
        
        notes = "; ".join(r.get("rationale", "") for r in results if r.get("rationale"))
        
        return OracleResult(
            market_id=market.market_id,
            outcome_id=outcome.id,
            mean_p_true=mean_p,
            median_p_true=float(np.median(p_values)),
            std_p_true=std_p,
            min_p_true=min(p_values),
            max_p_true=max(p_values),
            implied_p=outcome.price,
            edge=mean_p - outcome.price,
            inter_model_disagreement=std_p,
            avg_confidence=float(np.mean(confidences)),
            rule_risks=list(set(all_risks)),
            models_used=[f"model_{i}" for i in range(len(results))],
            evaluations=[],
        )
    
    async def close(self):
        """Close all providers."""
        for provider in self.providers.values():
            await provider.close()
