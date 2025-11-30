"""
Enhanced Longshot Strategy with news velocity filtering.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from ..config import settings
from ..llm.enhanced_oracle import EnhancedOracle
from ..markets import Market
from ..risk import BankrollManager
from .base_strategy import BaseStrategy, TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class LongshotSignal:
    """Combined signal for longshot opportunities."""
    news_velocity: float = 0.0  # Recent news activity
    smart_money_divergence: float = 0.0  # Smart money vs market
    social_momentum: float = 0.0  # Social trending
    llm_mispricing: float = 0.0  # LLM estimated edge
    
    @property
    def opportunity_score(self) -> float:
        """Score longshot opportunity."""
        return (
            self.news_velocity * 0.4 +  # Breaking news = biggest edge
            self.smart_money_divergence * 0.3 +
            self.social_momentum * 0.1 +
            self.llm_mispricing * 0.2
        )


class EnhancedLongshotStrategy(BaseStrategy):
    """
    Longshot strategy with news velocity filtering.
    Looks for underpriced outcomes with breaking news catalyst.
    """
    
    def __init__(
        self,
        config: dict,
        bankroll_manager: BankrollManager,
        oracle: Optional[EnhancedOracle] = None,
    ):
        """Initialize strategy."""
        strategy_config = config.get("strategies", config).get("longshot", {})
        super().__init__("longshot", strategy_config)

        self.full_config = config
        self.bankroll = bankroll_manager
        self.oracle = oracle or EnhancedOracle(config)

        # Strategy settings
        self.max_price = strategy_config.get("max_price", strategy_config.get("price_range", [0.0, 0.15])[1])
        self.min_upside = strategy_config.get("min_upside_multiplier", 3.0)
        self.bet_size = strategy_config.get("bet_size", strategy_config.get("fixed_bet_usd", 5.0))
        self.max_bets_per_day = strategy_config.get(
            "max_bets_per_day", strategy_config.get("max_daily_longshot_bets", 3)
        )

        # Enhanced settings
        self.min_news_velocity = 0.3  # Need breaking news
        self.min_opportunity_score = 0.25

        self.bets_today = 0

        logger.info(
            f"EnhancedLongshot initialized - "
            f"max_price={self.max_price}, "
            f"min_upside={self.min_upside}x, "
            f"min_news_velocity={self.min_news_velocity}"
        )
    
    async def select_markets(self, all_markets: list[Market]) -> list[Market]:
        """Filter longshot candidates before enhanced evaluation."""
        selected: list[Market] = []

        for market in all_markets:
            for outcome in market.outcomes:
                if outcome.price > self.max_price:
                    continue

                selected.append(market)
                break

        # Quick filter to focus on most promising catalysts
        if settings.enable_quick_filter and len(selected) > 20:
            filtered_ids = await self.oracle.quick_filter_markets(selected, top_n=20)
            selected = [m for m in selected if m.market_id in filtered_ids]
            logger.info(f"EnhancedLongshot quick filtered to {len(selected)} markets")

        return selected

    async def evaluate_markets(
        self,
        markets: list[Market],
        oracle_results: Optional[dict[str, list]] = None,
    ) -> list[dict]:
        """Evaluate markets for longshot opportunities."""
        if not markets:
            return []

        if self.bets_today >= self.max_bets_per_day:
            logger.info(f"Daily limit reached ({self.bets_today}/{self.max_bets_per_day})")
            return []

        logger.info(f"Evaluating {len(markets)} markets for longshots")

        # Get enhanced oracle results
        if oracle_results is None:
            oracle_results = await self.oracle.evaluate_markets_enhanced(
                markets, model_group=self.name
            )
        
        # Gather signals (focus on news velocity)
        news_signals = {}
        smart_money = {}
        social_buzz = {}
        
        if settings.enable_news_signals:
            try:
                from ..signals.news_signals import news_provider
                news_provider.set_api_keys(
                    newsapi_key=settings.newsapi_key,
                    gnews_key=settings.gnews_key
                )
                
                for market in markets:
                    query = self._extract_search_query(market.question)
                    try:
                        # Focus on RECENT news (6 hours)
                        signal = await news_provider.get_news_signal(query, hours_back=6)
                        if signal.article_count > 0:  # Only if there's recent news
                            news_signals[market.market_id] = signal
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"News signals unavailable: {e}")
        
        if settings.enable_smart_money_signals:
            try:
                from ..signals.polymarket_signals import polymarket_signals
                from ..markets import Venue
                
                for market in markets:
                    if market.venue == Venue.POLYMARKET:
                        try:
                            signal = await polymarket_signals.get_smart_money_signal(market)
                            smart_money[market.market_id] = signal
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Smart money unavailable: {e}")
        
        if settings.enable_social_signals:
            try:
                from ..signals.social_signals import social_signals as social_provider
                
                for market in markets:
                    query = self._extract_search_query(market.question)
                    try:
                        signal = await social_provider.get_social_buzz(query)
                        social_buzz[market.market_id] = signal
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Social signals unavailable: {e}")
        
        # Build recommendations
        recommendations = []
        
        for market in markets:
            if market.market_id not in oracle_results:
                continue
            
            for result in oracle_results[market.market_id]:
                # Basic filters
                if not self._passes_basic_filters(market, result):
                    continue
                
                # Calculate longshot signal
                signal = self._calculate_longshot_signal(
                    market.market_id,
                    result,
                    news_signals,
                    smart_money,
                    social_buzz,
                )
                
                # NEWS VELOCITY REQUIRED for longshots
                if signal.news_velocity < self.min_news_velocity:
                    logger.debug(
                        f"Skipping {market.question[:50]} - "
                        f"insufficient news velocity: {signal.news_velocity:.3f}"
                    )
                    continue
                
                # Check opportunity score
                if signal.opportunity_score < self.min_opportunity_score:
                    logger.debug(
                        f"Skipping {market.question[:50]} - "
                        f"low opportunity score: {signal.opportunity_score:.3f}"
                    )
                    continue
                
                # Find outcome
                outcome = next((o for o in market.outcomes if o.id == result.outcome_id), None)
                if not outcome:
                    continue
                
                # Check upside
                if outcome.price > 0:
                    upside = result.mean_p_true / outcome.price
                    if upside < self.min_upside:
                        continue
                else:
                    upside = float('inf')
                
                recommendations.append({
                    "market": market,
                    "outcome": outcome,
                    "oracle_result": result,
                    "bet_size": self.bet_size,
                    "signal": signal,
                    "upside": upside,
                    "rationale": self._build_rationale(market, result, signal, upside),
                })
        
        # Sort by opportunity score
        recommendations.sort(key=lambda x: x["signal"].opportunity_score, reverse=True)
        
        # Limit to remaining daily budget
        remaining = self.max_bets_per_day - self.bets_today
        recommendations = recommendations[:remaining]
        
        logger.info(
            f"Found {len(recommendations)} longshot opportunities "
            f"from {len(markets)} markets"
        )

        return recommendations

    async def evaluate(
        self, markets: list[Market], oracle_results: dict
    ) -> list[TradeDecision]:
        """Adapter to scheduler interface returning TradeDecision objects."""
        recommendations = await self.evaluate_markets(markets, oracle_results)

        decisions: list[TradeDecision] = []
        for rec in recommendations:
            result = rec["oracle_result"]
            direction = "BUY" if result.edge >= 0 else "SELL"

            decisions.append(
                TradeDecision(
                    venue=rec["market"].venue,
                    market_id=rec["market"].market_id,
                    outcome_id=result.outcome_id,
                    direction=direction,
                    size_usd=rec["bet_size"],
                    p_true=result.mean_p_true,
                    implied_p=result.implied_p,
                    edge=result.edge,
                    confidence=result.avg_confidence,
                    inter_model_disagreement=result.inter_model_disagreement,
                    rule_risks=result.rule_risks,
                    strategy_name=self.name,
                    rationale=rec.get("rationale", ""),
                    models_used=result.models_used,
                )
            )

        return decisions
    
    def _passes_basic_filters(self, market: Market, result) -> bool:
        """Check basic quality filters."""
        # Price limit
        if result.implied_p > self.max_price:
            return False
        
        # Must be underpriced by LLM
        if result.edge <= 0:
            return False
        
        # Minimum liquidity ($100) - use market volume as proxy
        if market.volume_24h and market.volume_24h < 100:
            return False
        
        return True
    
    def _calculate_longshot_signal(
        self,
        market_id: str,
        oracle_result,
        news_signals: dict,
        smart_money: dict,
        social_buzz: dict,
    ) -> LongshotSignal:
        """Calculate longshot opportunity signal."""
        signal = LongshotSignal()
        
        # LLM mispricing
        signal.llm_mispricing = max(0, oracle_result.edge)
        
        # NEWS VELOCITY (critical for longshots)
        if market_id in news_signals:
            news = news_signals[market_id]
            # High velocity = breaking story = opportunity
            signal.news_velocity = news.velocity * news.confidence
        
        # Smart money divergence
        if market_id in smart_money:
            sm = smart_money[market_id]
            # Smart money buying = good signal
            if sm.smart_money_score > 0:
                signal.smart_money_divergence = sm.smart_money_score * sm.confidence
        
        # Social momentum
        if market_id in social_buzz:
            social = social_buzz[market_id]
            # Trending topics can move markets
            signal.social_momentum = social.trending_score
        
        return signal
    
    def _extract_search_query(self, question: str) -> str:
        """Extract key search terms from market question."""
        remove_phrases = [
            "will", "be", "the", "a", "an", "to", "in", "on", "at",
            "by", "for", "of", "?", "before", "after", "during",
        ]
        
        words = question.lower().split()
        keywords = [w for w in words if w not in remove_phrases and len(w) > 2]
        
        return " ".join(keywords[:5])
    
    def _build_rationale(
        self,
        market: Market,
        oracle_result,
        signal: LongshotSignal,
        upside: float,
    ) -> str:
        """Build human-readable rationale."""
        parts = [
            f"Price: {oracle_result.implied_p:.1%}",
            f"LLM thinks: {oracle_result.mean_p_true:.1%}",
            f"Upside: {upside:.1f}x",
        ]
        
        if signal.news_velocity > 0.1:
            parts.append(f"NEWS VELOCITY: {signal.news_velocity:.2f}")
        
        if signal.smart_money_divergence > 0.1:
            parts.append(f"Smart money buying")
        
        if signal.social_momentum > 0.5:
            parts.append(f"Trending on social")
        
        parts.append(f"Score: {signal.opportunity_score:.3f}")
        
        return " | ".join(parts)
    
    def mark_bet_placed(self):
        """Track bet placement."""
        self.bets_today += 1
    
    async def close(self):
        """Cleanup."""
        await self.oracle.close()
