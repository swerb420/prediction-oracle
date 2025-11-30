"""
Enhanced Conservative Strategy with signal confluence scoring.
"""
import logging
from dataclasses import dataclass

from ..config import settings
from ..llm.enhanced_oracle import EnhancedOracle
from ..markets import Market
from ..risk import BankrollManager
from .base_strategy import EnhancedStrategy, TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class SignalConfluence:
    """Confluence score from multiple signals."""
    news_signal: float = 0.0  # -1 to 1
    smart_money: float = 0.0  # -1 to 1
    social_buzz: float = 0.0  # -1 to 1
    llm_edge: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Weighted confluence score."""
        return (
            self.news_signal * 0.2 +
            self.smart_money * 0.3 +
            self.social_buzz * 0.1 +
            self.llm_edge * 0.4
        )
    
    @property
    def confidence(self) -> float:
        """How aligned are the signals?"""
        signals = [self.news_signal, self.smart_money, self.social_buzz]
        active_signals = [s for s in signals if abs(s) > 0.01]
        
        if len(active_signals) < 2:
            return 0.5
        
        # All pointing same direction = high confidence
        same_sign = all(s > 0 for s in active_signals) or all(s < 0 for s in active_signals)
        if same_sign:
            return 0.8 + (len(active_signals) * 0.05)
        
        return 0.3


class EnhancedConservativeStrategy(EnhancedStrategy):
    """
    Conservative strategy with signal confluence.
    Only bets when multiple signals agree + LLM shows edge.
    """
    
    def __init__(self, config: dict, bankroll_manager: BankrollManager):
        """Initialize strategy."""
        super().__init__("conservative", config, bankroll_manager)
        self.oracle = EnhancedOracle(config)
        
        # Strategy settings
        self.min_edge = config.get("conservative", {}).get("min_edge", 0.04)
        self.min_prob = config.get("conservative", {}).get("min_prob_range", [0.3, 0.8])[0]
        self.max_prob = config.get("conservative", {}).get("min_prob_range", [0.3, 0.8])[1]
        self.min_liquidity = config.get("conservative", {}).get("min_liquidity", 500)
        self.max_spread = config.get("conservative", {}).get("max_spread", 0.05)
        
        # Enhanced settings
        self.min_confluence_score = 0.15  # Must have positive confluence
        self.min_confluence_confidence = 0.6
        
        logger.info(
            f"EnhancedConservative initialized - "
            f"min_edge={self.min_edge}, "
            f"min_confluence={self.min_confluence_score}"
        )
    
    async def evaluate_markets(self, markets: list[Market]) -> list[dict]:
        """Evaluate markets and return bet recommendations."""
        if not markets:
            return []
        
        logger.info(f"Evaluating {len(markets)} markets with enhanced strategy")
        
        # Quick filter if enabled
        if settings.enable_quick_filter and len(markets) > 20:
            filtered_ids = await self.oracle.quick_filter_markets(markets, top_n=20)
            markets = [m for m in markets if m.market_id in filtered_ids]
            logger.info(f"Quick filter reduced to {len(markets)} markets")
        
        # Get enhanced oracle results
        oracle_results = await self.oracle.evaluate_markets_enhanced(markets, model_group="conservative")
        
        # Gather signals for confluence
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
                        signal = await news_provider.get_news_signal(query, hours_back=48)
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
                
                # Calculate confluence
                confluence = self._calculate_confluence(
                    market.market_id,
                    result,
                    news_signals,
                    smart_money,
                    social_buzz,
                )
                
                # Check confluence requirements
                if confluence.total_score < self.min_confluence_score:
                    logger.debug(
                        f"Skipping {market.question[:50]} - "
                        f"low confluence score: {confluence.total_score:.3f}"
                    )
                    continue
                
                if confluence.confidence < self.min_confluence_confidence:
                    logger.debug(
                        f"Skipping {market.question[:50]} - "
                        f"low confluence confidence: {confluence.confidence:.3f}"
                    )
                    continue
                
                # Calculate position size (Kelly with confluence boost)
                outcome = next((o for o in market.outcomes if o.id == result.outcome_id), None)
                if not outcome:
                    continue
                
                kelly_fraction = self._calculate_kelly(
                    result.mean_p_true,
                    outcome.price,
                    confluence_boost=confluence.confidence,
                )
                
                if kelly_fraction <= 0:
                    continue
                
                current_bankroll = self.bankroll.get_available_bankroll()
                bet_size = current_bankroll * kelly_fraction
                
                # Position limits
                max_position = self.config.get("conservative", {}).get("max_position_pct", 0.05)
                bet_size = min(bet_size, current_bankroll * max_position)
                
                if bet_size < 1.0:
                    continue
                
                recommendations.append({
                    "market": market,
                    "outcome": outcome,
                    "oracle_result": result,
                    "bet_size": bet_size,
                    "kelly_fraction": kelly_fraction,
                    "confluence": confluence,
                    "rationale": self._build_rationale(market, result, confluence),
                })
        
        logger.info(
            f"Found {len(recommendations)} opportunities "
            f"from {len(markets)} markets"
        )
        
        return recommendations

    def _recommendation_to_decision(self, recommendation: dict) -> TradeDecision | None:
        """Convert enhanced recommendation to a standardized trade decision."""
        oracle_result = recommendation.get("oracle_result")
        market = recommendation.get("market")
        outcome = recommendation.get("outcome")
        rationale = recommendation.get("rationale", "")
        bet_size = recommendation.get("bet_size")

        if not (oracle_result and market and outcome and bet_size):
            return None

        direction = "BUY" if oracle_result.edge >= 0 else "SELL"

        return TradeDecision(
            venue=market.venue,
            market_id=market.market_id,
            outcome_id=outcome.id,
            direction=direction,
            size_usd=float(bet_size),
            p_true=oracle_result.mean_p_true,
            implied_p=oracle_result.implied_p,
            edge=oracle_result.edge,
            confidence=oracle_result.avg_confidence,
            inter_model_disagreement=oracle_result.inter_model_disagreement,
            rule_risks=oracle_result.rule_risks,
            strategy_name=self.name,
            rationale=rationale,
            models_used=oracle_result.models_used,
        )
    
    def _passes_basic_filters(self, market: Market, result) -> bool:
        """Check basic quality filters."""
        # Edge requirement
        if result.edge < self.min_edge:
            return False
        
        # Probability range
        if result.mean_p_true < self.min_prob or result.mean_p_true > self.max_prob:
            return False
        
        # Liquidity (use market volume as proxy)
        if market.volume_24h and market.volume_24h < self.min_liquidity:
            return False
        
        # Spread
        outcome = next((o for o in market.outcomes if o.id == result.outcome_id), None)
        if outcome:
            # Calculate spread from outcome data if available
            if hasattr(outcome, 'spread'):
                if outcome.spread > self.max_spread:
                    return False
        
        return True
    
    def _calculate_confluence(
        self,
        market_id: str,
        oracle_result,
        news_signals: dict,
        smart_money: dict,
        social_buzz: dict,
    ) -> SignalConfluence:
        """Calculate signal confluence."""
        confluence = SignalConfluence()
        
        # LLM edge (baseline)
        confluence.llm_edge = oracle_result.edge
        
        # News signal
        if market_id in news_signals:
            news = news_signals[market_id]
            # Positive if news sentiment aligns with bet direction
            confluence.news_signal = news.sentiment_score * news.confidence
        
        # Smart money
        if market_id in smart_money:
            sm = smart_money[market_id]
            confluence.smart_money = sm.smart_money_score * sm.confidence
        
        # Social buzz
        if market_id in social_buzz:
            social = social_buzz[market_id]
            # High buzz + positive sentiment = strong signal
            if social.trending_score > 0.5:
                confluence.social_buzz = social.sentiment_score * 0.5
        
        return confluence
    
    def _calculate_kelly(
        self,
        p_true: float,
        price: float,
        confluence_boost: float = 0.0,
    ) -> float:
        """
        Calculate Kelly fraction with confluence boost.
        """
        if price >= 1.0 or price <= 0.0:
            return 0.0
        
        # Kelly formula: (p * (1-price) - (1-p) * price) / (1-price)
        numerator = p_true * (1 - price) - (1 - p_true) * price
        denominator = 1 - price
        
        if denominator == 0 or numerator <= 0:
            return 0.0
        
        kelly = numerator / denominator
        
        # Fractional Kelly with confluence boost
        base_fraction = 0.25  # Conservative default
        boosted_fraction = min(0.4, base_fraction * (1 + confluence_boost))
        
        return kelly * boosted_fraction
    
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
        confluence: SignalConfluence,
    ) -> str:
        """Build human-readable rationale."""
        parts = [
            f"Edge: {oracle_result.edge:.1%}",
            f"LLM P={oracle_result.mean_p_true:.2%} vs Market={oracle_result.implied_p:.2%}",
        ]
        
        if abs(confluence.news_signal) > 0.05:
            parts.append(f"News: {'positive' if confluence.news_signal > 0 else 'negative'}")
        
        if abs(confluence.smart_money) > 0.05:
            parts.append(f"SmartMoney: {'bullish' if confluence.smart_money > 0 else 'bearish'}")
        
        if abs(confluence.social_buzz) > 0.05:
            parts.append(f"Social: {'trending' if confluence.social_buzz > 0 else 'quiet'}")
        
        parts.append(f"Confluence: {confluence.total_score:.3f} (conf={confluence.confidence:.2f})")
        
        return " | ".join(parts)
    
    async def close(self):
        """Cleanup."""
        await self.oracle.close()
