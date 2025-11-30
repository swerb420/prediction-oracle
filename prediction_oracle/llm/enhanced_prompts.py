"""
Enhanced prompt templates that incorporate external signals.
This is where the magic happens - feeding the LLM rich context.
"""
from datetime import datetime


def build_enhanced_probability_prompt(
    markets: list,
    news_signals: dict | None = None,
    smart_money: dict | None = None,
    social_buzz: dict | None = None,
) -> str:
    """
    Build a rich prompt with all available signals.
    This dramatically improves prediction accuracy.
    """
    
    prompt = f"""You are an elite prediction market analyst. Today is {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}.

TASK: Analyze each market and provide calibrated probability estimates.

For each market, you have:
1. Market details (question, rules, current prices)
2. NEWS SIGNALS - Recent news sentiment and velocity
3. SMART MONEY - Order book imbalance, whale activity, momentum
4. SOCIAL BUZZ - Reddit/social media sentiment

Use ALL signals to inform your probability. Be especially attentive to:
- Smart money divergence from market price (big edge opportunity)
- News velocity spikes (breaking developments)
- Whale accumulation patterns

MARKETS TO ANALYZE:
"""
    
    for i, market in enumerate(markets, 1):
        prompt += f"\n{'='*60}\n"
        prompt += f"MARKET {i}: {market.market_id}\n"
        prompt += f"{'='*60}\n"
        prompt += f"QUESTION: {market.question}\n"
        prompt += f"RESOLUTION RULES: {market.rules[:500]}...\n" if len(market.rules) > 500 else f"RESOLUTION RULES: {market.rules}\n"
        prompt += f"CLOSES: {market.close_time.isoformat()}\n"
        prompt += f"CATEGORY: {market.category}\n\n"
        
        prompt += "OUTCOMES & PRICES:\n"
        for outcome in market.outcomes:
            implied_prob = outcome.price * 100
            prompt += f"  - {outcome.label}: ${outcome.price:.3f} (implied {implied_prob:.1f}%)\n"
            if outcome.volume_24h:
                prompt += f"    24h Volume: ${outcome.volume_24h:,.0f}\n"
        
        # Add news signals
        if news_signals and market.market_id in news_signals:
            news = news_signals[market.market_id]
            prompt += f"\n📰 NEWS SIGNALS:\n"
            prompt += f"  - Articles (24h): {len(news.articles)}\n"
            prompt += f"  - News Sentiment: {news.avg_sentiment:+.2f} (-1 bearish to +1 bullish)\n"
            prompt += f"  - News Velocity: {news.news_velocity:.1f} articles/hour\n"
            prompt += f"  - Bullish Ratio: {news.bullish_ratio:.0%}\n"
            if news.articles:
                prompt += f"  - Top Headlines:\n"
                for article in news.articles[:3]:
                    prompt += f"    • {article.title[:100]}\n"
        
        # Add smart money signals
        if smart_money and market.market_id in smart_money:
            sm = smart_money[market.market_id]
            prompt += f"\n💰 SMART MONEY SIGNALS:\n"
            prompt += f"  - Order Book Imbalance: {sm.order_book.bid_ask_imbalance:+.2f} (+ = more buyers)\n"
            prompt += f"  - Spread: {sm.order_book.spread_bps:.0f} bps\n"
            prompt += f"  - Whale Bias: {sm.whales.whale_bias:+.2f} (+ = whales buying)\n"
            prompt += f"  - Large Buys: {sm.whales.large_buys_24h} | Large Sells: {sm.whales.large_sells_24h}\n"
            prompt += f"  - SIGNAL: {sm.signal_strength.upper()} (score: {sm.smart_money_score:+.2f})\n"
        
        # Add social buzz
        if social_buzz and market.market_id in social_buzz:
            buzz = social_buzz[market.market_id]
            prompt += f"\n📱 SOCIAL SIGNALS:\n"
            prompt += f"  - Reddit Posts (24h): {buzz.reddit_posts_24h}\n"
            prompt += f"  - Reddit Comments: {buzz.reddit_comments_24h}\n"
            prompt += f"  - Social Sentiment: {buzz.reddit_sentiment:+.2f}\n"
            prompt += f"  - Trending Score: {buzz.reddit_trending_score:.2f}\n"
    
    prompt += f"""
{'='*60}
OUTPUT FORMAT (strict JSON):
{'='*60}

Respond with a JSON array. For each market outcome you analyze:

[
  {{
    "market_id": "market-id-here",
    "outcome_id": "outcome-id",
    "outcome_label": "Yes/No/etc",
    "p_true": 0.XX,
    "confidence": 0.XX,
    "edge_vs_market": 0.XX,
    "signal_alignment": "aligned|divergent|mixed",
    "key_factors": ["factor1", "factor2"],
    "rule_risks": ["any resolution ambiguities"],
    "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "rationale": "2-3 sentence explanation"
  }}
]

CRITICAL INSTRUCTIONS:
1. p_true must be your TRUE probability estimate (0.0-1.0), not just following market
2. If smart money diverges from market price, EXPLAIN why and factor it in
3. High news velocity often precedes price moves - weight recent news heavily
4. Whale accumulation is a strong signal - they often have information edge
5. Flag ANY ambiguous resolution criteria in rule_risks
6. Be calibrated: a 70% prediction should resolve YES about 70% of the time

OUTPUT ONLY THE JSON ARRAY, no other text.
"""
    
    return prompt


def build_quick_filter_prompt(markets: list) -> str:
    """
    Quick pre-filter prompt to identify interesting markets.
    Cheaper than full analysis - use to filter before deep analysis.
    """
    
    market_summaries = []
    for m in markets:
        prices = [f"{o.label}:{o.price:.2f}" for o in m.outcomes]
        market_summaries.append(f"{m.market_id}|{m.question[:80]}|{','.join(prices)}")
    
    return f"""Quickly scan these {len(markets)} prediction markets and identify the TOP 10 with highest edge potential.

MARKETS (format: id|question|outcome:price):
{chr(10).join(market_summaries)}

Criteria for high edge:
- Obvious mispricing based on current events
- Binary markets with extreme prices (< 0.10 or > 0.90) that seem wrong
- Markets where you have strong knowledge suggesting different probability

Return JSON array of top 10 market IDs with brief reason:
[
  {{"market_id": "...", "reason": "clearly mispriced because...", "priority": 1-10}}
]

Only return the JSON, no other text."""
