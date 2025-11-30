"""Prompt templates for LLM market analysis."""

from datetime import datetime

from ..markets import Market


def build_probability_prompt(markets: list[Market]) -> str:
    """
    Build a prompt for LLM to analyze markets and provide probability estimates.
    
    Args:
        markets: List of markets to analyze
        
    Returns:
        Formatted prompt string
    """
    now = datetime.utcnow()
    
    market_descriptions = []
    for market in markets:
        hours_until_close = (market.close_time - now).total_seconds() / 3600
        
        outcomes_desc = []
        for outcome in market.outcomes:
            outcomes_desc.append(
                f"  - {outcome.label}: ${outcome.price:.3f} "
                f"(liquidity: ${outcome.liquidity or 0:.0f})"
            )
        
        market_desc = f"""
Market ID: {market.market_id}
Venue: {market.venue.value}
Question: {market.question}
Rules: {market.rules or 'Standard binary resolution'}
Category: {market.category}
Closes in: {hours_until_close:.1f} hours
Current Prices:
{chr(10).join(outcomes_desc)}
24h Volume: ${market.volume_24h or 0:.0f}
"""
        market_descriptions.append(market_desc.strip())
    
    prompt = f"""You are an expert prediction market analyst. Analyze the following {len(markets)} markets and provide calibrated probability estimates.

For each market, you must:
1. Analyze the question and resolution rules carefully
2. Consider base rates, current events, and market context
3. Provide a CALIBRATED probability (not just gut feeling) - your estimates will be scored on accuracy
4. Identify any ambiguities or risks in the resolution rules
5. Compare your probability to the current market price

Current timestamp: {now.isoformat()}

MARKETS TO ANALYZE:
{'=' * 80}
{chr(10).join(market_descriptions)}
{'=' * 80}

You MUST respond with ONLY a valid JSON array containing your analysis for each market. Do not include any other text.

Format (STRICT JSON):
[
  {{
    "market_id": "exact market ID from above",
    "outcome_id": "outcome ID (usually YES/NO)",
    "p_true": 0.0-1.0,  // Your calibrated probability this outcome occurs
    "confidence": 0.0-1.0,  // How confident are you in this estimate?
    "rule_risks": ["any ambiguous resolution criteria", "time zone issues", etc.],
    "edge_vs_market": 0.0,  // Your p_true minus current market price (auto-calculated is OK)
    "notes": "Brief 1-2 sentence rationale for your probability"
  }},
  ...
]

CRITICAL RULES:
- p_true should be your TRUE belief, calibrated for accuracy (Brier score)
- If you're 50/50 uncertain, say 0.50, don't pick a side
- Flag ANY rule ambiguity that could affect resolution
- Consider selection bias: markets exist because outcomes are uncertain
- Don't just anchor to market price - find genuine mispricing
- Response must be VALID JSON only, no markdown, no explanation outside JSON

Begin your JSON response:"""
    
    return prompt


def build_rule_check_prompt(market: Market) -> str:
    """
    Build a prompt to check for resolution rule ambiguities.
    
    Args:
        market: Market to analyze
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a prediction market rules expert. Analyze this market for resolution ambiguities.

Market: {market.question}
Rules: {market.rules or 'Standard binary resolution'}
Venue: {market.venue.value}
Close Time: {market.close_time.isoformat()}

Identify ANY potential issues:
1. Ambiguous language or definitions
2. Missing resolution source
3. Time zone confusion
4. Edge cases not covered
5. Conflicting interpretation possibilities

Respond with JSON:
{{
  "ambiguity_score": 0.0-1.0,  // 0 = crystal clear, 1 = very ambiguous
  "issues": ["issue 1", "issue 2", ...],
  "recommendation": "TRADE" | "CAUTION" | "AVOID"
}}"""
    
    return prompt
