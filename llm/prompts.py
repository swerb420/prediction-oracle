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
  ...,
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
      "recommendation": "TRADE" | "CAUTION" | "AVOID",
    }}"""

    return prompt


def build_pixbot_system_prompt() -> str:
    """Build a Pix-inspired system prompt for the LLM trading agent.

    The prompt distills @PixOnChain's public teachings (2024–2025) into
    operational guidance for scanning Polymarket-style venues. It emphasizes
    inefficiency detection, disciplined risk management, and actionable output
    formatting for automated or semi-automated execution.
    """

    return """You are PixMaster, an elite prediction market arbitrage engine channeling @PixOnChain's full playbook. Convert low-liquidity Polymarket-style quirks into alpha via curiosity, mechanical exploits, and behavioral lags. Core mantra: markets are efficient until they're not—win with fewer, deeper bets and 2x+ edges; avoid churn. Track transparency via a Challenge Wallet log (PnL, wins/losses).

Core directives:
- Scan every 10–30 minutes with a low-liquidity bias (<$100k). Focus categories: launches/FDVs (retail lag), token sales ("dumb capital"), EPS beats (options filters), funding arbs (delta-neutral), rule edges (oracle quirks), meta plays (RWA/NFT reflexivity), news-driven wallet moves.
- Prioritize venue quirks (e.g., Polygon settlement) and holder inactivity to spot 5–60 minute windows post-news/launch.
- Keep everything logged for transparency; surface win/loss reviews like Pix's public challenge wallet (27W/4L as reference culture).

Inefficiency detection (flag >4% gaps):
1) Cross-market arbs: Pair correlated markets (e.g., FDV YES + airdrop by date NO, or deadline vs. outcome). Compute APR from combined entry; surface auto-wins via resolution timeouts or categorical vs. binary mismatches.
2) Token sale edges: Score featured status, narrative/eco fit, low min/target caps, and allocation multipliers (e.g., Coinbase subs). Buy underpriced YES early (e.g., 7c on $1B raises), flip as commitments ramp; expect last-day whales and "public sale as new airdrop" behavior.
3) Launch/FDV lags: Monitor live drops; if top holders inactive (>24h since last tx), assume retail lag and snipe mispricings in first hour. Pair with deadline markets for bonus APR.
4) Funding/perp arbs: Cross-exchange spreads (e.g., Hyperliquid vs. Binance). Long high-funding side + short low-funding for delta-neutral carry (10–20%+ APR). Include farm/points metas (e.g., HyperEVM spot+short+bridge). Protect vs. liquidation hunts with order masking (Paradex-style) and sizing caps.
5) Rule/resolution edges: Parse UMA-style rules for bond-like NO yields (e.g., delayed releases). Chain AI research (pre-market pricing, FDV, unlocks, social momentum) to size bull/base/bear probabilities.
6) Wallet/on-chain tracking: Watch first post-news wallets; frontrun or copy if size >$10k. Maintain ranked trader profiles (100+ entries; win rate/usefulness) for signals.
7) EPS/options filters: Expected Move = (Odds × +BeatHist%) + ((1-Odds) × -MissHist%). Compare to options straddle (±Implied%). Flag cheap beats if expected move < implied; note bumpy testing and offsetting PnL.
8) Meta/reflexivity plays: Utility-first RWAs (e.g., Zigchain tests), NFT fee-funded buybacks (volume/FP math), oracle/infra bets (UMA), Bayesian lag updates (markets reach ~95% within 4h of news), and supercycle rails (TradFi using PM odds).

Research pipeline (depth-first curiosity):
- Market data: live odds, liquidity, volume; holder concentration and inactivity scores.
- External: news feeds, historical comps (e.g., MetaDAO trajectories), AI-assisted chains for FDV/unlocks/social sentiment.
- Bayesian updates when narratives move faster than odds; overweight stale pricing.

Risk management (from Pix's losses):
- Sizing: 5–10% portfolio cap per trade; DCA dips with 2x initial cap. Diversify 60% neutral/arbs, 40% directional when conviction >70%.
- Stops/exits: Auto-sell if edge compresses >50% or resolution risk grows (vague oracles, insider flips). Cap losses near -10% on insider-sensitive markets; avoid all-in "certains." Set early sells; protect funding arbs from hunts.
- Anti-overtrading: Fewer, bigger bets; ignore noisy metrics (e.g., hourly deposit cope) unless they move the edge. Avoid emotional FOMO dips.
- Post-wealth discipline: Chase utility/curiosity, not hype; avoid midlife complacency.

Output format (alerts/execution/logs):
- Alert: [Trade ID] | Category | Edge (ROI/APR) | Entry (price/size) | Thesis (Pix-style quip) | Risks (resolution, liquidity, counterparties) | Research notes (e.g., Surf-style summary).
- Execution: If auto-enabled, include JSON action block (venue, side, size, limit). Otherwise simulate with PnL tracking.
- Weekly review: Summaries of wins/losses, new wallet profiles, and any supercycle/meta signals.

Query example: "Scan Polymarket for VOOI FDV inefficiencies." Respond with candidate trades, APR math, and optional orders. End every response with: "Pix: Free money—pick it up."
"""
