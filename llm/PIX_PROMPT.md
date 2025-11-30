# PixMaster System Prompt

This prompt packages @PixOnChain's public teachings (2024–2025) into a reusable
system message for the prediction-market LLM agent.

## Purpose
- Focus on Polymarket-style inefficiencies, deadline quirks, and behavioral lags.
- Encourage fewer, higher-conviction trades with explicit risk controls and
  transparency via challenge-style logging.
- Provide structured output for alerts, auto-execution, and weekly reviews.

## Using the prompt
```python
from llm.prompts import build_pixbot_system_prompt

pix_prompt = build_pixbot_system_prompt()
# Supply `pix_prompt` as the system message when initializing your LLM client.
```

## Core behaviors included
- **Scanning cadence:** 10–30 minute sweeps; prefer low-liquidity markets (<$100k).
- **Edge detection:** cross-market arbs, token-sale mispricings, launch-window
  snipes, funding-rate spreads, rule-based yields, wallet-tracking/copy trades,
  EPS vs. straddle filters, and meta/reflexivity plays (RWA, NFT fee buybacks,
  oracle/infra bets, TradFi rails).
- **Research pipeline:** live odds + liquidity pulls, holder concentration and
  inactivity scoring, external news hooks, AI-assisted FDV/unlock/social probes,
  and Bayesian updates on narrative shifts.
- **Risk management:** portfolio caps per trade, DCA limits, stop/exit triggers
  when edges compress or rules turn vague, protection vs. liquidation hunts,
  diversification splits (neutral vs. directional), and anti-overtrading cues.
- **Output format:** alert template with edge size, entry, thesis, risks, and
  research notes; optional JSON for execution; weekly PnL-style summaries and
  profile updates for tracked wallets.

## Example
Prompt the model with:

```
Scan Polymarket for VOOI FDV inefficiencies.
```

You should receive structured candidate trades with APR estimates, optional
execution JSON, and the signature sign-off: "Pix: Free money—pick it up."
