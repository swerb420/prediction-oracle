#!/usr/bin/env python3
"""
Optimized Batch LLM Analysis with Category-Specific Prompts
Uses grok-4-1-fast-reasoning for better accuracy at lower cost
Batches markets by category for efficient processing
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

# Category-specific system prompts (compact for token efficiency)
CATEGORY_PROMPTS = {
    "SPORTS": """You are an expert sports betting analyst. Key factors:
- Injuries: Star player out = 3-7 point swing
- Rest: Back-to-back = 2-3 point disadvantage
- Home advantage: ~55-58% base win rate
- Weather for outdoor sports
Give probability based on matchup analysis.""",

    "POLITICS": """You are a political forecaster. Key factors:
- Polling aggregates (538, RCP) not individual polls
- Incumbency advantage ~70%
- Systematic polling bias 2-4 points
- Fundamentals > narratives
Base rates matter more than recent news.""",

    "CRYPTO": """You are a crypto analyst. Key factors:
- BTC dominance and momentum
- Macro (Fed, rates, liquidity)
- On-chain metrics if relevant
- Historical volatility patterns
Crypto moves fast - short timeframes favor momentum.""",

    "STOCKS": """You are a stock analyst. Key factors:
- Current price vs target in question
- Earnings/catalyst calendar
- Sector momentum
- Overall market conditions (VIX, SPY)
Markets are efficient - need specific edge.""",

    "WEATHER": """You are a meteorologist. Key factors:
- Use NWS/weather.gov forecasts as base
- Seasonal norms for location
- Current conditions and trends
Weather forecasts are very accurate <3 days out.""",

    "GEOPOLITICS": """You are a geopolitical analyst. Key factors:
- Historical base rates for similar events
- Current escalation/de-escalation signals
- Expert consensus vs market
- Default to status quo (things rarely change quickly)""",

    "AI_TECH": """You are a tech analyst. Key factors:
- Official announcements/roadmaps
- Historical accuracy of similar predictions
- Company track record on timelines
- Tech hype cycle position""",

    "ENTERTAINMENT": """You are an entertainment analyst. Key factors:
- Historical patterns (Oscar predictors, box office)
- Expert consensus (critics, industry)
- Social media sentiment
- Betting market movements""",

    "OTHER": """You are a superforecaster. Key factors:
- What's the base rate for this type of event?
- What specific evidence changes the base rate?
- What would change your mind?
Use 50% when truly uncertain."""
}

@dataclass
class BatchResult:
    market_id: str
    fair: float
    direction: str
    confidence: str
    reason: str
    raw_response: str


def get_category_prompt(category: str) -> str:
    """Get system prompt for category."""
    return CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["OTHER"])


def build_batch_prompt(markets: List[Dict], category: str) -> str:
    """Build efficient batch prompt for multiple markets in same category."""
    system = get_category_prompt(category)
    
    market_list = []
    for i, m in enumerate(markets, 1):
        extra = ""
        if m.get('current_price'):
            extra = f" (current: ${m['current_price']:,.0f})"
        market_list.append(
            f"{i}. {m['question'][:100]}\n"
            f"   YES price: {m['price']:.0%} | Closes: {m['hours_left']:.0f}h{extra}"
        )
    
    return f"""{system}

MARKETS TO ANALYZE:
{chr(10).join(market_list)}

For EACH market, estimate the TRUE probability it resolves YES.
Reply with one line per market in EXACTLY this format:
1. FAIR:XX% DIR:YES/NO CONF:H/M/L because [reason]
2. FAIR:XX% DIR:YES/NO CONF:H/M/L because [reason]
...

Rules:
- FAIR = your probability estimate (not the market's)
- DIR = YES if FAIR > market price, NO if FAIR < market price  
- CONF = H (>10% edge), M (5-10% edge), L (<5% edge)
- Keep reasons under 10 words"""


def parse_batch_response(text: str, markets: List[Dict]) -> List[BatchResult]:
    """Parse batch response into individual results."""
    results = []
    lines = [l.strip() for l in text.split('\n') if l.strip() and l[0].isdigit()]
    
    for i, market in enumerate(markets):
        result = BatchResult(
            market_id=market.get('id', ''),
            fair=market['price'],  # default to market price
            direction='SKIP',
            confidence='LOW',
            reason='',
            raw_response=''
        )
        
        if i < len(lines):
            line = lines[i]
            result.raw_response = line
            
            # Parse FAIR:XX%
            if 'FAIR:' in line.upper():
                try:
                    fair_part = line.upper().split('FAIR:')[1].split('%')[0].strip()
                    result.fair = float(fair_part) / 100
                except:
                    pass
            
            # Parse DIR:YES/NO
            if 'DIR:YES' in line.upper():
                result.direction = 'YES'
            elif 'DIR:NO' in line.upper():
                result.direction = 'NO'
            
            # Parse CONF:H/M/L
            if 'CONF:H' in line.upper():
                result.confidence = 'HIGH'
            elif 'CONF:M' in line.upper():
                result.confidence = 'MEDIUM'
            else:
                result.confidence = 'LOW'
            
            # Parse reason (after "because")
            if 'because' in line.lower():
                result.reason = line.lower().split('because')[1].strip()[:50]
        
        results.append(result)
    
    return results


async def analyze_batch_grok(
    markets: List[Dict],
    category: str,
    api_key: str,
    model: str = "grok-4-1-fast-reasoning"
) -> List[BatchResult]:
    """Analyze batch of markets using Grok."""
    if not markets:
        return []
    
    prompt = build_batch_prompt(markets, category)
    
    async with httpx.AsyncClient(timeout=90) as client:
        try:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50 * len(markets),  # ~50 tokens per market
                    "temperature": 0.2
                }
            )
            
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                return parse_batch_response(content, markets)
            else:
                print(f"   Grok batch error: {resp.status_code}")
                
        except Exception as e:
            print(f"   Grok batch exception: {e}")
    
    # Return defaults on failure
    return [BatchResult(m.get('id', ''), m['price'], 'SKIP', 'LOW', '', '') for m in markets]


async def analyze_batch_gpt(
    markets: List[Dict],
    category: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> List[BatchResult]:
    """Analyze batch of markets using GPT."""
    if not markets:
        return []
    
    prompt = build_batch_prompt(markets, category)
    
    async with httpx.AsyncClient(timeout=90) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50 * len(markets),
                    "temperature": 0.2
                }
            )
            
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                return parse_batch_response(content, markets)
            else:
                print(f"   GPT batch error: {resp.status_code}")
                
        except Exception as e:
            print(f"   GPT batch exception: {e}")
    
    return [BatchResult(m.get('id', ''), m['price'], 'SKIP', 'LOW', '', '') for m in markets]


async def analyze_markets_batched(
    markets: List[Dict],
    grok_key: Optional[str] = None,
    gpt_key: Optional[str] = None,
    batch_size: int = 5,
    use_grok4: bool = True
) -> List[Dict]:
    """
    Analyze markets in batches by category.
    Much more token-efficient than individual calls.
    
    Returns markets with added fields:
    - llm_fair, llm_direction, llm_confidence, llm_reason, edge
    """
    
    # Group by category
    by_category = {}
    for m in markets:
        cat = m.get('category', 'OTHER')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m)
    
    results = []
    grok_model = "grok-4-1-fast-reasoning" if use_grok4 else "grok-2-1212"
    
    for category, cat_markets in by_category.items():
        # Process in batches
        for i in range(0, len(cat_markets), batch_size):
            batch = cat_markets[i:i+batch_size]
            
            grok_results = []
            gpt_results = []
            
            # Run both in parallel if both keys available
            tasks = []
            if grok_key:
                tasks.append(analyze_batch_grok(batch, category, grok_key, grok_model))
            if gpt_key:
                tasks.append(analyze_batch_gpt(batch, category, gpt_key))
            
            if tasks:
                all_results = await asyncio.gather(*tasks)
                if grok_key and len(all_results) > 0:
                    grok_results = all_results[0]
                if gpt_key:
                    gpt_results = all_results[-1] if len(all_results) > 1 else all_results[0]
            
            # Combine results
            for j, market in enumerate(batch):
                grok = grok_results[j] if j < len(grok_results) else None
                gpt = gpt_results[j] if j < len(gpt_results) else None
                
                # Average fair values
                fairs = []
                if grok and grok.fair:
                    fairs.append(grok.fair)
                if gpt and gpt.fair:
                    fairs.append(gpt.fair)
                
                avg_fair = sum(fairs) / len(fairs) if fairs else market['price']
                
                # Calculate edge and direction
                edge_yes = avg_fair - market['price']
                edge_no = market['price'] - avg_fair
                
                if edge_yes > 0:
                    direction = 'YES'
                    edge = edge_yes
                else:
                    direction = 'NO'
                    edge = edge_no
                
                # Confidence from agreement
                confidences = []
                if grok:
                    confidences.append(grok.confidence)
                if gpt:
                    confidences.append(gpt.confidence)
                
                if 'HIGH' in confidences and abs(edge) > 0.10:
                    confidence = 'HIGH'
                elif abs(edge) > 0.05:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
                
                # Best reason
                reason = ''
                if grok and grok.reason:
                    reason = grok.reason
                elif gpt and gpt.reason:
                    reason = gpt.reason
                
                market['llm_fair'] = avg_fair
                market['llm_direction'] = direction
                market['llm_confidence'] = confidence
                market['llm_reason'] = reason
                market['edge'] = edge
                market['grok'] = {'fair': grok.fair, 'direction': grok.direction, 'confidence': grok.confidence} if grok else None
                market['gpt'] = {'fair': gpt.fair, 'direction': gpt.direction, 'confidence': gpt.confidence} if gpt else None
                
                results.append(market)
    
    return results


# Quick test
if __name__ == "__main__":
    import os
    
    test_markets = [
        {"id": "1", "question": "Will Bitcoin be above $100k by end of 2025?", "price": 0.45, "hours_left": 720, "category": "CRYPTO"},
        {"id": "2", "question": "Will ETH flip BTC market cap?", "price": 0.08, "hours_left": 720, "category": "CRYPTO"},
        {"id": "3", "question": "Will Lakers beat Celtics tonight?", "price": 0.42, "hours_left": 8, "category": "SPORTS"},
    ]
    
    async def test():
        grok_key = os.environ.get('XAI_API_KEY')
        results = await analyze_markets_batched(test_markets, grok_key=grok_key, batch_size=3)
        for r in results:
            print(f"{r['question'][:40]}... | Fair: {r['llm_fair']:.0%} | Edge: {r['edge']:+.1%} | {r['llm_direction']}")
    
    asyncio.run(test())
