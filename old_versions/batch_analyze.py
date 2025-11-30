#!/usr/bin/env python3
"""Efficient batch analysis - 1 Grok call for 8 markets."""
import asyncio
import httpx
import json
import os
import re
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
load_dotenv()

async def main():
    print("🔍 DEEP BATCH ANALYSIS - 7 Day Window")
    print("=" * 60)
    
    now = datetime.now(timezone.utc)
    all_markets = []
    
    # Fetch Kalshi (300 markets)
    print("Fetching Kalshi...")
    async with httpx.AsyncClient(timeout=30) as client:
        cursor = None
        for _ in range(3):
            params = {"limit": 100, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = await client.get("https://api.elections.kalshi.com/trade-api/v2/markets", params=params)
                if resp.status_code != 200:
                    break
                data = resp.json()
                for m in data.get("markets", []):
                    close_str = m.get("close_time") or m.get("expected_expiration_time")
                    if not close_str:
                        continue
                    try:
                        close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        hours = (close - now).total_seconds() / 3600
                        price = m.get("last_price", 50) / 100
                        if 0.05 <= price <= 0.95 and 1 <= hours <= 168:
                            all_markets.append({
                                "src": "KALSHI", "id": m.get("ticker"), "q": m.get("title"),
                                "price": price, "hours": hours, "vol": m.get("volume_24h", 0)
                            })
                    except:
                        pass
                cursor = data.get("cursor")
                if not cursor:
                    break
            except Exception as e:
                print(f"Kalshi error: {e}")
                break
    
    print(f"  Kalshi: {len([m for m in all_markets if m['src']=='KALSHI'])} markets")
    
    # Fetch Polymarket
    print("Fetching Polymarket...")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get("https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true", "limit": 100, "order": "volume24hr", "ascending": "false"})
            if resp.status_code == 200:
                for m in resp.json():
                    try:
                        prices_str = m.get("outcomePrices", "")
                        if not prices_str:
                            continue
                        prices = [float(p.strip().strip('"')) for p in prices_str.strip("[]").split(",") if p.strip()]
                        if len(prices) < 2 or not (0.05 < prices[0] < 0.95):
                            continue
                        end = m.get("endDate")
                        close = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else now + timedelta(days=30)
                        hours = (close - now).total_seconds() / 3600
                        if 1 <= hours <= 168:
                            all_markets.append({
                                "src": "POLY", "id": m.get("id"), "q": m.get("question"),
                                "price": prices[0], "hours": hours, "vol": float(m.get("volume24hr", 0) or 0)
                            })
                    except:
                        pass
        except Exception as e:
            print(f"Polymarket error: {e}")
    
    print(f"  Polymarket: {len([m for m in all_markets if m['src']=='POLY'])} markets")
    print(f"  Total: {len(all_markets)} markets in 7-day window")
    
    # Filter contested prices (25-75%) or high-volume extremes
    candidates = []
    for m in all_markets:
        p = m["price"]
        if 0.25 <= p <= 0.75:
            m["type"] = "CONTESTED"
            candidates.append(m)
        elif m["vol"] > 30000 and (p <= 0.15 or p >= 0.85):
            m["type"] = "EXTREME"
            candidates.append(m)
    
    candidates.sort(key=lambda x: (x["hours"], -x["vol"]))
    print(f"\n🎯 High-potential candidates: {len(candidates)}")
    
    # Take top 8 for analysis
    top_picks = candidates[:8]
    
    print("\n📋 TOP 8 FOR ANALYSIS:")
    print("-" * 60)
    for i, m in enumerate(top_picks, 1):
        hrs = m["hours"]
        ends = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
        print(f"{i}. [{m['src']}] {m['q'][:55]}")
        print(f"   YES: {m['price']:.0%} | Vol: ${m['vol']:,.0f} | Ends: {ends}")
    
    # Build batch prompt for Grok
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, ends {m['hours']:.0f}h" 
        for i, m in enumerate(top_picks, 1)
    ])
    
    prompt = f"""Prediction market analysis. For each, give TRUE probability and trading direction.

{markets_text}

Reply JSON only:
[{{"m":1,"fv":0.XX,"d":"YES/NO/HOLD","r":"10 word reason"}}]"""

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("\n❌ No XAI_API_KEY in .env")
        return
    
    print("\n🧠 Grok 4.1 analyzing (1 API call)...")
    
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "grok-4-1-fast-reasoning",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                }
            )
            
            if resp.status_code != 200:
                print(f"❌ API error: {resp.status_code} - {resp.text[:200]}")
                return
            
            content = resp.json()["choices"][0]["message"]["content"]
            
            # Parse JSON array
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                analyses = json.loads(match.group())
                
                print("\n" + "=" * 60)
                print("🏆 GROK'S WINNERS (edge >= 3%)")
                print("=" * 60)
                
                winners = []
                for a in analyses:
                    idx = a.get("m", 1) - 1
                    if idx >= len(top_picks):
                        continue
                    
                    m = top_picks[idx]
                    fv = float(a.get("fv", m["price"]))
                    d = a.get("d", "HOLD").upper()
                    r = a.get("r", "")
                    
                    # Calculate edge
                    if d == "YES":
                        edge = fv - m["price"]
                        entry = m["price"]
                    elif d == "NO":
                        edge = m["price"] - fv
                        entry = 1 - m["price"]
                    else:
                        edge = 0
                        entry = m["price"]
                    
                    hrs = m["hours"]
                    ends = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
                    
                    if edge >= 0.03:
                        payout = 5 / entry if entry > 0 else 0
                        profit = payout - 5
                        
                        print(f"\n🔥 {d} @ {entry:.0%} → Fair: {fv:.0%}")
                        print(f"   Edge: +{edge:.1%} | $5 → ${payout:.2f} (+${profit:.2f})")
                        print(f"   [{m['src']}] {m['q'][:55]}")
                        print(f"   Ends: {ends} | Reason: {r}")
                        
                        winners.append({
                            "source": m["src"],
                            "market_id": m["id"],
                            "question": m["q"],
                            "current_price": m["price"],
                            "fair_value": fv,
                            "direction": d,
                            "edge": edge,
                            "entry_price": entry,
                            "hours_left": hrs,
                            "volume_24h": m["vol"],
                            "reason": r,
                            "bet_5_payout": payout,
                            "analyzed_at": now.isoformat()
                        })
                    else:
                        print(f"\n⚪ {d} {m['q'][:40]}... (edge {edge:.1%})")
                
                # Save winners
                if winners:
                    with open("/root/prediction_oracle/winners.json", "w") as f:
                        json.dump(winners, f, indent=2)
                    print(f"\n✅ Saved {len(winners)} winners to winners.json")
                else:
                    print("\n📉 No markets with 3%+ edge found")
                
                print(f"\n💰 API Cost: ~$0.003 (1 call)")
                
            else:
                print("Raw response:", content)
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
