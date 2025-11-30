#!/usr/bin/env python3
"""
🎯 PREDICTION ORACLE - REAL-TIME PAPER TRADER
100% Real Data from Kalshi + Polymarket
Focus on markets ending in 24-48 hours for fast validation
"""
import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
import json

sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

from dotenv import load_dotenv
load_dotenv()

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

BET_SIZE = 5.00


class RealKalshiAPI:
    """Real Kalshi market data - NO MOCK DATA."""
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_markets(self, limit: int = 100) -> list[dict]:
        """Fetch real Kalshi markets."""
        markets = []
        cursor = None
        
        while len(markets) < limit:
            params = {"limit": min(100, limit - len(markets)), "status": "open"}
            if cursor:
                params["cursor"] = cursor
            
            try:
                resp = await self.client.get(f"{self.BASE_URL}/markets", params=params)
                if resp.status_code != 200:
                    console.print(f"[red]Kalshi API error: {resp.status_code}[/red]")
                    break
                
                data = resp.json()
                batch = data.get("markets", [])
                if not batch:
                    break
                
                markets.extend(batch)
                cursor = data.get("cursor")
                if not cursor:
                    break
                    
            except Exception as e:
                console.print(f"[red]Kalshi error: {e}[/red]")
                break
        
        return markets
    
    async def close(self):
        await self.client.aclose()


class RealPolymarketAPI:
    """Real Polymarket data."""
    
    BASE_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_markets(self, limit: int = 100) -> list[dict]:
        """Fetch real Polymarket markets."""
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/markets",
                params={"closed": "false", "active": "true", "limit": limit, "order": "volume24hr", "ascending": "false"}
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            console.print(f"[red]Polymarket error: {e}[/red]")
        return []
    
    async def close(self):
        await self.client.aclose()


class GrokAnalyzer:
    """Grok 4.1 Fast Reasoning for market analysis."""
    
    def __init__(self):
        self.api_key = os.getenv("XAI_API_KEY")
        self.model = os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning")
        self.calls = 0
        self.cost = 0.0
    
    async def analyze(self, question: str, price: float, closes_in: str, volume: float = 0) -> dict | None:
        """Analyze a market with Grok."""
        if not self.api_key:
            return None
        
        prompt = f"""Prediction market analysis needed.

Market: "{question}"
Current YES price: {price:.1%}
Closes in: {closes_in}
Volume: ${volume:,.0f}

Based on current events and your knowledge, what is the TRUE probability this resolves YES?

Reply ONLY with JSON:
{{"fair_value": 0.XX, "direction": "yes" or "no" or "hold", "confidence": "low"/"medium"/"high", "reason": "15 words max"}}"""

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}
                )
                
                if resp.status_code == 200:
                    self.calls += 1
                    self.cost += 0.002
                    content = resp.json()["choices"][0]["message"]["content"]
                    
                    # Parse JSON
                    import re
                    match = re.search(r'\{[^}]+\}', content)
                    if match:
                        return json.loads(match.group())
                        
        except Exception as e:
            console.print(f"[dim]Grok: {e}[/dim]")
        
        return None


class RealTimePaperTrader:
    """Paper trader using 100% REAL market data."""
    
    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.kalshi = RealKalshiAPI()
        self.polymarket = RealPolymarketAPI()
        self.grok = GrokAnalyzer()
        
        self.markets = []  # All markets
        self.soon_markets = []  # Markets ending soon (24-48h)
        self.opportunities = []  # Grok-analyzed opportunities
        self.paper_trades = []  # Our paper positions
    
    async def fetch_all_markets(self):
        """Fetch REAL markets from both sources."""
        console.print("\n[bold]📡 Fetching REAL Market Data[/bold]")
        
        self.markets = []
        now = datetime.now(timezone.utc)
        
        # Kalshi - REAL DATA
        kalshi_raw = await self.kalshi.get_markets(limit=200)
        kalshi_count = 0
        for m in kalshi_raw:
            try:
                close_time_str = m.get("close_time") or m.get("expected_expiration_time")
                if not close_time_str:
                    continue
                
                close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                hours_left = (close_time - now).total_seconds() / 3600
                
                # Get price (Kalshi uses cents)
                yes_price = m.get("last_price", 50) / 100
                if yes_price <= 0:
                    yes_bid = m.get("yes_bid", 0) / 100
                    yes_ask = m.get("yes_ask", 100) / 100
                    yes_price = (yes_bid + yes_ask) / 2
                
                self.markets.append({
                    "source": "KALSHI",
                    "id": m.get("ticker"),
                    "question": m.get("title", "Unknown"),
                    "yes_price": yes_price,
                    "no_price": 1 - yes_price,
                    "volume_24h": m.get("volume_24h", 0),
                    "liquidity": m.get("liquidity", 0) / 100,
                    "close_time": close_time,
                    "hours_left": hours_left,
                    "yes_bid": m.get("yes_bid", 0) / 100,
                    "yes_ask": m.get("yes_ask", 100) / 100,
                    "raw": m,
                })
                kalshi_count += 1
            except:
                continue
        
        console.print(f"  [green]Kalshi: {kalshi_count} REAL markets[/green]")
        
        # Polymarket - REAL DATA
        poly_raw = await self.polymarket.get_markets(limit=100)
        poly_count = 0
        for m in poly_raw:
            try:
                # Parse price
                prices_str = m.get("outcomePrices", "")
                if not prices_str or prices_str == "[]":
                    continue
                
                prices = [float(p.strip().strip('"')) for p in prices_str.strip("[]").split(",") if p.strip()]
                if len(prices) < 2:
                    continue
                
                yes_price = prices[0]
                if yes_price <= 0 or yes_price >= 1:
                    continue
                
                # Parse close time
                end_date = m.get("endDate")
                if end_date:
                    close_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                else:
                    close_time = now + timedelta(days=30)
                
                hours_left = (close_time - now).total_seconds() / 3600
                
                self.markets.append({
                    "source": "POLYMARKET",
                    "id": m.get("id"),
                    "question": m.get("question", "Unknown"),
                    "yes_price": yes_price,
                    "no_price": 1 - yes_price,
                    "volume_24h": float(m.get("volume24hr", 0) or 0),
                    "liquidity": float(m.get("liquidity", 0) or 0),
                    "close_time": close_time,
                    "hours_left": hours_left,
                    "raw": m,
                })
                poly_count += 1
            except:
                continue
        
        console.print(f"  [green]Polymarket: {poly_count} REAL markets[/green]")
        console.print(f"  [bold]Total: {len(self.markets)} real markets[/bold]")
    
    def filter_soon_markets(self, min_hours: float = 2, max_hours: float = 72):
        """Filter to markets ending soon for fast validation."""
        self.soon_markets = []
        
        for m in self.markets:
            hours = m["hours_left"]
            price = m["yes_price"]
            
            # Must be ending within our window
            if not (min_hours <= hours <= max_hours):
                continue
            
            # Must have tradeable price
            if not (0.02 <= price <= 0.98):
                continue
            
            # Calculate potential edge score
            if 0.35 <= price <= 0.65:
                edge_potential = "contested"
            elif price <= 0.15 or price >= 0.85:
                edge_potential = "extreme"
            else:
                edge_potential = "moderate"
            
            m["edge_potential"] = edge_potential
            self.soon_markets.append(m)
        
        # Sort by hours left (soonest first)
        self.soon_markets.sort(key=lambda x: x["hours_left"])
        
        console.print(f"\n[bold]⏰ Markets Ending in {min_hours}-{max_hours}h: {len(self.soon_markets)}[/bold]")
    
    async def analyze_with_grok(self, max_markets: int = 12):
        """Use Grok to find edge in soon-ending markets."""
        console.print(f"\n[bold]🧠 Grok 4.1 Fast Reasoning Analysis[/bold]")
        
        self.opportunities = []
        
        for m in self.soon_markets[:max_markets]:
            hours = m["hours_left"]
            if hours < 1:
                closes_in = f"{int(hours * 60)} minutes"
            elif hours < 24:
                closes_in = f"{hours:.1f} hours"
            else:
                closes_in = f"{hours/24:.1f} days"
            
            analysis = await self.grok.analyze(
                m["question"],
                m["yes_price"],
                closes_in,
                m["volume_24h"]
            )
            
            if analysis:
                fair_value = analysis.get("fair_value", m["yes_price"])
                direction = analysis.get("direction", "hold")
                confidence = analysis.get("confidence", "low")
                reason = analysis.get("reason", "")
                
                # Calculate edge
                if direction == "yes":
                    edge = fair_value - m["yes_price"]
                    side = "YES"
                    entry = m["yes_price"]
                elif direction == "no":
                    edge = m["yes_price"] - fair_value
                    side = "NO"
                    entry = m["no_price"]
                else:
                    edge = 0
                    side = "HOLD"
                    entry = m["yes_price"]
                
                # Only include if edge > 3%
                if edge >= 0.03:
                    # Risk/Reward for $5 bet
                    risk = BET_SIZE
                    if entry > 0:
                        payout = BET_SIZE / entry
                        reward = payout - BET_SIZE
                    else:
                        reward = 0
                    rr = reward / risk if risk > 0 else 0
                    
                    self.opportunities.append({
                        **m,
                        "side": side,
                        "entry": entry,
                        "fair_value": fair_value,
                        "edge": edge,
                        "confidence": confidence,
                        "reason": reason,
                        "risk": risk,
                        "reward": reward,
                        "rr_ratio": rr,
                        "closes_in": closes_in,
                    })
                    
                    console.print(f"  [green]✓[/green] {m['source']} | {side} @ {entry:.0%} | Edge: +{edge:.0%} | {closes_in}")
                    console.print(f"    {m['question'][:60]}...")
                else:
                    console.print(f"  [dim]- {m['question'][:50]}... (edge {edge:.1%})[/dim]")
            
            await asyncio.sleep(0.3)  # Rate limit
        
        # Sort by edge
        self.opportunities.sort(key=lambda x: x["edge"], reverse=True)
    
    def display_results(self):
        """Display opportunities with risk/reward."""
        console.print(Panel.fit(
            "[bold cyan]🎯 REAL-TIME PAPER TRADER[/bold cyan]\n"
            f"[dim]100% Real Data | Grok 4.1 Analysis | ${BET_SIZE:.2f} Bets[/dim]",
            border_style="cyan"
        ))
        
        # Stats
        console.print(f"\nBankroll: [green]${self.bankroll:,.0f}[/green] | "
                      f"Grok Calls: {self.grok.calls} (~${self.grok.cost:.3f})")
        
        # Soon-ending markets table
        if self.soon_markets:
            console.print(f"\n[bold]📊 Markets Ending Soon ({len(self.soon_markets)} total)[/bold]")
            
            table = Table()
            table.add_column("Source", width=10)
            table.add_column("Market", width=45)
            table.add_column("YES", justify="right")
            table.add_column("Vol 24h", justify="right")
            table.add_column("Ends In", justify="right")
            
            for m in self.soon_markets[:15]:
                hours = m["hours_left"]
                if hours < 1:
                    ends = f"{int(hours*60)}m"
                elif hours < 24:
                    ends = f"{hours:.0f}h"
                else:
                    ends = f"{hours/24:.1f}d"
                
                q = m["question"][:42] + "..." if len(m["question"]) > 45 else m["question"]
                
                table.add_row(
                    f"[{'green' if m['source']=='POLYMARKET' else 'blue'}]{m['source'][:4]}[/]",
                    q,
                    f"{m['yes_price']:.0%}",
                    f"${m['volume_24h']:,.0f}" if m['volume_24h'] else "-",
                    f"[{'red' if hours < 6 else 'yellow' if hours < 24 else 'white'}]{ends}[/]",
                )
            
            console.print(table)
        
        # Trading opportunities
        if self.opportunities:
            console.print(f"\n[bold]💰 GROK'S TOP PICKS - $5 BET ANALYSIS[/bold]\n")
            
            for i, opp in enumerate(self.opportunities[:8], 1):
                src = opp["source"][:4]
                q = opp["question"]
                if len(q) > 65:
                    q = q[:62] + "..."
                
                console.print(f"[bold]{i}. {opp['side']}[/bold] @ {opp['entry']:.1%} → Fair: {opp['fair_value']:.1%}")
                console.print(f"   [{src}] {q}")
                console.print(f"   [green]Edge: +{opp['edge']:.1%}[/green] | Confidence: {opp['confidence']} | Ends: [yellow]{opp['closes_in']}[/yellow]")
                console.print(f"   [cyan]$5 Bet → Risk: ${opp['risk']:.2f} | Win: ${opp['reward']:.2f} | R:R = 1:{opp['rr_ratio']:.1f}[/cyan]")
                console.print(f"   [dim]Grok: {opp['reason']}[/dim]")
                console.print()
        else:
            console.print("\n[yellow]No high-edge opportunities found in soon-ending markets.[/yellow]")
        
        # Summary
        kalshi_count = len([m for m in self.markets if m["source"] == "KALSHI"])
        poly_count = len([m for m in self.markets if m["source"] == "POLYMARKET"])
        
        console.print(Panel.fit(
            f"Kalshi: {kalshi_count} | Polymarket: {poly_count} | "
            f"Ending Soon: {len(self.soon_markets)} | Opportunities: {len(self.opportunities)}",
            title="Summary",
            border_style="dim"
        ))
    
    def record_paper_trade(self, opp: dict):
        """Record a paper trade for tracking."""
        self.paper_trades.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": opp["source"],
            "market_id": opp["id"],
            "question": opp["question"],
            "side": opp["side"],
            "entry_price": opp["entry"],
            "fair_value": opp["fair_value"],
            "edge": opp["edge"],
            "confidence": opp["confidence"],
            "bet_size": BET_SIZE,
            "potential_win": opp["reward"],
            "closes_at": opp["close_time"].isoformat(),
            "status": "OPEN",
        })
        
        # Save to file
        with open("/root/prediction_oracle/paper_trades.json", "w") as f:
            json.dump(self.paper_trades, f, indent=2)
    
    async def run(self, max_hours: float = 48):
        """Run single analysis cycle."""
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
            task = progress.add_task("Fetching markets...", total=None)
            await self.fetch_all_markets()
            
            progress.update(task, description="Filtering soon-ending markets...")
            self.filter_soon_markets(min_hours=1, max_hours=max_hours)
        
        await self.analyze_with_grok(max_markets=15)
        self.display_results()
        
        # Record top opportunities as paper trades
        if self.opportunities:
            console.print("\n[bold]📝 Recording Paper Trades...[/bold]")
            for opp in self.opportunities[:5]:
                self.record_paper_trade(opp)
                console.print(f"  Recorded: {opp['side']} {opp['question'][:40]}...")
            
            console.print(f"\n[green]Paper trades saved to paper_trades.json[/green]")
        
        await self.kalshi.close()
        await self.polymarket.close()
    
    async def run_continuous(self, interval: int = 600, max_hours: float = 48):
        """Run continuously."""
        console.print(f"[cyan]Continuous mode: scanning every {interval}s for markets ending in {max_hours}h[/cyan]")
        console.print("Press Ctrl+C to stop\n")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                console.print(f"\n{'='*70}")
                console.print(f"[bold]Scan #{cycle}[/bold] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                console.print('='*70)
                
                await self.run(max_hours=max_hours)
                
                console.print(f"\n[dim]Next scan in {interval}s...[/dim]")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                await asyncio.sleep(60)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", "-c", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=600, help="Scan interval (seconds)")
    parser.add_argument("--hours", type=float, default=48, help="Max hours until close")
    parser.add_argument("--bankroll", type=float, default=1000, help="Starting bankroll")
    args = parser.parse_args()
    
    trader = RealTimePaperTrader(bankroll=args.bankroll)
    
    if args.continuous:
        await trader.run_continuous(interval=args.interval, max_hours=args.hours)
    else:
        await trader.run(max_hours=args.hours)


if __name__ == "__main__":
    asyncio.run(main())
