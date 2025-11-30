#!/usr/bin/env python3
"""
🧠 PREDICTION ORACLE - PAPER TRADER
Real Polymarket data + Grok 4.1 Analysis + $5 Bet Risk/Reward
"""
import asyncio
import sys
import os
from datetime import datetime, timezone
from typing import Optional

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

# Config
BET_SIZE = 5.00  # $5 per trade


class GrokAnalyzer:
    """Use Grok 4.1 Fast Reasoning for market analysis."""
    
    def __init__(self):
        self.api_key = os.getenv("XAI_API_KEY")
        self.model = os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning")
        self.available = bool(self.api_key)
        self.calls = 0
        self.cost = 0.0
    
    async def analyze_market(self, question: str, price: float, volume: float) -> dict | None:
        """Get Grok's analysis of a prediction market."""
        if not self.available:
            return None
        
        prompt = f"""You are a prediction market analyst. Analyze this market:

Market: "{question}"
Current YES Price: {price:.1%} (${price:.2f})
Volume 24h: ${volume:,.0f}

Based on current events and probability assessment, provide your analysis in JSON:
{{"fair_value": 0.XX, "edge": "yes/no/none", "confidence": "low/medium/high", "reasoning": "1 sentence max"}}

Only output the JSON, nothing else."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                    }
                )
                
                if resp.status_code == 200:
                    self.calls += 1
                    self.cost += 0.001  # ~$0.001 per call estimate
                    
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Parse JSON from response
                    import json
                    import re
                    
                    # Extract JSON
                    json_match = re.search(r'\{[^}]+\}', content)
                    if json_match:
                        return json.loads(json_match.group())
                
                return None
                
        except Exception as e:
            console.print(f"[dim]Grok error: {e}[/dim]")
            return None


class PaperTrader:
    """Paper trader with real data and LLM analysis."""
    
    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.markets = []
        self.screened = []
        self.opportunities = []
        self.positions = []
        self.pnl = 0.0
        self.grok = GrokAnalyzer()
    
    async def fetch_markets(self):
        """Fetch real markets from Polymarket."""
        self.markets = []
        
        # Polymarket (real data, free!)
        try:
            from prediction_oracle.markets.real_polymarket import RealPolymarketClient
            poly = RealPolymarketClient()
            markets = await poly.list_markets(limit=50)
            self.markets.extend(markets)
            console.print(f"  Polymarket: [green]{len(markets)} real markets[/green]")
            await poly.close()
        except Exception as e:
            console.print(f"  [red]Polymarket error: {e}[/red]")
        
        # Kalshi (mock for now - real API needs RSA signing)
        try:
            from prediction_oracle.markets.real_kalshi import RealKalshiClient
            kalshi = RealKalshiClient()
            markets = await kalshi.list_markets(limit=15)
            self.markets.extend(markets)
            console.print(f"  Kalshi: [yellow]{len(markets)} mock markets[/yellow]")
            await kalshi.close()
        except Exception as e:
            console.print(f"  [red]Kalshi error: {e}[/red]")
    
    async def screen_markets(self):
        """Screen for tradeable markets."""
        now = datetime.now(timezone.utc)
        self.screened = []
        
        for market in self.markets:
            if not market.outcomes:
                continue
            
            price = market.outcomes[0].price
            volume = market.volume_24h or 0
            
            # Check if real market
            is_mock = market.market_id.startswith("KALSHI-") or market.market_id.startswith("MOCK")
            is_real = not is_mock
            
            # Filter
            if is_real and not (0.02 <= price <= 0.98):
                continue
            if not is_real and not (0.15 <= price <= 0.85):
                continue
            
            # Score
            volume_score = min(volume / 50000, 1.0)
            
            if 0.30 <= price <= 0.70:
                edge_type = "contested"
            elif price <= 0.15 or price >= 0.85:
                edge_type = "extreme"
            else:
                edge_type = "moderate"
            
            # Heavily favor real markets
            priority = volume_score * 0.3 + (0.7 if is_real else 0.0)
            
            self.screened.append({
                "market": market,
                "price": price,
                "volume": volume,
                "edge_type": edge_type,
                "is_real": is_real,
                "priority": priority,
            })
        
        self.screened.sort(key=lambda x: x["priority"], reverse=True)
        self.screened = self.screened[:20]
    
    async def analyze_opportunities(self):
        """Use Grok to analyze top markets."""
        self.opportunities = []
        
        console.print("\n[bold]🧠 Grok 4.1 Analysis[/bold]")
        
        # Analyze top real markets with Grok
        real_markets = [m for m in self.screened if m["is_real"]][:8]
        
        for item in real_markets:
            market = item["market"]
            price = item["price"]
            volume = item["volume"]
            
            # Get Grok's analysis
            analysis = await self.grok.analyze_market(
                market.question,
                price,
                volume
            )
            
            if analysis:
                fair_value = analysis.get("fair_value", price)
                edge_dir = analysis.get("edge", "none")
                confidence = analysis.get("confidence", "low")
                reasoning = analysis.get("reasoning", "")
                
                # Calculate edge
                if edge_dir == "yes":
                    edge = fair_value - price
                    side = "YES"
                    entry_price = price
                elif edge_dir == "no":
                    edge = price - fair_value
                    side = "NO"
                    entry_price = 1 - price
                else:
                    edge = 0
                    side = "NONE"
                    entry_price = price
                
                if edge > 0.02:  # At least 2% edge
                    # Calculate $5 bet risk/reward
                    risk = BET_SIZE  # Max loss
                    reward = BET_SIZE * (1 / entry_price - 1) if entry_price > 0 else 0
                    rr_ratio = reward / risk if risk > 0 else 0
                    
                    self.opportunities.append({
                        "market": market,
                        "side": side,
                        "price": entry_price,
                        "fair_value": fair_value if side == "YES" else 1 - fair_value,
                        "edge": edge,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "risk": risk,
                        "reward": reward,
                        "rr_ratio": rr_ratio,
                        "is_real": True,
                    })
                    
                    console.print(f"  [green]✓[/green] {market.question[:50]}...")
                    console.print(f"    → {side} @ {entry_price:.1%} | Edge: +{edge:.1%} | R:R = 1:{rr_ratio:.1f}")
                else:
                    console.print(f"  [dim]- {market.question[:50]}... (no edge)[/dim]")
            
            await asyncio.sleep(0.5)  # Rate limit
        
        self.opportunities.sort(key=lambda x: x["edge"], reverse=True)
    
    def display_results(self):
        """Display trading opportunities with risk/reward."""
        
        console.print(Panel.fit(
            "[bold cyan]🎯 PREDICTION ORACLE - PAPER TRADER[/bold cyan]\n"
            f"[dim]Powered by Grok 4.1 Fast Reasoning | ${BET_SIZE:.2f} bets[/dim]",
            border_style="cyan"
        ))
        
        # Status
        console.print(f"\nBankroll: [green]${self.bankroll:,.0f}[/green] | "
                      f"Grok Calls: {self.grok.calls} | "
                      f"Est. Cost: ${self.grok.cost:.4f}")
        
        # Markets table
        if self.screened:
            console.print("\n[bold]📊 Top Screened Markets[/bold]")
            table = Table()
            table.add_column("Market", width=48)
            table.add_column("Price", justify="right")
            table.add_column("Volume", justify="right")
            table.add_column("Type", justify="center")
            
            for item in self.screened[:10]:
                m = item["market"]
                q = m.question[:45] + "..." if len(m.question) > 48 else m.question
                real_marker = "[green]●[/green]" if item["is_real"] else "[dim]○[/dim]"
                
                table.add_row(
                    f"{real_marker} {q}",
                    f"{item['price']:.0%}",
                    f"${item['volume']:,.0f}" if item['volume'] else "-",
                    item["edge_type"],
                )
            
            console.print(table)
        
        # Opportunities with Risk/Reward
        if self.opportunities:
            console.print("\n[bold]💰 TRADING OPPORTUNITIES ($5 BETS)[/bold]\n")
            
            for i, opp in enumerate(self.opportunities[:6], 1):
                m = opp["market"]
                q = m.question[:60] + "..." if len(m.question) > 63 else m.question
                
                console.print(f"[bold]{i}. {opp['side']}[/bold] @ {opp['price']:.1%}")
                console.print(f"   {q}")
                console.print(f"   [green]Edge: +{opp['edge']:.1%}[/green] | Confidence: {opp['confidence']}")
                console.print(f"   [cyan]$5 Bet → Risk: ${opp['risk']:.2f} | Win: ${opp['reward']:.2f} | R:R = 1:{opp['rr_ratio']:.1f}[/cyan]")
                console.print(f"   [dim]{opp['reasoning']}[/dim]")
                console.print()
        else:
            console.print("\n[yellow]No high-confidence opportunities found this scan.[/yellow]")
        
        # Summary
        console.print(Panel.fit(
            f"Markets: {len(self.markets)} | Screened: {len(self.screened)} | "
            f"Opportunities: {len(self.opportunities)}",
            title="Summary",
            border_style="dim"
        ))
    
    async def run(self):
        """Run single scan."""
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("Fetching markets...", total=None)
            await self.fetch_markets()
            
            progress.update(task, description="Screening...")
            await self.screen_markets()
        
        await self.analyze_opportunities()
        self.display_results()
    
    async def run_continuous(self, interval: int = 300):
        """Run continuously."""
        console.print(f"[cyan]Continuous mode - scanning every {interval}s[/cyan]")
        console.print("Press Ctrl+C to stop\n")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                console.print(f"\n{'='*70}")
                console.print(f"[bold]Scan #{cycle}[/bold] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                console.print('='*70 + "\n")
                
                await self.run()
                
                console.print(f"\n[dim]Next scan in {interval}s...[/dim]")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/yellow]")
                break


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", "-c", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--bankroll", type=float, default=1000)
    args = parser.parse_args()
    
    trader = PaperTrader(bankroll=args.bankroll)
    
    if args.continuous:
        await trader.run_continuous(args.interval)
    else:
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
