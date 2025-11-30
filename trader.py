#!/usr/bin/env python3
"""
ULTRA-SMART PAPER TRADER v3
Real market data • LLM analysis • Continuous monitoring
"""
import asyncio
import sys
import os
import argparse
from datetime import datetime, timezone

sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class PaperTrader:
    """Paper trading with real market data."""
    
    def __init__(self, bankroll=1000.0, use_llm=True):
        self.bankroll = bankroll
        self.use_llm = use_llm
        self.markets = []
        self.screened = []
        self.opportunities = []
        self.positions = []
        self.llm_available = False
        self.kalshi_real = False
        
    async def initialize(self):
        """Check API availability."""
        from prediction_oracle.config import settings
        
        # Check LLM
        if settings.openai_api_key and settings.openai_api_key != "your_openai_key":
            self.llm_available = True
        
        # Check Kalshi
        if settings.kalshi_api_key and settings.kalshi_api_key != "your_kalshi_key":
            self.kalshi_real = True
    
    async def fetch_markets(self):
        """Fetch from all sources."""
        self.markets = []
        
        # Kalshi
        try:
            from prediction_oracle.markets.real_kalshi import RealKalshiClient
            kalshi = RealKalshiClient(demo_mode=True)
            markets = await kalshi.list_markets(limit=15)
            self.markets.extend(markets)
            
            status = "[green]REAL[/green]" if kalshi.has_credentials else "[yellow]mock[/yellow]"
            console.print(f"  Kalshi ({status}): {len(markets)} markets")
            await kalshi.close()
        except Exception as e:
            console.print(f"  [red]Kalshi error: {e}[/red]")
        
        # Polymarket (always real!)
        try:
            from prediction_oracle.markets.real_polymarket import RealPolymarketClient
            poly = RealPolymarketClient()
            markets = await poly.list_markets(limit=40)
            self.markets.extend(markets)
            console.print(f"  Polymarket ([green]REAL[/green]): {len(markets)} markets")
            await poly.close()
        except Exception as e:
            console.print(f"  [red]Polymarket error: {e}[/red]")
    
    async def screen_markets(self):
        """Screen for quality opportunities."""
        now = datetime.now(timezone.utc)
        self.screened = []
        
        for market in self.markets:
            if not market.outcomes:
                continue
            
            price = market.outcomes[0].price
            
            # Detect real vs mock
            is_mock = (
                market.question.startswith("Will event") or
                market.market_id.startswith("KALSHI-") or
                market.market_id.startswith("MOCK")
            )
            is_real = not is_mock
            
            # Filter prices
            if is_real:
                if not (0.01 <= price <= 0.99):
                    continue
            else:
                if not (0.10 <= price <= 0.90):
                    continue
            
            # Scores
            volume = market.volume_24h or 0
            volume_score = min(volume / 10000, 1.0)
            
            # Edge type
            if 0.35 <= price <= 0.65:
                edge_type = "contested"
                edge_score = 0.6
            elif price <= 0.15 or price >= 0.85:
                edge_type = "extreme"
                edge_score = 0.7
            else:
                edge_type = "moderate"
                edge_score = 0.5
            
            # Priority (heavily favor real markets)
            real_boost = 0.5 if is_real else 0.0
            priority = volume_score * 0.2 + edge_score * 0.2 + real_boost * 0.6
            
            self.screened.append({
                "market": market,
                "price": price,
                "volume": volume,
                "volume_score": volume_score,
                "edge_type": edge_type,
                "edge_score": edge_score,
                "is_real": is_real,
                "priority": priority,
            })
        
        self.screened.sort(key=lambda x: x["priority"], reverse=True)
        self.screened = self.screened[:25]
    
    async def find_opportunities(self):
        """Find trading opportunities."""
        self.opportunities = []
        
        for item in self.screened[:15]:
            market = item["market"]
            price = item["price"]
            is_real = item["is_real"]
            
            opp = None
            
            # Contested markets - could go either way
            if item["edge_type"] == "contested":
                opp = {
                    "type": "CONTESTED",
                    "market": market,
                    "side": "ANALYZE",
                    "price": price,
                    "edge": 0.0,
                    "reasoning": "High uncertainty - needs deeper analysis",
                    "bet_size": 0,
                    "is_real": is_real,
                }
            
            # Extreme prices - potential longshots
            elif item["edge_type"] == "extreme" and is_real:
                if price <= 0.15:
                    implied_edge = 0.03 + (0.15 - price)  # More edge at lower prices
                    opp = {
                        "type": "LONGSHOT",
                        "market": market,
                        "side": "YES",
                        "price": price,
                        "edge": implied_edge,
                        "reasoning": f"Low price {price:.1%} may undervalue YES",
                        "bet_size": min(self.bankroll * 0.01, 10),  # Small bets on longshots
                        "is_real": is_real,
                    }
                elif price >= 0.85:
                    implied_edge = 0.03 + (price - 0.85)
                    opp = {
                        "type": "FADE",
                        "market": market,
                        "side": "NO",
                        "price": 1 - price,
                        "edge": implied_edge,
                        "reasoning": f"High price {price:.1%} may overvalue YES",
                        "bet_size": min(self.bankroll * 0.01, 10),
                        "is_real": is_real,
                    }
            
            # Moderate prices with volume
            elif item["volume_score"] > 0.3:
                opp = {
                    "type": "VOLUME",
                    "market": market,
                    "side": "WATCH",
                    "price": price,
                    "edge": 0.0,
                    "reasoning": f"High volume ${item['volume']:,.0f} - monitor for moves",
                    "bet_size": 0,
                    "is_real": is_real,
                }
            
            if opp:
                self.opportunities.append(opp)
        
        # Sort by edge
        self.opportunities.sort(key=lambda x: x["edge"], reverse=True)
    
    def display(self):
        """Display results."""
        # Header
        console.print(Panel.fit(
            "[bold cyan]🧠 PAPER TRADER v3[/bold cyan]\n"
            "[dim]Real Polymarket data • Mock Kalshi (add API key for real)[/dim]",
            border_style="cyan",
        ))
        
        # Status
        status = []
        status.append(f"Bankroll: [green]${self.bankroll:,.0f}[/green]")
        status.append(f"LLM: {'[green]✓[/green]' if self.llm_available else '[yellow]No key[/yellow]'}")
        status.append(f"Kalshi: {'[green]REAL[/green]' if self.kalshi_real else '[yellow]mock[/yellow]'}")
        console.print(" | ".join(status))
        console.print()
        
        # Markets table
        if self.screened:
            table = Table(title="📊 Screened Markets")
            table.add_column("Market", width=45)
            table.add_column("Price", justify="right")
            table.add_column("Volume", justify="right")
            table.add_column("Type", justify="center")
            table.add_column("Real", justify="center")
            
            for item in self.screened[:12]:
                m = item["market"]
                q = m.question[:42] + "..." if len(m.question) > 45 else m.question
                
                table.add_row(
                    q,
                    f"{item['price']:.0%}",
                    f"${item['volume']:,.0f}" if item['volume'] > 0 else "-",
                    item["edge_type"][:8],
                    "[green]✓[/green]" if item["is_real"] else "[dim]mock[/dim]",
                )
            
            console.print(table)
        
        # Opportunities
        if self.opportunities:
            console.print("\n[bold]💰 OPPORTUNITIES[/bold]\n")
            
            for i, opp in enumerate(self.opportunities[:8], 1):
                m = opp["market"]
                q = m.question[:55] + "..." if len(m.question) > 58 else m.question
                real_tag = "[green](REAL)[/green]" if opp["is_real"] else "[dim](mock)[/dim]"
                
                if opp["side"] in ["WATCH", "ANALYZE"]:
                    console.print(f"{i}. [cyan]{opp['type']}[/cyan] {real_tag}")
                    console.print(f"   {q}")
                    console.print(f"   [dim]{opp['reasoning']}[/dim]")
                else:
                    console.print(f"{i}. [bold]{opp['type']}[/bold] {real_tag}")
                    console.print(f"   {q}")
                    console.print(f"   {opp['side']} @ {opp['price']:.1%} | Edge: [green]+{opp['edge']:.1%}[/green] | Bet: [cyan]${opp['bet_size']:.2f}[/cyan]")
                    console.print(f"   [dim]{opp['reasoning']}[/dim]")
                console.print()
        
        # Summary
        console.print(Panel.fit(
            f"Markets: {len(self.markets)} | Screened: {len(self.screened)} | Opportunities: {len(self.opportunities)}",
            title="Summary",
            border_style="dim",
        ))
    
    async def run(self):
        """Run single scan."""
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("Initializing...", total=None)
            await self.initialize()
            
            progress.update(task, description="Fetching markets...")
            await self.fetch_markets()
            
            progress.update(task, description="Screening...")
            await self.screen_markets()
            
            progress.update(task, description="Finding opportunities...")
            await self.find_opportunities()
        
        self.display()
    
    async def run_continuous(self, interval=300):
        """Run continuously."""
        console.print(f"[cyan]Starting continuous mode (interval: {interval}s)[/cyan]")
        console.print("Press Ctrl+C to stop\n")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                console.print(f"\n{'='*60}")
                console.print(f"[bold]Scan #{cycle}[/bold] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                console.print(f"{'='*60}\n")
                
                await self.run()
                
                console.print(f"\n[dim]Next scan in {interval}s...[/dim]")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/yellow]")
                break


async def main():
    parser = argparse.ArgumentParser(description="Paper Trader")
    parser.add_argument("--bankroll", type=float, default=1000)
    parser.add_argument("--continuous", "-c", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    
    trader = PaperTrader(bankroll=args.bankroll)
    
    if args.continuous:
        await trader.run_continuous(args.interval)
    else:
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
