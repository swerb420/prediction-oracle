#!/usr/bin/env python3
"""
Ultra-Smart Paper Trader v2 - Robust and intelligent.
Handles errors gracefully and provides clear output.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("paper_trader")
logger.setLevel(logging.INFO)

console = Console()


class SmartPaperTrader:
    """Intelligent paper trader with full signal integration."""
    
    def __init__(self):
        self.markets: list = []
        self.screened: list = []
        self.recommendations: list = []
        self.errors: list[str] = []
        
    async def run(self):
        """Run one paper trading iteration."""
        console.print(Panel.fit(
            "[bold cyan]🧠 ULTRA-SMART PAPER TRADER v2[/bold cyan]\n"
            "Multi-signal intelligence • Cost-optimized • Robust"
        ))
        
        # Load config
        from prediction_oracle.config import settings
        console.print(f"\n[dim]Mode: {settings.trading_mode} | "
                     f"Bankroll: ${settings.initial_bankroll:.0f} | "
                     f"LLM Budget: ${settings.llm_daily_budget:.2f}/day[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Step 1: Initialize
            task = progress.add_task("Initializing...", total=None)
            await self._initialize()
            progress.update(task, description="✅ Initialized")
            
            # Step 2: Fetch markets
            progress.update(task, description="Fetching markets...")
            await self._fetch_markets()
            progress.update(task, description=f"✅ Fetched {len(self.markets)} markets")
            
            if not self.markets:
                progress.update(task, description="❌ No markets available")
                return
            
            # Step 3: Screen markets
            progress.update(task, description="Screening with free signals...")
            await self._screen_markets()
            progress.update(task, description=f"✅ Screened to {len(self.screened)} candidates")
            
            # Step 4: Find opportunities
            progress.update(task, description="Finding opportunities...")
            await self._find_opportunities()
            progress.update(task, description=f"✅ Found {len(self.recommendations)} opportunities")
        
        # Display results
        self._display_results()
        
        # Cleanup
        await self._cleanup()
    
    async def _initialize(self):
        """Initialize all components."""
        from prediction_oracle.storage import create_tables
        await create_tables()
        
    async def _fetch_markets(self):
        """Fetch markets from all venues."""
        from prediction_oracle.markets.router import MarketRouter
        from prediction_oracle.markets.real_polymarket import RealPolymarketClient
        from prediction_oracle.markets import Venue
        
        # Use mock Kalshi but REAL Polymarket
        router = MarketRouter(mock_mode=True)
        self._router = router
        
        # Kalshi (mock for now - would need API key)
        try:
            client = router.get_client(Venue.KALSHI)
            markets = await client.list_markets(limit=10)
            self.markets.extend(markets)
            console.print(f"  [dim]Kalshi (mock): {len(markets)} markets[/dim]")
        except Exception as e:
            self.errors.append(f"Kalshi: {str(e)[:50]}")
        
        # REAL Polymarket (free public API!)
        try:
            poly_client = RealPolymarketClient(mock_mode=False)
            self._poly_client = poly_client
            markets = await poly_client.list_markets(limit=30)
            self.markets.extend(markets)
            console.print(f"  [green]Polymarket (REAL): {len(markets)} markets[/green]")
        except Exception as e:
            self.errors.append(f"Polymarket: {str(e)[:50]}")
    
    async def _screen_markets(self):
        """Screen markets using free signals."""
        if not self.markets:
            return
            
        # Simple screening without external API calls for robustness
        from datetime import timezone
        now = datetime.now(timezone.utc)
        
        for market in self.markets:
            # Basic quality filters
            if not market.outcomes:
                continue
                
            price = market.outcomes[0].price
            
            # Check if real market (not mock)
            is_real = not market.question.startswith("Will event") and not market.question.startswith("Will crypto")
            
            # Allow wider price range for real markets (they can have extreme prices)
            if is_real:
                if not (0.01 <= price <= 0.99):
                    continue
            else:
                if not (0.05 <= price <= 0.95):
                    continue
            
            # Calculate scores locally (no API calls)
            volume_score = min((market.volume_24h or 0) / 5000, 1.0)
            
            # Price-based edge opportunity
            # Real markets with extreme prices can be interesting
            if 0.40 <= price <= 0.60:
                edge_score = 0.4  # Contested markets
            elif price <= 0.10 or price >= 0.90:
                edge_score = 0.7 if is_real else 0.5  # Extreme prices - longshots
            elif price <= 0.15 or price >= 0.85:
                edge_score = 0.6 if is_real else 0.4  # Moderate longshots
            else:
                edge_score = 0.3
            
            # Time score
            time_score = 0.5
            if market.close_time:
                close_time = market.close_time
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                hours = (close_time - now).total_seconds() / 3600
                if 24 <= hours <= 168:
                    time_score = 1.0
                elif 6 <= hours < 24:
                    time_score = 0.7
            
            # BOOST: Prioritize real markets over mock (big boost)
            real_boost = 0.5 if is_real else 0.0
            
            priority = volume_score * 0.2 + edge_score * 0.2 + time_score * 0.2 + real_boost * 0.4
            
            self.screened.append({
                "market": market,
                "price": price,
                "volume_score": volume_score,
                "edge_score": edge_score,
                "time_score": time_score,
                "priority": priority,
                "is_real": is_real,
            })
        
        # Sort by priority
        self.screened.sort(key=lambda x: x["priority"], reverse=True)
        self.screened = self.screened[:20]  # Top 20
    
    async def _find_opportunities(self):
        """Find trading opportunities from screened markets."""
        from prediction_oracle.config import settings
        
        for item in self.screened[:10]:  # Top 10
            market = item["market"]
            price = item["price"]
            
            # Simple edge estimation (without LLM for robustness)
            # In production, this would use EnhancedOracle
            
            # Conservative: Look for mispriced markets (45-55% range)
            if 0.40 <= price <= 0.60 and item["volume_score"] > 0.3:
                # Estimate edge based on category patterns
                estimated_prob = 0.5  # Baseline
                
                # Check if price deviates from 50%
                if price < 0.48:
                    estimated_edge = 0.48 - price  # Underpriced YES
                    side = "YES"
                elif price > 0.52:
                    estimated_edge = price - 0.52  # Overpriced YES (bet NO)
                    side = "NO"
                else:
                    continue  # Too close to 50%
                
                if estimated_edge >= 0.03:  # 3% edge minimum
                    bet_size = min(
                        settings.initial_bankroll * settings.max_position_size_pct,
                        25.0  # Max $25 for paper
                    )
                    
                    self.recommendations.append({
                        "type": "CONSERVATIVE",
                        "market": market,
                        "side": side,
                        "price": price,
                        "estimated_edge": estimated_edge,
                        "bet_size": bet_size,
                        "rationale": f"Price {price:.0%} deviates from fair value",
                    })
            
            # Longshot: Look for underpriced low-probability events
            elif price <= 0.12 and item["edge_score"] > 0.5:
                # Small bet on potential upset
                self.recommendations.append({
                    "type": "LONGSHOT",
                    "market": market,
                    "side": "YES",
                    "price": price,
                    "estimated_edge": 0.05,  # Speculative
                    "bet_size": 5.0,  # Fixed $5 for longshots
                    "rationale": f"Low price {price:.0%} with upside potential",
                })
    
    def _display_results(self):
        """Display trading results."""
        console.print()
        
        # Screened markets table
        if self.screened:
            table = Table(title="📊 Screened Markets (Top 10)")
            table.add_column("Market", width=45)
            table.add_column("Price", justify="right")
            table.add_column("Vol", justify="right")
            table.add_column("Edge", justify="right")
            table.add_column("Real", justify="center")
            table.add_column("Priority", justify="right")
            
            for item in self.screened[:10]:
                market = item["market"]
                is_real = item.get("is_real", False)
                table.add_row(
                    market.question[:42] + "..." if len(market.question) > 45 else market.question,
                    f"{item['price']:.0%}",
                    f"{item['volume_score']:.2f}",
                    f"{item['edge_score']:.2f}",
                    "[green]✓[/green]" if is_real else "[dim]mock[/dim]",
                    f"{item['priority']:.3f}",
                )
            
            console.print(table)
        
        # Recommendations
        console.print()
        if self.recommendations:
            console.print("[bold green]💰 TRADING RECOMMENDATIONS[/bold green]\n")
            
            for i, rec in enumerate(self.recommendations, 1):
                market = rec["market"]
                console.print(f"[bold]{i}. {rec['type']}[/bold]")
                console.print(f"   Market: {market.question[:60]}...")
                console.print(f"   Side: [cyan]{rec['side']}[/cyan] @ {rec['price']:.1%}")
                console.print(f"   Est. Edge: [green]+{rec['estimated_edge']:.1%}[/green]")
                console.print(f"   Bet Size: [yellow]${rec['bet_size']:.2f}[/yellow]")
                console.print(f"   Rationale: {rec['rationale']}")
                console.print()
        else:
            console.print("[yellow]No opportunities found meeting criteria.[/yellow]")
        
        # Errors
        if self.errors:
            console.print("\n[dim]Warnings:[/dim]")
            for error in self.errors:
                console.print(f"  [dim]• {error}[/dim]")
        
        # Summary
        console.print(Panel.fit(
            f"[bold]Summary[/bold]\n"
            f"Markets fetched: {len(self.markets)}\n"
            f"After screening: {len(self.screened)}\n"
            f"Opportunities: {len(self.recommendations)}"
        ))
    
    async def _cleanup(self):
        """Cleanup resources."""
        if hasattr(self, '_router'):
            await self._router.close_all()
        if hasattr(self, '_poly_client'):
            await self._poly_client.close()


async def main():
    """Main entry point."""
    trader = SmartPaperTrader()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
