#!/usr/bin/env python3
"""
Paper Trading Runner - Uses all ultra-smart enhancements.
Run this to start paper trading with full signal integration.
"""

import asyncio
import logging
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from prediction_oracle.config import settings
from prediction_oracle.markets.router import MarketRouter
from prediction_oracle.signals import SmartScreener, free_api
from prediction_oracle.llm import (
    cost_tracker,
    smart_router,
    create_fast_provider,
    EnhancedOracle,
)
from prediction_oracle.strategies import (
    EnhancedConservativeStrategy,
    EnhancedLongshotStrategy,
)
from prediction_oracle.risk import BankrollManager
from prediction_oracle.storage import create_tables

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("paper_trader")

console = Console()


async def run_paper_trader():
    """Run one iteration of paper trading."""
    console.print(Panel.fit(
        "[bold green]🚀 ULTRA-SMART PAPER TRADER[/bold green]\n"
        f"Budget: ${settings.llm_daily_budget:.2f}/day | "
        f"Mode: {settings.trading_mode}"
    ))
    
    # Initialize
    await create_tables()
    
    # Components
    market_router = MarketRouter(mock_mode=True)  # Use mock for now
    screener = SmartScreener(
        free_api=free_api,
        min_volume=settings.screener_min_volume,
        top_n_for_deep=settings.screener_top_n
    )
    
    bankroll = BankrollManager(settings.initial_bankroll)
    
    console.print(f"\n[cyan]Bankroll: ${bankroll.current:.2f}[/cyan]")
    console.print(f"[cyan]LLM Budget: ${cost_tracker.remaining_budget:.2f} remaining[/cyan]\n")
    
    # Step 1: Fetch markets
    console.print("[bold]Step 1: Fetching markets...[/bold]")
    
    from prediction_oracle.markets import Venue
    
    markets = []
    try:
        kalshi_client = market_router.get_client(Venue.KALSHI)
        kalshi_markets = await kalshi_client.list_markets(limit=30)
        markets.extend(kalshi_markets)
        console.print(f"  ✓ Kalshi: {len(kalshi_markets)} markets")
    except Exception as e:
        console.print(f"  ⚠ Kalshi error: {e}")
    
    try:
        poly_client = market_router.get_client(Venue.POLYMARKET)
        poly_markets = await poly_client.list_markets(limit=30)
        markets.extend(poly_markets)
        console.print(f"  ✓ Polymarket: {len(poly_markets)} markets")
    except Exception as e:
        console.print(f"  ⚠ Polymarket error: {e}")
    
    console.print(f"  Total: {len(markets)} markets\n")
    
    if not markets:
        console.print("[yellow]No markets found. Using mock mode.[/yellow]")
        return
    
    # Step 2: Smart screening
    console.print("[bold]Step 2: Smart screening (FREE signals)...[/bold]")
    
    screened = await screener.screen_markets(markets)
    
    # Display screened markets
    table = Table(title="Top Screened Markets")
    table.add_column("Market", width=50)
    table.add_column("Price", justify="right")
    table.add_column("Attention", justify="right")
    table.add_column("News", justify="right")
    table.add_column("Priority", justify="right")
    
    for m in screened[:10]:
        table.add_row(
            m.question[:47] + "..." if len(m.question) > 50 else m.question,
            f"{m.current_price:.1%}",
            f"{m.attention_score:.2f}",
            str(m.news_volume),
            f"{m.priority_score:.3f}"
        )
    
    console.print(table)
    console.print(f"\n  Screened to {len(screened)} candidates\n")
    
    # Step 3: Strategy evaluation
    console.print("[bold]Step 3: Strategy evaluation...[/bold]")
    
    # Load config for strategies
    import yaml
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {
            "strategies": {
                "conservative": {"enabled": True, "min_edge": 0.04},
                "longshot": {"enabled": True, "max_price": 0.15}
            },
            "llm_groups": {}
        }
    
    # Get original market objects for screened
    screened_ids = {m.market_id for m in screened}
    filtered_markets = [m for m in markets if m.market_id in screened_ids]
    
    # Conservative strategy
    if config.get("strategies", {}).get("conservative", {}).get("enabled", True):
        console.print("  Running Enhanced Conservative Strategy...")
        try:
            conservative = EnhancedConservativeStrategy(config["strategies"], bankroll)
            cons_recs = await conservative.evaluate_markets(filtered_markets[:10])
            
            if cons_recs:
                console.print(f"  ✓ Found {len(cons_recs)} conservative opportunities")
                for rec in cons_recs[:3]:
                    console.print(f"    • {rec['market'].question[:50]}...")
                    console.print(f"      Edge: {rec['oracle_result'].edge:.1%}, Size: ${rec['bet_size']:.2f}")
            else:
                console.print("  ⚠ No conservative opportunities found")
                
            await conservative.close()
        except Exception as e:
            console.print(f"  ⚠ Conservative error: {e}")
    
    # Longshot strategy
    if config.get("strategies", {}).get("longshot", {}).get("enabled", True):
        console.print("\n  Running Enhanced Longshot Strategy...")
        try:
            longshot = EnhancedLongshotStrategy(config["strategies"], bankroll)
            long_recs = await longshot.evaluate_markets(filtered_markets[:10])
            
            if long_recs:
                console.print(f"  ✓ Found {len(long_recs)} longshot opportunities")
                for rec in long_recs[:3]:
                    console.print(f"    • {rec['market'].question[:50]}...")
                    console.print(f"      Upside: {rec['upside']:.1f}x, Size: ${rec['bet_size']:.2f}")
            else:
                console.print("  ⚠ No longshot opportunities found")
                
            await longshot.close()
        except Exception as e:
            console.print(f"  ⚠ Longshot error: {e}")
    
    # Step 4: Cost summary
    console.print("\n[bold]Step 4: Cost Summary[/bold]")
    console.print(f"  LLM spend this run: ${cost_tracker.daily_spend:.4f}")
    console.print(f"  Remaining budget: ${cost_tracker.remaining_budget:.2f}")
    
    # Cleanup
    await screener.close()
    await market_router.close_all()
    
    console.print("\n[bold green]✅ Paper trading iteration complete![/bold green]")


async def run_continuous(interval_seconds: int = 300):
    """Run paper trader continuously."""
    console.print(f"\n[bold]Starting continuous paper trading (every {interval_seconds}s)...[/bold]")
    console.print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            await run_paper_trader()
            console.print(f"\n[dim]Next run in {interval_seconds}s...[/dim]\n")
            await asyncio.sleep(interval_seconds)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            await asyncio.sleep(60)  # Wait 1 min on error


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        asyncio.run(run_continuous(settings.scan_interval_seconds))
    else:
        asyncio.run(run_paper_trader())
