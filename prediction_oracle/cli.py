"""Command-line interface for the prediction oracle."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from .config import settings
from .logging_config import setup_logging
from .runner import OracleScheduler

app = typer.Typer(
    name="oracle",
    help="Prediction Oracle: LLM-driven prediction market trading",
)
console = Console()


@app.command()
def run(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
    ),
    mode: str = typer.Option(
        None,
        "--mode",
        "-m",
        help="Trading mode: research, paper, or live (overrides .env)",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock data for testing",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
):
    """
    Run the prediction oracle main loop.
    
    Examples:
        oracle run --config config.yaml --mode research
        oracle run --mode paper --log-level DEBUG
        oracle run --mode live  # CAUTION: Real money!
    """
    # Setup logging
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Determine mode
    trading_mode = mode or settings.trading_mode
    
    # Safety check for live mode
    if trading_mode == "live" and not typer.confirm(
        "⚠️  You are about to run in LIVE mode with REAL MONEY. Continue?"
    ):
        console.print("[red]Aborted.[/red]")
        raise typer.Exit(1)
    
    # Verify config exists
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Starting Prediction Oracle[/green]")
    console.print(f"Mode: [bold]{trading_mode}[/bold]")
    console.print(f"Config: {config_path}")
    console.print(f"Mock data: {mock}")
    console.print()
    
    # Create and run scheduler
    scheduler = OracleScheduler(
        config_path=str(config_path),
        mode=trading_mode,
        mock_mode=mock,
    )
    
    try:
        asyncio.run(scheduler.main_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def test(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
    ),
):
    """
    Run a single test iteration with mock data.
    """
    setup_logging(level="DEBUG")
    logger = logging.getLogger(__name__)
    
    console.print("[cyan]Running test iteration with mock data...[/cyan]")
    
    scheduler = OracleScheduler(
        config_path=config,
        mode="research",
        mock_mode=True,
    )
    
    async def test_run():
        await scheduler.initialize()
        await scheduler.run_once()
        await scheduler.cleanup()
    
    try:
        asyncio.run(test_run())
        console.print("[green]✓ Test completed successfully![/green]")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        console.print(f"[red]✗ Test failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def init(
    directory: str = typer.Argument(
        ".",
        help="Directory to initialize (default: current directory)",
    ),
):
    """
    Initialize a new prediction oracle workspace.
    """
    import shutil
    
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy example files
    example_env = target_dir / ".env.example"
    env_file = target_dir / ".env"
    config_file = target_dir / "config.yaml"
    
    console.print(f"Initializing prediction oracle in {target_dir}")
    
    # Create .env if it doesn't exist
    if not env_file.exists() and example_env.exists():
        shutil.copy(example_env, env_file)
        console.print(f"[green]Created {env_file}[/green]")
        console.print("[yellow]⚠️  Edit .env with your API keys![/yellow]")
    
    console.print(f"\n[green]Workspace initialized![/green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Edit {env_file} with your API keys")
    console.print(f"  2. Review {config_file}")
    console.print(f"  3. Run: oracle test")
    console.print(f"  4. Run: oracle run --mode research")


@app.command()
def version():
    """Show version information."""
    from . import __version__
    
    console.print(f"Prediction Oracle v{__version__}")


@app.command()
def whale(
    min_amount: float = typer.Option(
        25000.0,
        "--min-amount",
        "-a",
        help="Minimum trade size in USD to alert",
    ),
    discord: str = typer.Option(
        None,
        "--discord",
        "-d",
        help="Discord webhook URL (overrides .env)",
    ),
    telegram_token: str = typer.Option(
        None,
        "--telegram-token",
        help="Telegram bot token (overrides .env)",
    ),
    telegram_chat: str = typer.Option(
        None,
        "--telegram-chat",
        help="Telegram chat ID (overrides .env)",
    ),
    copy_trade: bool = typer.Option(
        False,
        "--copy",
        help="Enable copy trading (DANGEROUS!)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Run the real-time whale scanner.
    
    Monitors Polymarket for large trades and sends instant alerts.
    
    Examples:
        oracle whale --min-amount 50000
        oracle whale --discord https://discord.com/api/webhooks/xxx
        oracle whale --copy  # Enable copy trading (use with caution!)
    """
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Check for required API key
    if not settings.alchemy_api_key:
        console.print("[red]Error: ALCHEMY_API_KEY required for whale scanner[/red]")
        console.print("Get a free key at: https://www.alchemy.com/")
        raise typer.Exit(1)
    
    # Safety check for copy trading
    if copy_trade and not typer.confirm(
        "⚠️  Copy trading enabled. You will automatically trade real money. Continue?"
    ):
        console.print("[red]Aborted.[/red]")
        raise typer.Exit(1)
    
    from .signals.whale_scanner import WhaleScannerConfig, WhaleFilter, run_whale_scanner
    
    config = WhaleScannerConfig(
        alchemy_api_key=settings.alchemy_api_key,
        discord_webhook_url=discord or settings.discord_webhook_url,
        telegram_bot_token=telegram_token or settings.telegram_bot_token,
        telegram_chat_id=telegram_chat or settings.telegram_chat_id,
        filter=WhaleFilter(
            min_amount_usd=min_amount,
            min_price_impact_pct=settings.whale_min_price_impact,
            max_market_volume_usd=settings.whale_max_market_volume,
            only_labeled_wallets=settings.whale_only_labeled,
            min_wallet_win_rate=settings.whale_min_win_rate,
        ),
        enable_copy_trading=copy_trade,
        copy_trade_max_usd=settings.copy_trade_max_usd,
    )
    
    console.print("[green]🐋 Starting Whale Scanner[/green]")
    console.print(f"Min trade size: ${min_amount:,.0f}")
    console.print(f"Discord: {'✓' if config.discord_webhook_url else '✗'}")
    console.print(f"Telegram: {'✓' if config.telegram_bot_token else '✗'}")
    console.print(f"Copy trading: {'⚠️ ENABLED' if copy_trade else 'Disabled'}")
    console.print()
    
    try:
        asyncio.run(run_whale_scanner(config))
    except KeyboardInterrupt:
        console.print("\n[yellow]Whale scanner stopped.[/yellow]")
    except Exception as e:
        logger.error(f"Whale scanner error: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def wallets(
    action: str = typer.Argument(
        "list",
        help="Action: list, add, import, export, stats",
    ),
    address: str = typer.Option(
        None,
        "--address",
        "-a",
        help="Wallet address for add/stats",
    ),
    label: str = typer.Option(
        None,
        "--label",
        "-l",
        help="Label for wallet",
    ),
    file: str = typer.Option(
        None,
        "--file",
        "-f",
        help="File for import/export",
    ),
    min_trades: int = typer.Option(
        10,
        "--min-trades",
        help="Minimum trades for list/leaderboard",
    ),
):
    """
    Manage tracked whale wallets.
    
    Examples:
        oracle wallets list
        oracle wallets add -a 0x123... -l "Sharp Trader #1"
        oracle wallets stats -a 0x123...
        oracle wallets import -f wallets.json
        oracle wallets export -f wallet_stats.json
    """
    setup_logging(level="INFO")
    
    from .signals.whale_db import WhaleDatabase
    
    async def run_action():
        db = WhaleDatabase()
        await db.connect()
        
        try:
            if action == "list":
                wallets = await db.get_all_wallets(min_trades=min_trades)
                console.print(f"\n[bold]Tracked Wallets ({len(wallets)})[/bold]\n")
                
                for w in wallets[:50]:
                    label_str = f"[cyan]{w.label}[/cyan]" if w.label else "Unlabeled"
                    console.print(
                        f"{w.address[:10]}...{w.address[-8:]} | "
                        f"{label_str} | "
                        f"Trades: {w.total_trades} | "
                        f"Win: {w.win_rate*100:.0f}% | "
                        f"PnL: ${w.total_pnl_usd:,.0f}"
                    )
            
            elif action == "add":
                if not address:
                    console.print("[red]Address required: --address 0x...[/red]")
                    return
                await db.add_wallet(address, label=label)
                console.print(f"[green]Added wallet: {address}[/green]")
            
            elif action == "stats":
                if not address:
                    console.print("[red]Address required: --address 0x...[/red]")
                    return
                stats = await db.get_wallet(address)
                if stats:
                    console.print(f"\n[bold]Wallet Stats: {address}[/bold]\n")
                    console.print(f"Label: {stats.label or 'None'}")
                    console.print(f"Total Trades: {stats.total_trades}")
                    console.print(f"Win Rate: {stats.win_rate*100:.1f}%")
                    console.print(f"Total PnL: ${stats.total_pnl_usd:,.2f}")
                    console.print(f"Volume: ${stats.total_volume_usd:,.2f}")
                    console.print(f"Reputation: {stats.reputation_score:.2f}")
                else:
                    console.print("[yellow]Wallet not found[/yellow]")
            
            elif action == "import":
                if not file:
                    console.print("[red]File required: --file wallets.json[/red]")
                    return
                await db.import_known_wallets(file)
                console.print(f"[green]Imported wallets from {file}[/green]")
            
            elif action == "export":
                if not file:
                    file = "wallet_stats.json"
                await db.export_stats(file)
                console.print(f"[green]Exported to {file}[/green]")
            
            else:
                console.print(f"[red]Unknown action: {action}[/red]")
                console.print("Available: list, add, stats, import, export")
        
        finally:
            await db.close()
    
    asyncio.run(run_action())


if __name__ == "__main__":
    app()
