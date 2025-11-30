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


if __name__ == "__main__":
    app()
