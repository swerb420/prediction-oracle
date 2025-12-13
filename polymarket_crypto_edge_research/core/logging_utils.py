"""
Logging utilities using loguru with rich formatting.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

# Global console for rich output
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru with rich formatting and file output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        rotation: Log file rotation size
        retention: Log file retention period
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with rich formatting
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
        )
    
    logger.info(f"Logging initialized at {level} level")


def get_logger(name: str) -> Any:
    """
    Get a logger instance bound to a specific name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Bound logger instance
    """
    return logger.bind(name=name)


class LogContext:
    """Context manager for structured logging with extra fields."""
    
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self._token: Any = None
    
    def __enter__(self) -> "LogContext":
        self._token = logger.contextualize(**self.kwargs)
        self._token.__enter__()
        return self
    
    def __exit__(self, *args: Any) -> None:
        if self._token:
            self._token.__exit__(*args)


def log_trade(
    action: str,
    symbol: str,
    side: str,
    size: float,
    price: float,
    **extra: Any
) -> None:
    """Log a trade with structured format."""
    emoji = "🟢" if side.upper() == "BUY" else "🔴"
    logger.info(
        f"{emoji} TRADE | {action} | {symbol} | {side} | "
        f"Size: {size:.4f} | Price: ${price:,.2f} | {extra}"
    )


def log_signal(
    symbol: str,
    signal: str,
    confidence: float,
    source: str,
    **extra: Any
) -> None:
    """Log a trading signal."""
    emoji_map = {"LONG": "📈", "SHORT": "📉", "HOLD": "⏸️", "SKIP": "⏭️"}
    emoji = emoji_map.get(signal.upper(), "❓")
    logger.info(
        f"{emoji} SIGNAL | {symbol} | {signal} | "
        f"Conf: {confidence:.1%} | Source: {source} | {extra}"
    )


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """Log performance metrics in a formatted way."""
    lines = [f"{prefix}Performance Metrics:"]
    for key, value in metrics.items():
        if "pct" in key.lower() or "rate" in key.lower():
            lines.append(f"  {key}: {value:.2%}")
        elif "sharpe" in key.lower() or "ratio" in key.lower():
            lines.append(f"  {key}: {value:.3f}")
        else:
            lines.append(f"  {key}: {value:,.2f}")
    logger.info("\n".join(lines))
