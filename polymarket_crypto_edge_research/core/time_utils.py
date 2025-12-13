"""
Time utilities for consistent timestamp handling across the system.
All times are UTC-based to avoid timezone confusion.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

IntervalType = Literal["1m", "5m", "15m", "1h", "4h", "1d"]

# Interval to seconds mapping
INTERVAL_SECONDS: dict[IntervalType, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def to_timestamp_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def from_timestamp_ms(ts_ms: int) -> datetime:
    """Convert milliseconds timestamp to UTC datetime."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def from_timestamp_s(ts_s: float | int) -> datetime:
    """Convert seconds timestamp to UTC datetime."""
    return datetime.fromtimestamp(ts_s, tz=timezone.utc)


def round_to_interval(dt: datetime, interval: IntervalType) -> datetime:
    """
    Round datetime down to the nearest interval boundary.
    
    Args:
        dt: Datetime to round
        interval: Interval to round to
        
    Returns:
        Rounded datetime
    """
    seconds = INTERVAL_SECONDS[interval]
    timestamp = int(dt.timestamp())
    rounded_ts = (timestamp // seconds) * seconds
    return datetime.fromtimestamp(rounded_ts, tz=timezone.utc)


def get_next_interval(dt: datetime, interval: IntervalType) -> datetime:
    """
    Get the start of the next interval after dt.
    
    Args:
        dt: Reference datetime
        interval: Interval type
        
    Returns:
        Start of next interval
    """
    seconds = INTERVAL_SECONDS[interval]
    current_rounded = round_to_interval(dt, interval)
    return current_rounded + timedelta(seconds=seconds)


def time_until_next_interval(interval: IntervalType) -> timedelta:
    """
    Get time remaining until the next interval boundary.
    
    Args:
        interval: Interval type
        
    Returns:
        Timedelta until next interval
    """
    now = now_utc()
    next_interval = get_next_interval(now, interval)
    return next_interval - now


def seconds_until_next_interval(interval: IntervalType) -> float:
    """Get seconds until next interval boundary."""
    return time_until_next_interval(interval).total_seconds()


def is_interval_boundary(dt: datetime, interval: IntervalType, tolerance_seconds: int = 5) -> bool:
    """
    Check if datetime is at or near an interval boundary.
    
    Args:
        dt: Datetime to check
        interval: Interval type
        tolerance_seconds: Seconds of tolerance around boundary
        
    Returns:
        True if near boundary
    """
    rounded = round_to_interval(dt, interval)
    diff = abs((dt - rounded).total_seconds())
    return diff <= tolerance_seconds


def get_interval_range(
    start: datetime,
    end: datetime,
    interval: IntervalType
) -> list[datetime]:
    """
    Generate list of interval timestamps between start and end.
    
    Args:
        start: Start datetime
        end: End datetime
        interval: Interval type
        
    Returns:
        List of interval timestamps
    """
    seconds = INTERVAL_SECONDS[interval]
    current = round_to_interval(start, interval)
    result = []
    
    while current <= end:
        result.append(current)
        current = current + timedelta(seconds=seconds)
    
    return result


def parse_iso_datetime(s: str) -> datetime:
    """Parse ISO format datetime string to UTC datetime."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def format_iso_datetime(dt: datetime) -> str:
    """Format datetime to ISO string."""
    return dt.isoformat()


def format_human_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def market_hours_since_open() -> float:
    """
    Get hours since midnight UTC (useful for crypto 'market open' concept).
    Returns value between 0.0 and 24.0.
    """
    now = now_utc()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (now - midnight).total_seconds() / 3600


def is_within_trading_hours(
    start_hour: int = 0,
    end_hour: int = 24,
    dt: datetime | None = None
) -> bool:
    """
    Check if current time is within specified trading hours (UTC).
    Default is 24/7 for crypto.
    """
    if dt is None:
        dt = now_utc()
    hour = dt.hour
    
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    else:
        # Wraps around midnight
        return hour >= start_hour or hour < end_hour
