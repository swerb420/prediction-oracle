"""
Centralized configuration with strict validation.
All config from environment variables via pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with strict validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # === XAI / Grok 4.1 ===
    xai_api_key: str = Field(default="", description="xAI API key for Grok")
    xai_model: str = Field(default="grok-3-fast", description="Grok model to use")
    xai_base_url: str = Field(default="https://api.x.ai/v1", description="xAI API base URL")
    
    # === Polymarket ===
    polymarket_api_key: str = Field(default="", description="Polymarket API key")
    polymarket_secret: str = Field(default="", description="Polymarket secret")
    polymarket_passphrase: str = Field(default="", description="Polymarket passphrase")
    polymarket_gamma_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Polymarket Gamma API URL"
    )
    polymarket_clob_url: str = Field(
        default="https://clob.polymarket.com",
        description="Polymarket CLOB URL"
    )
    polymarket_data_url: str = Field(
        default="https://data-api.polymarket.com",
        description="Polymarket Data API URL"
    )
    
    # === CEX (Binance) ===
    binance_api_key: str = Field(default="", description="Binance API key")
    binance_secret: str = Field(default="", description="Binance secret")
    binance_base_url: str = Field(
        default="https://api.binance.com",
        description="Binance REST API URL"
    )
    binance_ws_url: str = Field(
        default="wss://stream.binance.com:9443/ws",
        description="Binance WebSocket URL"
    )
    
    # === Storage ===
    sqlite_db_path: Path = Field(
        default=Path("./data/edge_research.db"),
        description="SQLite database path"
    )
    parquet_data_dir: Path = Field(
        default=Path("./data/parquet"),
        description="Parquet data directory"
    )
    
    # === Trading Config ===
    paper_trading: bool = Field(default=True, description="Enable paper trading mode")
    initial_capital: float = Field(default=10000.0, ge=100.0, description="Initial capital")
    max_position_size_pct: float = Field(
        default=0.10, ge=0.01, le=0.5,
        description="Max position size as fraction of capital"
    )
    kelly_fraction: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Kelly criterion fraction"
    )
    min_edge_threshold: float = Field(
        default=0.02, ge=0.0, le=0.5,
        description="Minimum edge to trade"
    )
    max_concurrent_positions: int = Field(
        default=10, ge=1, le=100,
        description="Maximum concurrent positions"
    )
    
    # === ML Config ===
    model_dir: Path = Field(default=Path("./models"), description="Model storage directory")
    retrain_hour_utc: int = Field(default=4, ge=0, le=23, description="Hour to run daily retrain")
    min_sharpe_improvement: float = Field(
        default=0.3, ge=0.0,
        description="Minimum Sharpe improvement to promote model"
    )
    lookback_days: int = Field(default=60, ge=7, le=365, description="Training lookback period")
    direction_threshold_pct: float = Field(
        default=0.08, ge=0.0, le=1.0,
        description="Threshold for UP/DOWN classification (%)"
    )
    
    # === Grok Config ===
    grok_call_interval_minutes: int = Field(
        default=15, ge=5, le=60,
        description="Minimum interval between Grok calls"
    )
    grok_max_input_tokens: int = Field(
        default=1200, ge=100, le=4000,
        description="Max input tokens for Grok"
    )
    grok_cache_ttl_seconds: int = Field(
        default=900, ge=60, le=3600,
        description="Grok response cache TTL"
    )
    
    # === Risk Limits ===
    max_daily_loss_pct: float = Field(
        default=0.05, ge=0.01, le=0.5,
        description="Max daily loss before stopping"
    )
    max_drawdown_pct: float = Field(
        default=0.15, ge=0.05, le=0.5,
        description="Max drawdown before system halt"
    )
    position_correlation_limit: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Max correlation between positions"
    )
    
    # === Logging ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Path = Field(
        default=Path("./logs/edge_research.log"),
        description="Log file path"
    )
    
    @field_validator("sqlite_db_path", "parquet_data_dir", "model_dir", "log_file", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def has_xai_key(self) -> bool:
        """Check if xAI API key is configured."""
        return bool(self.xai_api_key and self.xai_api_key != "your_xai_api_key_here")
    
    @property
    def has_polymarket_keys(self) -> bool:
        """Check if Polymarket keys are configured."""
        return bool(
            self.polymarket_api_key and 
            self.polymarket_api_key != "your_polymarket_api_key_here"
        )
    
    @property
    def has_binance_keys(self) -> bool:
        """Check if Binance keys are configured."""
        return bool(
            self.binance_api_key and 
            self.binance_api_key != "your_binance_api_key_here"
        )
    
    @property
    def direction_threshold_decimal(self) -> float:
        """Get direction threshold as decimal (0.08% -> 0.0008)."""
        return self.direction_threshold_pct / 100.0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
