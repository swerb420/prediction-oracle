"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys - Kalshi
    kalshi_api_key: str = Field(default="", alias="KALSHI_API_KEY")
    kalshi_api_secret: str = Field(default="", alias="KALSHI_API_SECRET")
    kalshi_base_url: str = Field(
        default="https://trading-api.kalshi.com", alias="KALSHI_BASE_URL"
    )

    # API Keys - Polymarket
    polymarket_api_key: str = Field(default="", alias="POLYMARKET_API_KEY")
    polymarket_private_key: str = Field(default="", alias="POLYMARKET_PRIVATE_KEY")
    polymarket_clob_url: str = Field(
        default="https://clob.polymarket.com", alias="POLYMARKET_CLOB_URL"
    )

    # API Keys - OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )

    # API Keys - Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL"
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com", alias="ANTHROPIC_BASE_URL"
    )

    # API Keys - xAI
    xai_api_key: str = Field(default="", alias="XAI_API_KEY")
    xai_model: str = Field(default="grok-2-1212", alias="XAI_MODEL")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")

    # Trading Mode
    trading_mode: Literal["research", "paper", "live"] = Field(
        default="research", alias="TRADING_MODE"
    )

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./prediction_oracle.db", alias="DATABASE_URL"
    )

    # Risk Management
    initial_bankroll: float = Field(default=1000.0, alias="INITIAL_BANKROLL")
    max_position_size_pct: float = Field(default=0.01, alias="MAX_POSITION_SIZE_PCT")
    max_venue_exposure_pct: float = Field(default=0.20, alias="MAX_VENUE_EXPOSURE_PCT")
    max_daily_drawdown_pct: float = Field(default=0.05, alias="MAX_DAILY_DRAWDOWN_PCT")

    # Scanning
    scan_interval_seconds: int = Field(default=300, alias="SCAN_INTERVAL_SECONDS")
    max_concurrent_llm_calls: int = Field(default=2, alias="MAX_CONCURRENT_LLM_CALLS")

    # Telegram notifications
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_ids: str = Field(
        default="",
        alias="TELEGRAM_CHAT_IDS",
        description="Comma-separated list of chat IDs to notify",
    )

    # Hugging Face inference (free/open LLM backstop)
    huggingface_api_token: str = Field(default="", alias="HUGGINGFACE_API_TOKEN")
    huggingface_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3", alias="HUGGINGFACE_MODEL"
    )
    huggingface_api_url: str = Field(
        default="https://api-inference.huggingface.co/models", alias="HUGGINGFACE_API_URL"
    )


# Global settings instance
settings = Settings()
