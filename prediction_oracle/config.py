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

    # ============================================
    # FREE LLM Providers
    # ============================================
    
    # Groq - FREE! Very fast Llama inference
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    
    # Google Gemini - FREE tier
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-2.0-flash", alias="GOOGLE_MODEL")
    
    # Together.ai - $25 free credits
    together_api_key: str = Field(default="", alias="TOGETHER_API_KEY")
    together_model: str = Field(default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", alias="TOGETHER_MODEL")
    together_base_url: str = Field(default="https://api.together.xyz/v1", alias="TOGETHER_BASE_URL")
    
    # OpenRouter - has free models
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="meta-llama/llama-3.1-8b-instruct:free", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    
    # HuggingFace - FREE inference
    huggingface_api_key: str = Field(default="", alias="HUGGINGFACE_API_KEY")
    huggingface_model: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct", alias="HUGGINGFACE_MODEL")

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

    # === Signal Source Settings ===
    enable_news_signals: bool = Field(default=True, alias="ENABLE_NEWS_SIGNALS")
    enable_smart_money_signals: bool = Field(default=True, alias="ENABLE_SMART_MONEY_SIGNALS")
    enable_social_signals: bool = Field(default=True, alias="ENABLE_SOCIAL_SIGNALS")
    
    # Free API Keys (optional, improves news coverage)
    newsapi_key: str = Field(default="", alias="NEWSAPI_KEY")
    gnews_key: str = Field(default="", alias="GNEWS_KEY")
    
    # === Enhanced Strategy Settings ===
    smart_money_min_signal: float = Field(default=0.2, alias="SMART_MONEY_MIN_SIGNAL")
    smart_money_weight: float = Field(default=0.3, alias="SMART_MONEY_WEIGHT")
    news_velocity_spike: float = Field(default=2.0, alias="NEWS_VELOCITY_SPIKE")
    news_sentiment_threshold: float = Field(default=0.3, alias="NEWS_SENTIMENT_THRESHOLD")
    enable_quick_filter: bool = Field(default=True, alias="ENABLE_QUICK_FILTER")
    quick_filter_top_n: int = Field(default=20, alias="QUICK_FILTER_TOP_N")
    enable_enhanced_strategies: bool = Field(default=True, alias="ENABLE_ENHANCED_STRATEGIES")
    
    # === Cost Optimization ===
    llm_daily_budget: float = Field(default=10.0, alias="LLM_DAILY_BUDGET")
    max_cost_per_query: float = Field(default=0.50, alias="MAX_COST_PER_QUERY")
    
    # === Smart Screener ===
    use_smart_screener: bool = Field(default=True, alias="USE_SMART_SCREENER")
    screener_min_volume: float = Field(default=100.0, alias="SCREENER_MIN_VOLUME")
    screener_top_n: int = Field(default=20, alias="SCREENER_TOP_N")
    
    # ============================================
    # WHALE SCANNER - Real-time blockchain monitoring
    # ============================================
    
    # Alchemy (Polygon) - Primary node provider for WebSocket streaming
    # Get key at: https://www.alchemy.com/ (free tier: 300M compute units/month)
    alchemy_api_key: str = Field(default="", alias="ALCHEMY_API_KEY")
    alchemy_ws_url: str = Field(
        default="wss://polygon-mainnet.g.alchemy.com/v2/{api_key}",
        alias="ALCHEMY_WS_URL"
    )
    alchemy_http_url: str = Field(
        default="https://polygon-mainnet.g.alchemy.com/v2/{api_key}",
        alias="ALCHEMY_HTTP_URL"
    )
    
    # QuickNode (Polygon) - Backup/alternative node provider
    # Get key at: https://www.quicknode.com/ (free tier available)
    quicknode_api_key: str = Field(default="", alias="QUICKNODE_API_KEY")
    quicknode_ws_url: str = Field(default="", alias="QUICKNODE_WS_URL")
    
    # Shyft (Solana) - For future Hyperliquid/Drift Polymarket support
    # Get key at: https://shyft.to/ (free tier: 100K credits/month)
    shyft_api_key: str = Field(default="", alias="SHYFT_API_KEY")
    
    # Helius (Solana) - Premium Solana node with Geyser
    # Get key at: https://www.helius.dev/ (free tier: 100K credits/day)
    helius_api_key: str = Field(default="", alias="HELIUS_API_KEY")
    helius_rpc_url: str = Field(
        default="https://mainnet.helius-rpc.com/?api-key={api_key}",
        alias="HELIUS_RPC_URL"
    )
    
    # ============================================
    # WHALE SCANNER - Alerting
    # ============================================
    
    # Discord Webhook for whale alerts
    # Create at: Discord Server Settings > Integrations > Webhooks
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")
    
    # Telegram Bot for whale alerts
    # Create at: https://t.me/BotFather
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    
    # ============================================
    # WHALE SCANNER - Filters
    # ============================================
    
    # Minimum trade size to trigger alert (in USD)
    whale_min_amount_usd: float = Field(default=25000.0, alias="WHALE_MIN_AMOUNT_USD")
    
    # Minimum price impact percentage to alert
    whale_min_price_impact: float = Field(default=5.0, alias="WHALE_MIN_PRICE_IMPACT")
    
    # Only alert for markets with volume under this (smaller = easier to move)
    whale_max_market_volume: float = Field(default=5000000.0, alias="WHALE_MAX_MARKET_VOLUME")
    
    # Only alert for labeled/known wallets
    whale_only_labeled: bool = Field(default=False, alias="WHALE_ONLY_LABELED")
    
    # Minimum win rate for wallet to trigger alert
    whale_min_win_rate: float = Field(default=0.0, alias="WHALE_MIN_WIN_RATE")
    
    # ============================================
    # COPY TRADING (Advanced - Use with caution!)
    # ============================================
    
    # Enable automatic copy trading of whale moves
    enable_copy_trading: bool = Field(default=False, alias="ENABLE_COPY_TRADING")
    
    # Maximum USD to spend per copy trade
    copy_trade_max_usd: float = Field(default=100.0, alias="COPY_TRADE_MAX_USD")
    
    # Daily limit for copy trading
    copy_trade_daily_limit: float = Field(default=500.0, alias="COPY_TRADE_DAILY_LIMIT")
    
    # Use Flashbots/MEV protection for copy trades
    use_flashbots: bool = Field(default=True, alias="USE_FLASHBOTS")
    
    # ============================================
    # WALLET LABELS (Nansen/Arkham alternatives)
    # ============================================
    
    # Arkham Intelligence API (optional, $1-2k/month)
    arkham_api_key: str = Field(default="", alias="ARKHAM_API_KEY")
    
    # Nansen API (optional, expensive)
    nansen_api_key: str = Field(default="", alias="NANSEN_API_KEY")


# Global settings instance
settings = Settings()
