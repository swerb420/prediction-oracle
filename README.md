# Prediction Oracle

A unified, resource-efficient, LLM-driven prediction market trading framework for Kalshi and Polymarket.

## Features

- **Multi-Platform**: Supports both Kalshi and Polymarket
- **LLM-Powered**: Uses GPT-4, Claude, and Grok for market analysis
- **Multiple Strategies**: Conservative edge harvesting and longshot value hunting
- **Risk Management**: Sophisticated bankroll and exposure management
- **Modes**: Research (logging only), Paper (simulation), Live (real trading)
- **VPS-Optimized**: Runs efficiently on small CPU-only servers
- **Enrichment & Monitoring**: Free news/crowd calendars, open LLM fallback, and Telegram alerts

## Quick Start

```bash
# Install
pip install -e .

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run in research mode
oracle run --config config.yaml --mode research

# Run in paper trading mode
oracle run --config config.yaml --mode paper
```

## Architecture

- `markets/`: Kalshi and Polymarket API clients
- `llm/`: Multi-provider LLM engine with batching and caching
- `strategies/`: Conservative and longshot trading strategies
- `risk/`: Bankroll and risk management
- `execution/`: Order routing and execution
- `storage/`: SQLite database for tracking and calibration
- `runner/`: Main event loop and scheduler

## Configuration

Edit `config.yaml` to tune:
- LLM provider groups
- Strategy thresholds and filters
- Risk limits
- Market filters

Key environment variables (see `config.py` for defaults):

- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_IDS` (comma-separated) to push alerts via a BotFather bot.
- `HUGGINGFACE_API_TOKEN`, `HUGGINGFACE_MODEL`, `HUGGINGFACE_API_URL` to enable a free/open LLM backstop.

Optional enrichment helpers (async):

- `extra_data_sources.fetch_gdelt_news` – Headlines and tone from the GDELT Doc API.
- `extra_data_sources.fetch_metaculus_questions` – Crowd medians for active Metaculus questions.
- `extra_data_sources.fetch_hackernews_mentions` – Tech sentiment proxy via HN Algolia search.
- `extra_data_sources.fetch_calendar_events` – Parse public ICS feeds for macro calendars/holidays.
- `risk.evaluate_forecasts` – Brier/log-loss calibration utilities for research-mode backtests.

## Safety

Always start in `research` mode. Only use `live` mode after extensive paper trading and when legally permitted in your jurisdiction.
