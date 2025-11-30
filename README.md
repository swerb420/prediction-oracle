# Prediction Oracle

A unified, resource-efficient, LLM-driven prediction market trading framework for Kalshi and Polymarket.

## Features

- **Multi-Platform**: Supports both Kalshi and Polymarket
- **LLM-Powered**: Uses GPT-4, Claude, and Grok for market analysis
- **Multiple Strategies**: Conservative edge harvesting and longshot value hunting
- **Risk Management**: Sophisticated bankroll and exposure management
- **Modes**: Research (logging only), Paper (simulation), Live (real trading)
- **VPS-Optimized**: Runs efficiently on small CPU-only servers

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

## Safety

Always start in `research` mode. Only use `live` mode after extensive paper trading and when legally permitted in your jurisdiction.
