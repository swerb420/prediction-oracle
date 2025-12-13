# Polymarket 15M Crypto Trader

**NO FAKE DATA. EVER.**

A smart ML + Grok 4.1 system for paper trading 15-minute crypto price direction bets on Polymarket.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         poly_15m_trader.py                          │
│                        (Main Orchestrator)                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Outcome     │    │   Learning ML   │    │     Grok        │
│   Collector   │    │   Predictor     │    │   Provider      │
│               │    │                 │    │                 │
│ Scrapes real  │    │ Trains on real  │    │ Smart triggers  │
│ market data   │    │ outcomes only   │    │ Rate limited    │
└───────┬───────┘    └────────┬────────┘    └────────┬────────┘
        │                     │                      │
        ▼                     ▼                      ▼
┌───────────────────────────────────────────────────────────────────┐
│                       real_data_store.py                          │
│                     (SQLite Database)                             │
│                                                                   │
│  • market_snapshots - Real-time price data                        │
│  • market_outcomes - Actual UP/DOWN after resolution              │
│  • predictions - ML predictions for evaluation                    │
│  • paper_trades - All trades with PnL                            │
│  • grok_calls - Grok API call log                                │
└───────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. NO SYNTHETIC DATA
The system never generates fake training data. All ML training comes from:
- Real market snapshots
- Real market outcomes
- Real price movements

### 2. Conservative Cold Start
When the system has no training data:
- ML returns 50% confidence
- Selective bettor rejects all trades
- System collects data without risking capital

### 3. Learning Over Time
As real outcomes are collected:
- Models are retrained on labeled data
- Confidence calibration improves
- Betting becomes more selective

### 4. Selective Betting
Not every opportunity is traded. Filters include:
- **Data Quality**: Minimum 50 training examples
- **Confidence**: Minimum 60% ML confidence
- **Market Conditions**: Minimum liquidity, max spread
- **Rate Limiting**: Max 4 bets per hour
- **Risk Management**: Max 25% position size

### 5. Smart Grok Triggers
Grok API is only called when valuable:
- High-value opportunities (>70% confidence)
- ML uncertain (55-65% confidence)
- ML disagrees with market
- High liquidity markets
- Volatility spikes

## Usage

### Collect Training Data
```bash
# Collect one cycle
python poly_15m_trader.py --collect

# Collect 10 cycles, 2 minutes apart
python poly_15m_trader.py --collect --cycles 10 --interval 120
```

### View Predictions (No Trading)
```bash
python poly_15m_trader.py --predict
```

### Paper Trade
```bash
# With Grok validation
python poly_15m_trader.py --trade

# Without Grok (faster, cheaper)
python poly_15m_trader.py --trade --no-grok

# Multiple cycles
python poly_15m_trader.py --trade --cycles 10 --interval 300
```

### Monitor Markets
```bash
python poly_15m_trader.py --monitor --interval 30
```

### Check Status
```bash
python poly_15m_trader.py --status
```

## Collecting Training Data

To build up a training dataset, run the collector over time:

```bash
# Run collection every 2 minutes for a few hours
python poly_15m_trader.py --collect --cycles 100 --interval 120
```

The system needs:
- **10+ examples**: Basic predictions start
- **50+ examples**: Betting enabled
- **200+ examples**: Full model with calibration

## Logs

All activity is logged to:
- **Console**: Human-readable with emoji markers
- **JSON files**: `./logs/trading_YYYY-MM-DD.jsonl`
- **SQLite**: Persistent storage in `./data/polymarket_real.db`

## Files

| File | Purpose |
|------|---------|
| `poly_15m_trader.py` | Main runner script |
| `real_data_store.py` | SQLite storage for all data |
| `outcome_collector.py` | Fetches real market data from Polymarket |
| `learning_ml_predictor.py` | ML model that trains on real outcomes |
| `selective_bettor.py` | Applies filters, decides whether to bet |
| `grok_provider.py` | Smart Grok API integration |
| `paper_trading_engine.py` | Position tracking and PnL |
| `trading_logger.py` | Comprehensive structured logging |

## Requirements

- Python 3.12+
- httpx
- scikit-learn
- numpy

## Environment Variables

- `XAI_API_KEY`: Your xAI API key for Grok (defaults to provided key)

## Symbols

Currently supports:
- **BTC** - Bitcoin
- **ETH** - Ethereum  
- **SOL** - Solana
- **XRP** - Ripple

## How It Works

1. **Discovery**: Scrapes `polymarket.com/crypto/15M` to find current 15-minute markets
2. **Data Collection**: Fetches YES/NO prices, volume, liquidity from event pages
3. **Prediction**: ML model predicts direction based on market features
4. **Filtering**: Selective bettor applies confidence, data quality, risk filters
5. **Grok Validation**: Optionally calls Grok for high-value opportunities
6. **Trading**: Opens paper positions on approved bets
7. **Resolution**: Tracks outcomes, calculates PnL, updates training data

## Performance Tracking

The system tracks:
- Win rate
- PnL
- Prediction accuracy
- Confidence calibration
- Grok agreement rate
- ROI

View with:
```bash
python poly_15m_trader.py --status
```

## Optimization

To optimize the system:
1. Collect more training data
2. Analyze logs: `./logs/trading_*.jsonl`
3. Query the database: `./data/polymarket_real.db`
4. Adjust filters in `selective_bettor.py`
5. Tune ML hyperparameters in `learning_ml_predictor.py`
