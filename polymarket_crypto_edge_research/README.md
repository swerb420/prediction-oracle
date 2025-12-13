# Polymarket Crypto Edge Research

**Production-grade, self-improving ML + Grok 4.1 research stack for:**
1. 15-minute directional predictions on BTC/ETH/SOL
2. Last-seconds Polymarket scalping with microstructure edge
3. Cross-market arbitrage detection

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ CEX Client   │ │ Polymarket   │ │ Storage (SQLite+Parquet) │ │
│  │ (Binance WS) │ │ Gamma/CLOB   │ │                          │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FEATURE LAYER                              │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────┐ │
│  │ Underlyings  │ │ Microstructure│ │Cross-Market│ │ Grok 4.1  │ │
│  │ (TA/Vol/Mom) │ │ (Book/Flow)  │ │ (Arb/Corr) │ │ (Regime)  │ │
│  └──────────────┘ └──────────────┘ └────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ML LAYER                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ LightGBM     │ │ Ridge/Lasso  │ │ LSTM Sequence Models     │ │
│  │ Ensemble     │ │ Linear       │ │                          │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│                         │                                        │
│                   ┌─────┴─────┐                                  │
│                   │ Calibration│                                 │
│                   │ (Isotonic) │                                 │
│                   └───────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STRATEGY LAYER                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ 15m Direction│ │ Last-Seconds │ │ Cross-Market Arb         │ │
│  │ Policy       │ │ Scalper      │ │ (Sum≠1, Semantic Match)  │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
│                         │                                        │
│                   ┌─────┴─────┐                                  │
│                   │Risk Manager│                                 │
│                   │(Kelly+Corr)│                                 │
│                   └───────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EXECUTION LAYER                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Paper Trader │ │ Backtester   │ │ Live Orchestrator        │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# 1. Clone and setup
cd polymarket_crypto_edge_research
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp config.example.env .env
# Edit .env with your API keys

# 3. Sanity check
python -m examples.sanity_check_ingestion

# 4. Run paper trading
python -m scripts.run_live_paper

# 5. Daily retrain (run via cron at 4 AM UTC)
python -m scripts.daily_retrain
```

## 📊 Key Features

### 15-Minute Direction Predictions
- **Features**: 40+ technical indicators, volatility metrics, funding rates
- **Models**: LightGBM ensemble + Ridge + LSTM
- **Labels**: Ternary (UP/FLAT/DOWN) with 0.08% threshold
- **Grok Enhancement**: Regime detection (risk-on/risk-off/choppy)

### Last-Seconds Polymarket Scalper
- **Horizon**: Markets resolving in <60 minutes
- **Edge**: ML probability vs implied probability
- **Execution**: Simulate realistic fills from orderbook depth
- **Risk**: Kelly-capped sizing with correlation clustering

### Cross-Market Arbitrage
- **Sum≠1 Detection**: Find mutually exclusive outcomes not summing to 1.00
- **Semantic Matching**: Grok-powered clustering of equivalent markets
- **Execution**: Simultaneous paper orders on arbitrage legs

### Grok 4.1 Integration
- **Usage**: Regime detection, sentiment scoring, semantic clustering
- **Efficiency**: Batched calls every 15-30 min, <1200 input tokens
- **Output**: Strict JSON with fallback parsing

## 📁 Directory Structure

```
polymarket_crypto_edge_research/
├── core/           # Config, logging, time utilities
├── data/           # API clients, storage, rate limiting
├── features/       # Feature engineering for all data sources
├── ml/             # Models, training, calibration, registry
├── llm/            # Grok 4.1 client and regime classifier
├── strategy/       # Trading policies and risk management
├── exec/           # Paper trading, backtesting, orchestration
├── reports/        # Metrics and visualization
├── scripts/        # Daily retrain, live paper trading
└── examples/       # Sanity check scripts
```

## ⚙️ Configuration

All config via `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_API_KEY` | xAI API key for Grok 4.1 | Required |
| `PAPER_TRADING` | Enable paper trading mode | `true` |
| `INITIAL_CAPITAL` | Starting paper capital | `10000.0` |
| `KELLY_FRACTION` | Kelly criterion fraction | `0.25` |
| `MIN_EDGE_THRESHOLD` | Minimum edge to trade | `0.02` |
| `DIRECTION_THRESHOLD_PCT` | UP/DOWN classification threshold | `0.08` |

## 📈 Performance Metrics

The system tracks:
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: % of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown
- **Hit Rate by Strategy**: Per-strategy performance

## 🔄 Daily Retraining

The `daily_retrain.py` script:
1. Trains 4-6 candidate models on latest data
2. Runs full backtest on hold-out period
3. Promotes to champion only if:
   - Sharpe > current champion + 0.3
   - Max DD not worse than champion

## 🛡️ Risk Management

- **Position Sizing**: Kelly criterion with configurable fraction
- **Correlation Limits**: Max correlation between positions
- **Daily Loss Limit**: Auto-stop at configurable daily loss
- **Max Drawdown**: System halt if drawdown exceeds limit

## 📝 License

MIT License - See LICENSE file for details.
