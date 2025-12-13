# Whale Scanner API Setup Guide

This guide covers all the APIs needed to run the Polymarket whale scanner at production quality.

## 🚀 Quick Start (Minimum Required)

For basic whale scanning, you need:
1. **Alchemy API** (free tier) - Polygon blockchain access
2. **Discord or Telegram** - For alerts

---

## 📡 Blockchain Node Providers

### 1. Alchemy (PRIMARY - Required)
**What:** Polygon blockchain WebSocket access for real-time transaction streaming.
**Why:** Sub-200ms latency, reliable, great free tier.
**Cost:** FREE tier = 300M compute units/month (plenty for whale scanning)

**Setup:**
1. Go to https://www.alchemy.com/
2. Create account → Create App → Select "Polygon Mainnet"
3. Copy your API key

```bash
# Add to .env
ALCHEMY_API_KEY=your_alchemy_key_here
```

### 2. QuickNode (BACKUP - Optional)
**What:** Alternative Polygon node provider.
**Why:** Backup if Alchemy has issues, sometimes faster.
**Cost:** FREE tier available

**Setup:**
1. Go to https://www.quicknode.com/
2. Create Polygon Mainnet endpoint
3. Copy WebSocket URL

```bash
# Add to .env
QUICKNODE_API_KEY=your_quicknode_key
QUICKNODE_WS_URL=wss://your-endpoint.polygon-mainnet.quiknode.pro/your-key/
```

### 3. Shyft (SOLANA - Future)
**What:** Solana blockchain access for Hyperliquid/Drift Polymarket.
**Why:** Some Polymarket volume is moving to Solana.
**Cost:** FREE tier = 100K credits/month

**Setup:**
1. Go to https://shyft.to/
2. Create account and get API key

```bash
# Add to .env
SHYFT_API_KEY=your_shyft_key
```

### 4. Helius (SOLANA - Premium)
**What:** Premium Solana node with Geyser (fastest streaming).
**Why:** Sub-100ms Solana transaction detection.
**Cost:** FREE tier = 100K credits/day, paid plans for more

**Setup:**
1. Go to https://www.helius.dev/
2. Create account and get API key

```bash
# Add to .env
HELIUS_API_KEY=your_helius_key
```

---

## 🔔 Alert Channels

### Discord Webhook (Recommended)
**What:** Instant whale alerts to your Discord server.
**Cost:** FREE

**Setup:**
1. Open Discord → Server Settings → Integrations → Webhooks
2. Create Webhook → Copy URL

```bash
# Add to .env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy
```

### Telegram Bot
**What:** Whale alerts via Telegram.
**Cost:** FREE

**Setup:**
1. Message @BotFather on Telegram
2. Send `/newbot` and follow prompts
3. Copy the bot token
4. Create a channel/group and add your bot
5. Get chat ID: https://api.telegram.org/bot<TOKEN>/getUpdates

```bash
# Add to .env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=-1001234567890
```

---

## 🏷️ Wallet Labeling Services (Optional - Premium)

### Arkham Intelligence
**What:** On-chain wallet labels and entity tracking.
**Why:** Know who's behind whale wallets (VCs, funds, etc.)
**Cost:** $100-2000/month depending on tier

**Setup:**
1. Apply at https://www.arkhamintelligence.com/
2. Get API access

```bash
ARKHAM_API_KEY=your_arkham_key
```

### Nansen
**What:** Premium wallet labels and analytics.
**Why:** Most comprehensive wallet database.
**Cost:** $150-2000/month

```bash
NANSEN_API_KEY=your_nansen_key
```

**Alternative (FREE):** Build your own wallet database by:
- Tracking wallets that win consistently
- Using the whale_db.py module to score wallets
- Manually labeling known funds/traders

---

## 💰 Estimated Monthly Costs

### Minimum Setup (Hobbyist) - ~$0/month
- Alchemy FREE tier
- Discord/Telegram FREE
- Self-built wallet labels
- No copy trading

### Recommended Setup (Serious) - ~$100-200/month
- Alchemy Growth ($49/month)
- Helius Free + Shyft Free
- Discord/Telegram
- VPS for 24/7 running ($20-50/month)

### Pro Setup (Competing with Sharps) - ~$500-3000/month
- Alchemy/QuickNode Enterprise ($200-600/month)
- Helius Pro ($100-300/month)
- Arkham/Nansen ($100-2000/month)
- Dedicated server ($100-300/month)

---

## 🔧 Complete .env Template

```bash
# ============================================
# WHALE SCANNER - BLOCKCHAIN NODES
# ============================================

# Alchemy (Polygon) - PRIMARY
# Get at: https://www.alchemy.com/
ALCHEMY_API_KEY=

# QuickNode (Polygon) - BACKUP
# Get at: https://www.quicknode.com/
QUICKNODE_API_KEY=
QUICKNODE_WS_URL=

# Shyft (Solana) - For future Solana Polymarket
# Get at: https://shyft.to/
SHYFT_API_KEY=

# Helius (Solana) - Premium Solana node
# Get at: https://www.helius.dev/
HELIUS_API_KEY=

# ============================================
# WHALE SCANNER - ALERTS
# ============================================

# Discord Webhook
# Create at: Discord > Server Settings > Integrations > Webhooks
DISCORD_WEBHOOK_URL=

# Telegram Bot
# Create at: https://t.me/BotFather
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ============================================
# WHALE SCANNER - FILTERS
# ============================================

# Minimum trade size to alert (USD)
WHALE_MIN_AMOUNT_USD=25000

# Minimum price impact to alert (%)
WHALE_MIN_PRICE_IMPACT=5.0

# Max market volume (smaller markets move more)
WHALE_MAX_MARKET_VOLUME=5000000

# Only alert for known/labeled wallets
WHALE_ONLY_LABELED=false

# Minimum wallet win rate to alert (0.0-1.0)
WHALE_MIN_WIN_RATE=0.0

# ============================================
# COPY TRADING (Advanced - Use with caution!)
# ============================================

# Enable automatic copy trading
ENABLE_COPY_TRADING=false

# Max USD per copy trade
COPY_TRADE_MAX_USD=100

# Daily copy trading limit
COPY_TRADE_DAILY_LIMIT=500

# Use Flashbots/MEV protection
USE_FLASHBOTS=true

# ============================================
# WALLET LABELS (Optional - Premium)
# ============================================

# Arkham Intelligence - https://arkhamintelligence.com
ARKHAM_API_KEY=

# Nansen - https://nansen.ai
NANSEN_API_KEY=
```

---

## 🏃 Running the Whale Scanner

### Start Scanner
```bash
# Activate environment
cd ~/prediction-oracle
source venv/bin/activate

# Run whale scanner
python -m prediction_oracle.signals.whale_scanner
```

### Run as Background Service
```bash
# Create systemd service
sudo tee /etc/systemd/system/whale-scanner.service << EOF
[Unit]
Description=Polymarket Whale Scanner
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/prediction-oracle
ExecStart=/root/prediction-oracle/venv/bin/python -m prediction_oracle.signals.whale_scanner
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable whale-scanner
sudo systemctl start whale-scanner

# Check status
sudo systemctl status whale-scanner
journalctl -u whale-scanner -f
```

---

## 📊 Expected Performance

With proper setup, you should see:

- **Latency:** 200-500ms from block confirmation to alert
- **Coverage:** All trades >$25K on Polymarket
- **Uptime:** 99%+ with proper monitoring
- **Alert Speed:** <1 second to Discord/Telegram

The top Polymarket traders (like PixOnChain) use similar systems to:
1. Catch whale moves within seconds
2. Identify patterns in sharp money
3. Copy trade with small amounts
4. Build edge through faster information

---

## 🔒 Security Notes

1. **Never share API keys** - Keep .env secure
2. **Use small amounts for copy trading** - Start with $10-50
3. **Monitor for MEV attacks** - Use Flashbots
4. **Separate wallets** - Use a dedicated trading wallet
5. **Rate limit awareness** - Don't spam APIs

---

## 🆘 Troubleshooting

### "WebSocket connection failed"
- Check Alchemy API key is correct
- Verify you selected Polygon Mainnet
- Check firewall allows outbound WebSocket

### "No trades detected"
- Ensure contract addresses are current (check Polymarket docs)
- Lower WHALE_MIN_AMOUNT_USD for testing
- Check logs for parsing errors

### "Alert not received"
- Test webhook manually with curl
- Verify Discord webhook URL is complete
- Check Telegram bot has permission to post

### "High latency"
- Upgrade to paid node tier
- Use geographic-closer node
- Reduce other network usage
