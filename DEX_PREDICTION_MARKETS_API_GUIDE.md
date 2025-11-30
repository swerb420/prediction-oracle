# DEX Prediction Markets API Guide
## Complete List of Available APIs for Your Prediction Oracle

**Last Updated:** November 2025  
**Tested From:** VPS Linux Environment

---

## 🟢 WORKING APIs (Confirmed)

### 1. **Polymarket** ✅ REAL MONEY (Polygon L2)
**Status:** FULLY WORKING  
**Money Type:** REAL (USDC on Polygon)  
**Min Bet:** ~$1  
**Categories:** Politics, Crypto, Sports, AI, World Events, Pop Culture  

**API Endpoint:**
```bash
# Get markets
curl 'https://gamma-api.polymarket.com/markets?limit=10&active=true&closed=false'

# Get specific market
curl 'https://gamma-api.polymarket.com/markets?slug=your-market-slug'

# Search by category
curl 'https://gamma-api.polymarket.com/markets?category=Sports'
```

**Key Fields:**
- `question` - Market question
- `outcomePrices` - Current YES/NO prices
- `volume` - Total volume traded
- `endDate` - Resolution date
- `active`, `closed` - Market status

**Docs:** https://docs.polymarket.com/

---

### 2. **Manifold Markets** ✅ PLAY MONEY
**Status:** FULLY WORKING  
**Money Type:** PAPER (Mana currency)  
**Min Bet:** Any amount  
**Categories:** Everything (user-created markets)  

**API Endpoint:**
```bash
# Get markets
curl -L 'https://manifold.markets/api/v0/markets?limit=10'

# Get specific market
curl -L 'https://manifold.markets/api/v0/market/{marketId}'

# Search markets
curl -L 'https://manifold.markets/api/v0/search-markets?term=bitcoin'
```

**Key Fields:**
- `question` - Market question
- `probability` - Current probability (0-1)
- `volume` - Total volume
- `closeTime` - Market close time (Unix ms)
- `isResolved` - Resolution status

**Docs:** https://docs.manifold.markets/api

---

### 3. **Metaculus** ✅ FORECASTING (No Money)
**Status:** WORKING  
**Money Type:** Reputation points only  
**Categories:** Science, AI, Geopolitics, Long-term forecasts  

**API Endpoint:**
```bash
# Get questions
curl 'https://www.metaculus.com/api2/questions/?limit=10'

# Get specific question
curl 'https://www.metaculus.com/api2/questions/{id}/'
```

**Notes:** Great for long-term forecasts. No real betting, but useful for training LLM predictions.

---

### 4. **ESPN Sports** ✅ REAL DATA (Indirect Betting)
**Status:** WORKING  
**Money Type:** N/A (use with sportsbooks)  
**Categories:** NFL, NBA, MLB, NHL, Soccer, MMA  

**API Endpoints:**
```bash
# NFL Scoreboard
curl 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'

# NBA Scoreboard  
curl 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

# Get game result
curl 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={gameId}'
```

**Use Case:** Get real game data for sports predictions, resolve trades against actual results.

---

## 🟡 LIMITED ACCESS APIs

### 5. **Kalshi** ⚠️ REAL MONEY (US Regulated)
**Status:** API MIGRATING  
**Money Type:** REAL (USD)  
**Min Bet:** $1  
**Categories:** Politics, Economics, Weather, Events  

**Notes:**
- API moved to `https://api.elections.kalshi.com/`
- Requires authentication for most endpoints
- US residents only
- No sports betting (CFTC regulated)

**Old Endpoint (deprecated):**
```bash
curl 'https://trading-api.kalshi.com/trade-api/v2/markets'
# Returns: "API has been moved to https://api.elections.kalshi.com/"
```

**Docs:** https://kalshi.com/docs/api

---

### 6. **Azuro Protocol** ⚠️ REAL MONEY (Sports DEX)
**Status:** API REQUIRES GRAPHQL  
**Money Type:** REAL (Various chains: Polygon, Gnosis, Base, Chiliz)  
**Categories:** Sports betting only  

**Working Endpoints (Production):**
```bash
# Polygon GraphQL
POST https://thegraph.onchainfeed.org/subgraphs/name/azuro-protocol/azuro-api-polygon-v3

# Gnosis GraphQL  
POST https://thegraph.onchainfeed.org/subgraphs/name/azuro-protocol/azuro-api-gnosis-v3

# Base GraphQL
POST https://thegraph.onchainfeed.org/subgraphs/name/azuro-protocol/azuro-api-base-v3
```

**Example GraphQL Query:**
```graphql
{
  games(first: 10, where: { status: Created }, orderBy: startsAt) {
    id
    title
    startsAt
    status
    sport { name }
    league { name }
  }
}
```

**Frontend:** https://bookmaker.xyz/  
**Docs:** https://gem.azuro.org/hub/apps/APIs/overview

---

### 7. **Overtime/Thales** ⚠️ REAL MONEY (Sports DEX)
**Status:** NO PUBLIC REST API  
**Money Type:** REAL (Optimism, Arbitrum, Base, Polygon)  
**Categories:** Sports, Digital Options  

**Notes:**
- Built on Thales Protocol
- Uses Chainlink oracles for settlement
- Must interact via smart contracts or SDK
- No simple REST API available

**Frontend:** https://overtimemarkets.xyz/  
**Docs:** https://docs.overtime.io/

---

## 🔴 NOT WORKING / DEPRECATED

### 8. **Gnosis/Omen** ❌ DEPRECATED
**Status:** Documentation returns 404  
**Notes:** Conditional tokens framework still exists but Omen prediction market is deprecated.

### 9. **Augur** ❌ DEPRECATED  
**Status:** V2 shut down, no active markets  

### 10. **Hedgehog Markets** ❌ SOLANA  
**Status:** Solana-based, different API paradigm

---

## 📊 Summary Matrix

| Platform | Money | API Status | Min Bet | Sports | Politics | Crypto | Best For |
|----------|-------|------------|---------|--------|----------|--------|----------|
| **Polymarket** | REAL | ✅ REST | $1 | ✅ | ✅ | ✅ | General prediction |
| **Manifold** | PAPER | ✅ REST | Any | ✅ | ✅ | ✅ | LLM training |
| **Metaculus** | NONE | ✅ REST | N/A | ❌ | ✅ | ❌ | Long-term forecasts |
| **ESPN** | N/A | ✅ REST | N/A | ✅ | ❌ | ❌ | Game data/resolution |
| **Kalshi** | REAL | ⚠️ Auth | $1 | ❌ | ✅ | ❌ | US regulated bets |
| **Azuro** | REAL | ⚠️ GraphQL | ~$1 | ✅ | ❌ | ❌ | Sports DEX |
| **Overtime** | REAL | ❌ SDK only | ~$1 | ✅ | ❌ | ❌ | Sports DEX |

---

## 🎯 Recommended Strategy for Your Oracle

### For Paper Trading (LLM Testing):
1. **Polymarket** - Real odds, paper trade against them
2. **Manifold** - Actual play money trading
3. **ESPN** - Real sports results for validation

### For Real Money ($1 bets):
1. **Polymarket** - Best overall (needs USDC on Polygon)
2. **Azuro** - Sports via bookmaker.xyz frontend
3. **Kalshi** - US regulated events (requires US residency)

### Resolution Sources:
- Polymarket: API provides `closed` and resolution status
- ESPN: Game summary API for sports results
- Manifold: API provides `isResolved` and `resolution`
- Kalshi: API provides settlement info

---

## 🔧 Quick Setup for Your Oracle

Add these to your `multi_scanner.py`:

```python
# Already have:
# - fetch_polymarket()  ✅
# - fetch_manifold()    ✅
# - fetch_espn_games()  ✅
# - fetch_kalshi()      ✅ (limited)

# Consider adding:
def fetch_azuro_games():
    """Fetch sports from Azuro via GraphQL"""
    query = '''
    {
        games(first: 20, where: { status: Created }) {
            id
            title
            startsAt
            sport { name }
        }
    }
    '''
    resp = requests.post(
        'https://thegraph.onchainfeed.org/subgraphs/name/azuro-protocol/azuro-api-polygon-v3',
        json={'query': query}
    )
    return resp.json().get('data', {}).get('games', [])
```

---

## 📝 Notes

- Polymarket and Manifold are your **primary sources** - both have excellent APIs
- ESPN provides **ground truth** for sports resolution
- Azuro requires GraphQL but has **real liquidity for sports**
- Kalshi API is migrating - check their docs for current endpoints
- Most DEX platforms require **web3 wallet integration** for actual betting

**For your LLM prediction oracle, focus on:**
1. Polymarket for real market odds
2. Manifold for volume and user sentiment
3. ESPN for sports data/resolution
4. Metaculus for long-term forecasts
