#!/usr/bin/env python3
"""
Multi-Source Scanner v3 - Short-term trade finder (24h-7d focus)
Sources: Polymarket, Manifold Markets, and crypto price predictions
Tracks LLM performance by category for optimization
"""
import asyncio
import sqlite3
import httpx
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
import os
import sys

sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

from data_collector import fetch_manifold, fetch_polymarket, fetch_kalshi, fetch_espn
from trade_filter_ml import TradeFilter

trade_filter = TradeFilter(model_path=None)  # Set model path if available

# Category detection - expanded with GEOPOLITICS and more
def categorize(question: str) -> str:
    q = question.lower()
    
    # Sports - most time-sensitive
    if any(x in q for x in ['vs', 'nba', 'nfl', 'nhl', 'mlb', 'ufc', 'boxing', 'f1', 'game', 'match', 'win', 'score', 'points', 'touchdown', 'goal', 'super bowl', 'championship']):
        return 'SPORTS'
    
    # Geopolitics - wars, international conflicts, foreign policy
    if any(x in q for x in ['ukraine', 'russia', 'putin', 'zelensky', 'ceasefire', 'war', 'nato', 'israel', 'gaza', 'hamas', 'netanyahu', 
                            'venezuela', 'maduro', 'china', 'taiwan', 'xi jinping', 'north korea', 'kim jong', 'iran', 'military', 
                            'invasion', 'troops', 'missile', 'nuclear', 'sanctions', 'peace deal', 'territory']):
        return 'GEOPOLITICS'
    
    # Crypto - high volatility, good for short-term
    if any(x in q for x in ['bitcoin', 'btc', 'eth', 'ethereum', 'crypto', 'token', 'solana', 'doge', 'xrp', 'price above', 'price below']):
        return 'CRYPTO'
    
    # Economics/Finance - Fed, rates, etc
    if any(x in q for x in ['fed', 'interest rate', 'gdp', 'inflation', 'unemployment', 'fomc', 'cpi', 'jobs report', 'recession', 'rate cut', 'rate hike', 'powell']):
        return 'ECONOMICS'
    
    # Entertainment - media, celebrities
    if any(x in q for x in ['movie', 'film', 'oscar', 'box office', 'grammy', 'emmy', 'album', 'song', 'tv', 'show', 'netflix', 'celebrity', 
                            'elon', 'musk', 'tweet', 'kardashian', 'taylor swift', 'kanye', 'youtube', 'tiktok', 'viral']):
        return 'ENTERTAINMENT'
    
    # Politics - US and world politics
    if any(x in q for x in ['trump', 'biden', 'election', 'president', 'congress', 'senate', 'governor', 'vote', 'poll', 'republican', 
                            'democrat', 'nomination', 'cabinet', 'impeach', 'indictment', 'epstein', 'pardon']):
        return 'POLITICS'
    
    # Stocks/Markets - equities and indices
    if any(x in q for x in ['stock', 's&p', 'nasdaq', 'dow', 'market cap', 'nvidia', 'apple', 'tesla', 'earnings', 'ipo', 
                            'googl', 'msft', 'amzn', 'meta', 'share price', 'market close']):
        return 'STOCKS'
    
    # Weather - natural events
    if any(x in q for x in ['weather', 'hurricane', 'temperature', 'rain', 'snow', 'storm', 'flood', 'tornado', 'wildfire', 'drought']):
        return 'WEATHER'
    
    # AI/Tech - artificial intelligence and tech
    if any(x in q for x in ['ai', 'gpt', 'claude', 'llm', 'openai', 'anthropic', 'google', 'chatbot', 'gemini', 'artificial intelligence', 
                            'machine learning', 'deepmind', 'agi', 'robotics']):
        return 'AI_TECH'
    
    return 'OTHER'


async def fetch_polymarket(max_hours=168):
    """Fetch Polymarket markets closing within max_hours (default 7 days)."""
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Get active markets sorted by volume - INCREASED LIMIT
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true", "limit": 1000}
            )
            if resp.status_code != 200:
                print(f"   Polymarket API error: {resp.status_code}")
                return []
            
            data = resp.json()
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=max_hours)
            
            for m in data:
                end_str = m.get('endDate') or m.get('end_date_iso')
                if not end_str:
                    continue
                
                try:
                    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                except:
                    continue
                
                if end > cutoff or end < now:
                    continue
                
                # Get price
                prices_str = m.get('outcomePrices', '')
                if not prices_str:
                    continue
                
                try:
                    prices = [float(p.strip('" ')) for p in prices_str.strip('[]').split(',') if p.strip()]
                    price = prices[0] if prices else 0.5
                except:
                    continue
                
                if price < 0.03 or price > 0.97:
                    continue
                
                hours_left = (end - now).total_seconds() / 3600
                volume = float(m.get('volume24hr', 0) or 0)
                
                markets.append({
                    'id': m.get('id'),
                    'source': 'POLYMARKET',
                    'question': m.get('question', 'Unknown'),
                    'price': price,
                    'volume': volume,
                    'hours_left': hours_left,
                    'closes_at': end.isoformat(),
                    'category': categorize(m.get('question', '')),
                })
            
        except Exception as e:
            print(f"   Polymarket error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])[:200]


async def fetch_kalshi(max_hours=168):
    """Fetch Kalshi prediction markets - CFTC-regulated US exchange.
    Great for politics, economics, weather, and current events.
    """
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=max_hours)
            
            cursor = None
            for _ in range(5):  # Paginate up to 500 markets
                params = {"limit": 100, "status": "open"}
                if cursor:
                    params["cursor"] = cursor
                
                resp = await client.get(
                    "https://api.elections.kalshi.com/trade-api/v2/markets",
                    params=params
                )
                
                if resp.status_code != 200:
                    print(f"   Kalshi API error: {resp.status_code}")
                    break
                
                data = resp.json()
                
                for m in data.get("markets", []):
                    close_str = m.get("close_time") or m.get("expected_expiration_time")
                    if not close_str:
                        continue
                    
                    try:
                        close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    except:
                        continue
                    
                    if close > cutoff or close < now:
                        continue
                    
                    # Price is 0-100 on Kalshi, convert to 0-1
                    price = m.get("last_price", 50) / 100
                    if price < 0.03 or price > 0.97:
                        continue
                    
                    hours_left = (close - now).total_seconds() / 3600
                    volume = float(m.get("volume_24h", 0) or 0)
                    
                    markets.append({
                        'id': m.get("ticker"),
                        'source': 'KALSHI',
                        'question': m.get("title", "Unknown"),
                        'price': price,
                        'volume': volume,
                        'hours_left': hours_left,
                        'closes_at': close.isoformat(),
                        'category': categorize(m.get("title", "")),
                    })
                
                cursor = data.get("cursor")
                if not cursor:
                    break
            
            print(f"   Kalshi: Found {len(markets)} markets")
            
        except Exception as e:
            print(f"   Kalshi error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])[:200]


async def fetch_manifold(max_hours=168):
    """Fetch Manifold Markets closing within max_hours."""
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Get recently active markets (more likely to be closing soon) - INCREASED LIMIT
            resp = await client.get(
                "https://api.manifold.markets/v0/markets",
                params={"limit": 1000, "sort": "last-bet-time"}
            )
            if resp.status_code != 200:
                print(f"   Manifold API error: {resp.status_code}")
                return []
            
            data = resp.json()
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=max_hours)
            
            for m in data:
                # Only binary markets
                if m.get('outcomeType') != 'BINARY':
                    continue
                
                close_time = m.get('closeTime')
                if not close_time:
                    continue
                
                # closeTime is in milliseconds
                try:
                    close = datetime.fromtimestamp(close_time / 1000, tz=timezone.utc)
                except:
                    continue
                
                if close > cutoff or close < now:
                    continue
                
                price = m.get('probability', 0.5)
                if price < 0.03 or price > 0.97:
                    continue
                
                hours_left = (close - now).total_seconds() / 3600
                volume = float(m.get('volume24Hours', 0) or 0)
                
                markets.append({
                    'id': m.get('id'),
                    'source': 'MANIFOLD',
                    'question': m.get('question', 'Unknown'),
                    'price': price,
                    'volume': volume,
                    'hours_left': hours_left,
                    'closes_at': close.isoformat(),
                    'category': categorize(m.get('question', '')),
                    'url': m.get('url', ''),
                })
            
        except Exception as e:
            print(f"   Manifold error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])[:200]


async def fetch_manifold_search(max_hours=168):
    """Fetch additional Manifold markets using search API for high-volume markets."""
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Search for high-volume open markets
            resp = await client.get(
                "https://api.manifold.markets/v0/search-markets",
                params={
                    "term": "",
                    "sort": "24-hour-vol",
                    "filter": "open",
                    "limit": 500
                }
            )
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=max_hours)
            
            for m in data:
                if m.get('outcomeType') != 'BINARY':
                    continue
                
                close_time = m.get('closeTime')
                if not close_time:
                    continue
                
                try:
                    close = datetime.fromtimestamp(close_time / 1000, tz=timezone.utc)
                except:
                    continue
                
                if close > cutoff or close < now:
                    continue
                
                price = m.get('probability', 0.5)
                if price < 0.03 or price > 0.97:
                    continue
                
                hours_left = (close - now).total_seconds() / 3600
                volume = float(m.get('volume24Hours', 0) or 0)
                
                markets.append({
                    'id': m.get('id'),
                    'source': 'MANIFOLD_HOT',
                    'question': m.get('question', 'Unknown'),
                    'price': price,
                    'volume': volume,
                    'hours_left': hours_left,
                    'closes_at': close.isoformat(),
                    'category': categorize(m.get('question', '')),
                    'url': m.get('url', ''),
                })
            
        except Exception as e:
            print(f"   Manifold search error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])[:100]


async def fetch_polymarket_events(max_hours=168):
    """Fetch markets from Polymarket events API for better coverage."""
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Get active events
            resp = await client.get(
                "https://gamma-api.polymarket.com/events",
                params={"closed": "false", "active": "true", "limit": 100}
            )
            if resp.status_code != 200:
                return []
            
            events = resp.json()
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=max_hours)
            
            for event in events:
                event_markets = event.get('markets', [])
                
                for m in event_markets:
                    end_str = m.get('endDate') or m.get('end_date_iso')
                    if not end_str:
                        continue
                    
                    try:
                        end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    except:
                        continue
                    
                    if end > cutoff or end < now:
                        continue
                    
                    prices_str = m.get('outcomePrices', '')
                    if not prices_str:
                        continue
                    
                    try:
                        prices = [float(p.strip('" ')) for p in prices_str.strip('[]').split(',') if p.strip()]
                        price = prices[0] if prices else 0.5
                    except:
                        continue
                    
                    if price < 0.03 or price > 0.97:
                        continue
                    
                    hours_left = (end - now).total_seconds() / 3600
                    volume = float(m.get('volume24hr', 0) or 0)
                    
                    markets.append({
                        'id': m.get('id'),
                        'source': 'POLY_EVENT',
                        'question': m.get('question', 'Unknown'),
                        'price': price,
                        'volume': volume,
                        'hours_left': hours_left,
                        'closes_at': end.isoformat(),
                        'category': categorize(m.get('question', '')),
                        'event_title': event.get('title', ''),
                    })
            
        except Exception as e:
            print(f"   Polymarket events error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])[:100]


async def fetch_crypto_predictions(max_hours=168):
    """Create crypto price prediction markets based on current prices.
    These resolve based on whether price goes up/down in the timeframe.
    """
    markets = []
    
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            # Get current crypto prices
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": "bitcoin,ethereum,solana,dogecoin,ripple",
                    "vs_currencies": "usd",
                    "include_24hr_change": "true"
                }
            )
            if resp.status_code != 200:
                print(f"   CoinGecko API error: {resp.status_code}")
                return []
            
            data = resp.json()
            now = datetime.now(timezone.utc)
            
            # Create prediction markets for each crypto
            timeframes = [
                (24, "24 hours"),
                (48, "48 hours"),
                (168, "1 week"),
            ]
            
            crypto_names = {
                'bitcoin': ('BTC', 1000),      # Round to nearest $1000
                'ethereum': ('ETH', 100),       # Round to nearest $100
                'solana': ('SOL', 10),          # Round to nearest $10
                'dogecoin': ('DOGE', 0.05),     # Round to nearest $0.05
                'ripple': ('XRP', 0.25),        # Round to nearest $0.25
            }
            
            for crypto_id, (symbol, round_unit) in crypto_names.items():
                if crypto_id not in data:
                    continue
                
                price = data[crypto_id].get('usd', 0)
                change_24h = data[crypto_id].get('usd_24h_change', 0)
                
                if not price or price <= 0:
                    continue
                
                for hours, label in timeframes:
                    if hours > max_hours:
                        continue
                    
                    close = now + timedelta(hours=hours)
                    
                    # Create "Will X be above Y" market
                    # Price threshold is current price rounded to appropriate unit
                    if round_unit >= 1:
                        threshold = round(price / round_unit) * round_unit
                    else:
                        # For small prices, round to nearest round_unit
                        threshold = round(price / round_unit) * round_unit
                    
                    # Make sure threshold is valid and reasonable
                    if threshold <= 0:
                        threshold = round(price, 2)  # Use actual price rounded to cents
                    
                    # Estimate probability based on recent momentum
                    # If up 24h, slightly favor YES; if down, slightly favor NO
                    base_prob = 0.50
                    momentum_adj = min(0.15, max(-0.15, change_24h / 100))
                    prob = base_prob + momentum_adj
                    
                    markets.append({
                        'id': f"CRYPTO-{symbol}-{hours}h-{threshold}",
                        'source': 'CRYPTO_PRED',
                        'question': f"Will {symbol} be above ${threshold:,.2f} in {label}?",
                        'price': prob,
                        'volume': 0,  # Synthetic market
                        'hours_left': hours,
                        'closes_at': close.isoformat(),
                        'category': 'CRYPTO',
                        'current_price': price,
                        'threshold': threshold,
                        'change_24h': change_24h,
                    })
            
        except Exception as e:
            print(f"   Crypto predictions error: {e}")
    
    return markets


async def fetch_espn_sports(max_hours=48):
    """Fetch NBA/NFL/NHL games with betting odds from ESPN API.
    Games resolve within hours - perfect for short-term predictions.
    """
    markets = []
    
    # Sports to fetch
    sports = [
        ("basketball", "nba", "NBA"),
        ("football", "nfl", "NFL"),
        ("hockey", "nhl", "NHL"),
    ]
    
    async with httpx.AsyncClient(timeout=15) as client:
        for sport, league, league_name in sports:
            try:
                resp = await client.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
                )
                if resp.status_code != 200:
                    continue
                
                data = resp.json()
                events = data.get('events', [])
                now = datetime.now(timezone.utc)
                cutoff = now + timedelta(hours=max_hours)
                
                for event in events:
                    # Get game time
                    date_str = event.get('date')
                    if not date_str:
                        continue
                    
                    try:
                        game_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        continue
                    
                    if game_time > cutoff or game_time < now:
                        continue
                    
                    # Get competitors
                    comps = event.get('competitions', [{}])[0].get('competitors', [])
                    if len(comps) < 2:
                        continue
                    
                    home = next((c for c in comps if c.get('homeAway') == 'home'), comps[0])
                    away = next((c for c in comps if c.get('homeAway') == 'away'), comps[1])
                    
                    home_name = home.get('team', {}).get('shortDisplayName', 'Home')
                    away_name = away.get('team', {}).get('shortDisplayName', 'Away')
                    
                    # Get betting odds
                    odds_data = event.get('competitions', [{}])[0].get('odds', [{}])
                    if not odds_data:
                        continue
                    
                    odds = odds_data[0] if odds_data else {}
                    
                    # Get moneyline odds and convert to probability
                    home_odds = odds.get('homeTeamOdds', {})
                    away_odds = odds.get('awayTeamOdds', {})
                    
                    home_ml = home_odds.get('moneyLine', 0)
                    away_ml = away_odds.get('moneyLine', 0)
                    
                    # Convert American odds to probability
                    def ml_to_prob(ml):
                        if not ml:
                            return 0.5
                        if ml > 0:
                            return 100 / (ml + 100)
                        else:
                            return abs(ml) / (abs(ml) + 100)
                    
                    home_prob = ml_to_prob(home_ml)
                    away_prob = ml_to_prob(away_ml)
                    
                    # Normalize probabilities (remove vig)
                    total = home_prob + away_prob
                    if total > 0:
                        home_prob = home_prob / total
                        away_prob = away_prob / total
                    
                    hours_left = (game_time - now).total_seconds() / 3600
                    
                    # Create market for home team win
                    markets.append({
                        'id': f"ESPN-{league}-{event.get('id')}-HOME",
                        'source': f'ESPN_{league_name}',
                        'question': f"Will {home_name} beat {away_name}? ({league_name})",
                        'price': home_prob,
                        'volume': 0,
                        'hours_left': hours_left,
                        'closes_at': game_time.isoformat(),
                        'category': 'SPORTS',
                        'home_team': home_name,
                        'away_team': away_name,
                        'home_record': home.get('records', [{}])[0].get('summary', ''),
                        'away_record': away.get('records', [{}])[0].get('summary', ''),
                        'spread': odds.get('spread', 0),
                        'over_under': odds.get('overUnder', 0),
                    })
                    
            except Exception as e:
                print(f"   ESPN {league_name} error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])


async def fetch_stock_predictions(max_hours=168):
    """Create stock market prediction markets based on current data.
    Uses free Yahoo Finance API for stock data.
    """
    markets = []
    
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            # Major indices and popular stocks
            symbols = {
                '^GSPC': ('S&P 500', 100),      # Round to nearest 100
                '^DJI': ('Dow Jones', 500),     # Round to nearest 500
                '^IXIC': ('NASDAQ', 100),       # Round to nearest 100
                'NVDA': ('NVIDIA', 5),          # Round to nearest $5
                'TSLA': ('Tesla', 10),          # Round to nearest $10
                'AAPL': ('Apple', 5),           # Round to nearest $5
            }
            
            now = datetime.now(timezone.utc)
            
            for symbol, (name, round_unit) in symbols.items():
                try:
                    # Use Yahoo Finance API
                    resp = await client.get(
                        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                        params={"interval": "1d", "range": "5d"},
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    
                    if resp.status_code != 200:
                        continue
                    
                    data = resp.json()
                    result = data.get('chart', {}).get('result', [{}])[0]
                    meta = result.get('meta', {})
                    
                    price = meta.get('regularMarketPrice', 0)
                    prev_close = meta.get('previousClose', price)
                    
                    if not price or price <= 0:
                        continue
                    
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                    
                    # Create predictions for different timeframes
                    timeframes = [(24, "tomorrow"), (120, "this week")]
                    
                    for hours, label in timeframes:
                        if hours > max_hours:
                            continue
                        
                        close = now + timedelta(hours=hours)
                        threshold = round(price / round_unit) * round_unit
                        
                        if threshold <= 0:
                            threshold = round(price, 2)
                        
                        # Base probability with momentum adjustment
                        base_prob = 0.50
                        momentum_adj = min(0.10, max(-0.10, change_pct / 10))
                        prob = base_prob + momentum_adj
                        
                        markets.append({
                            'id': f"STOCK-{symbol}-{hours}h-{threshold}",
                            'source': 'STOCK_PRED',
                            'question': f"Will {name} be above ${threshold:,.0f} {label}?",
                            'price': prob,
                            'volume': 0,
                            'hours_left': hours,
                            'closes_at': close.isoformat(),
                            'category': 'STOCKS',
                            'current_price': price,
                            'threshold': threshold,
                            'change_today': change_pct,
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"   Stock predictions error: {e}")
    
    return markets


async def fetch_weather_predictions(max_hours=168):
    """Create weather prediction markets using free weather API."""
    markets = []
    
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            # Major US cities
            cities = [
                ("New York", 40.7128, -74.0060),
                ("Los Angeles", 34.0522, -118.2437),
                ("Chicago", 41.8781, -87.6298),
                ("Miami", 25.7617, -80.1918),
            ]
            
            now = datetime.now(timezone.utc)
            
            for city, lat, lon in cities:
                try:
                    # Use Open-Meteo free API
                    resp = await client.get(
                        "https://api.open-meteo.com/v1/forecast",
                        params={
                            "latitude": lat,
                            "longitude": lon,
                            "daily": "temperature_2m_max,precipitation_probability_max",
                            "timezone": "auto",
                            "forecast_days": 7
                        }
                    )
                    
                    if resp.status_code != 200:
                        continue
                    
                    data = resp.json()
                    daily = data.get('daily', {})
                    temps = daily.get('temperature_2m_max', [])
                    precip = daily.get('precipitation_probability_max', [])
                    dates = daily.get('time', [])
                    
                    if not temps or not dates:
                        continue
                    
                    # Create predictions for each day
                    for i, (date_str, temp, rain_prob) in enumerate(zip(dates[:3], temps[:3], precip[:3])):
                        try:
                            forecast_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                            hours_left = (forecast_date - now).total_seconds() / 3600
                            
                            if hours_left <= 0 or hours_left > max_hours:
                                continue
                            
                            # Temperature prediction (will it be above X?)
                            temp_f = temp * 9/5 + 32  # Convert to Fahrenheit
                            temp_threshold = round(temp_f / 5) * 5
                            
                            markets.append({
                                'id': f"WEATHER-{city.replace(' ', '')}-TEMP-{i}",
                                'source': 'WEATHER',
                                'question': f"Will {city} high temp exceed {temp_threshold}°F on {date_str}?",
                                'price': 0.50,  # 50/50 at threshold
                                'volume': 0,
                                'hours_left': hours_left,
                                'closes_at': forecast_date.isoformat(),
                                'category': 'WEATHER',
                                'forecast_temp': temp_f,
                                'threshold': temp_threshold,
                            })
                            
                            # Rain prediction
                            if rain_prob and rain_prob > 10:
                                rain_threshold = 50  # Will it rain (>50% chance)?
                                markets.append({
                                    'id': f"WEATHER-{city.replace(' ', '')}-RAIN-{i}",
                                    'source': 'WEATHER',
                                    'question': f"Will there be rain in {city} on {date_str}?",
                                    'price': rain_prob / 100,
                                    'volume': 0,
                                    'hours_left': hours_left,
                                    'closes_at': forecast_date.isoformat(),
                                    'category': 'WEATHER',
                                    'rain_probability': rain_prob,
                                })
                                
                        except:
                            continue
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"   Weather predictions error: {e}")
    
    return markets


async def analyze_with_llm(markets: List[Dict], max_calls: int = 20) -> List[Dict]:
    """Analyze markets with LLMs using efficient batch processing."""
    from prediction_oracle.config import settings
    import sys
    sys.path.insert(0, '/root/prediction_oracle')
    from llm_batch import analyze_markets_batched
    
    has_grok = settings.xai_api_key and settings.xai_api_key != 'your_xai_key'
    has_gpt = settings.openai_api_key and settings.openai_api_key != 'your_openai_key'
    
    if not has_grok and not has_gpt:
        print("   No LLM API keys configured")
        return markets
    
    # Prioritize markets by category diversity and time urgency
    by_cat = {}
    for m in markets:
        cat = m['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(m)
    
    # Pick diverse candidates
    candidates = []
    # First pass: 2 from each category (most urgent)
    for cat, items in by_cat.items():
        sorted_items = sorted(items, key=lambda x: x['hours_left'])
        for item in sorted_items[:2]:
            candidates.append(item)
    
    # Second pass: fill remaining slots with highest volume
    remaining = [m for m in markets if m not in candidates]
    remaining.sort(key=lambda x: x['volume'], reverse=True)
    for m in remaining:
        if len(candidates) >= max_calls:
            break
        if m not in candidates:
            candidates.append(m)
    
    # Use batched analysis (more token efficient)
    try:
        analyzed = await analyze_markets_batched(
            candidates[:max_calls],
            grok_key=settings.xai_api_key if has_grok else None,
            gpt_key=settings.openai_api_key if has_gpt else None,
            batch_size=5,  # 5 markets per API call
            use_grok4=True  # Use grok-4-1-fast-reasoning
        )
        
        for m in analyzed:
            dir_str = m.get('llm_direction', 'SKIP')
            edge_str = f"+{m.get('edge', 0):.1%}" if m.get('edge', 0) > 0 else f"{m.get('edge', 0):.1%}"
            print(f"   ✓ [{m['category'][:5]}] {m['question'][:35]}... → {dir_str} ({edge_str})")
        
        return analyzed
        
    except Exception as e:
        print(f"   Batch analysis error: {e}")
        # Fallback to old method
        pass
    
    # Fallback: individual calls (old method)
    analyzed = []
    
    async with httpx.AsyncClient(timeout=60) as client:
        for m in candidates[:max_calls]:
            try:
                # Compact prompt
                extra_context = ""
                if m['source'] == 'CRYPTO_PRED':
                    extra_context = f"\nCurrent {m['question'].split()[1]} price: ${m.get('current_price', 0):,.2f} (24h change: {m.get('change_24h', 0):+.1f}%)"
                
                prompt = f"""You are a superforecaster. Analyze this prediction market:

Question: {m['question']}
Current YES price: {m['price']:.0%} (this is what the market thinks)
Closes in: {m['hours_left']:.0f} hours
Source: {m['source']}{extra_context}

TASK: Estimate the TRUE probability this resolves YES.
- Consider base rates (how often do events like this happen?)
- Consider recent news and current conditions
- The market price is often efficient but not always

YOU MUST provide a FAIR probability even if uncertain. Use 50% if truly unknown.

Reply EXACTLY in this format (no other text):
FAIR: XX%
DIRECTION: YES or NO (bet YES if FAIR > market price, NO if FAIR < market price)
CONFIDENCE: HIGH/MEDIUM/LOW
REASON: (one sentence, max 15 words)"""

                grok_result = None
                gpt_result = None
                
                # Query Grok
                if has_grok:
                    try:
                        resp = await client.post(
                            "https://api.x.ai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {settings.xai_api_key}"},
                            json={
                                "model": "grok-2-1212",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 100,
                                "temperature": 0.3
                            }
                        )
                        if resp.status_code == 200:
                            grok_result = parse_llm_response(resp.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        print(f"   Grok error: {e}")
                
                # Query GPT
                if has_gpt:
                    try:
                        resp = await client.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                            json={
                                "model": "gpt-4o-mini",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 100,
                                "temperature": 0.3
                            }
                        )
                        if resp.status_code == 200:
                            gpt_result = parse_llm_response(resp.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        print(f"   GPT error: {e}")
                
                # Calculate consensus
                m['grok'] = grok_result
                m['gpt'] = gpt_result
                
                consensus = calculate_consensus(m, grok_result, gpt_result)
                m.update(consensus)
                
                dir_str = m.get('llm_direction', 'SKIP')
                edge_str = f"+{m.get('edge', 0):.1%}" if m.get('edge', 0) > 0 else f"{m.get('edge', 0):.1%}"
                print(f"   ✓ [{m['category'][:5]}] {m['question'][:35]}... → {dir_str} ({edge_str})")
                
                analyzed.append(m)
                
            except Exception as e:
                print(f"   ✗ Error analyzing: {e}")
                m['edge'] = 0
                m['llm_direction'] = 'SKIP'
                analyzed.append(m)
    
    # Add non-analyzed markets
    for m in markets:
        if m not in analyzed:
            m['edge'] = 0
            m['llm_direction'] = 'SKIP'
            analyzed.append(m)
    
    return analyzed


def parse_llm_response(text: str) -> Dict:
    """Parse LLM response into structured data."""
    result = {'fair': None, 'direction': 'SKIP', 'confidence': 'LOW', 'reason': ''}
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('FAIR:'):
            try:
                result['fair'] = float(line.split(':')[1].strip().replace('%', '')) / 100
            except:
                pass
        elif line.startswith('DIRECTION:'):
            result['direction'] = line.split(':')[1].strip().upper()
        elif line.startswith('CONFIDENCE:'):
            result['confidence'] = line.split(':')[1].strip().upper()
        elif line.startswith('REASON:'):
            result['reason'] = line.split(':', 1)[1].strip()
    
    return result


def calculate_consensus(market: Dict, grok: Optional[Dict], gpt: Optional[Dict]) -> Dict:
    """Calculate consensus between LLM predictions."""
    results = [r for r in [grok, gpt] if r and r.get('fair') is not None]
    
    if not results:
        return {'llm_direction': 'SKIP', 'llm_confidence': 'LOW', 'llm_fair': None, 'llm_reason': '', 'edge': 0}
    
    # Average fair value
    fair_values = [r['fair'] for r in results if r.get('fair')]
    avg_fair = sum(fair_values) / len(fair_values) if fair_values else market['price']
    
    # Determine direction based on FAIR vs market price (not LLM direction)
    edge_yes = avg_fair - market['price']
    edge_no = market['price'] - avg_fair
    
    if abs(edge_yes) > 0.03:  # 3% minimum edge threshold
        if edge_yes > 0:
            direction = 'YES'
            edge = edge_yes
        else:
            direction = 'NO'
            edge = edge_no
    else:
        # No meaningful edge, but still pick a side for data gathering
        if edge_yes > 0:
            direction = 'YES'
            edge = edge_yes
        else:
            direction = 'NO'
            edge = edge_no
    
    # Confidence based on agreement and edge size
    directions = [r['direction'] for r in results if r.get('direction') in ['YES', 'NO']]
    if len(directions) >= 2 and len(set(directions)) == 1 and abs(edge) > 0.10:
        confidence = 'HIGH'
    elif abs(edge) > 0.05:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    # Best reason
    reason = ''
    for r in results:
        if r.get('reason'):
            reason = r['reason']
            break
    
    return {
        'llm_direction': direction,
        'llm_confidence': confidence,
        'llm_fair': avg_fair,
        'llm_reason': reason,
        'edge': edge,
    }


def save_to_db(markets: List[Dict]):
    """Save analyzed markets to database."""
    db = sqlite3.connect('master_trades.db')
    c = db.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY,
        scan_time TEXT,
        source TEXT,
        market_id TEXT,
        category TEXT,
        question TEXT,
        price REAL,
        volume REAL,
        hours_left REAL,
        closes_at TEXT,
        llm_fair REAL,
        llm_direction TEXT,
        llm_confidence TEXT,
        llm_reason TEXT,
        edge REAL,
        grok_fair REAL,
        grok_dir TEXT,
        gpt_fair REAL,
        gpt_dir TEXT
    )''')
    
    now = datetime.now(timezone.utc).isoformat()
    
    for m in markets:
        grok = m.get('grok', {}) or {}
        gpt = m.get('gpt', {}) or {}
        
        c.execute('''INSERT INTO scans 
            (scan_time, source, market_id, category, question, price, volume, hours_left, closes_at,
             llm_fair, llm_direction, llm_confidence, llm_reason, edge,
             grok_fair, grok_dir, gpt_fair, gpt_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (now, m.get('source'), m.get('id'), m.get('category'), m.get('question'),
             m.get('price'), m.get('volume'), m.get('hours_left'), m.get('closes_at'),
             m.get('llm_fair'), m.get('llm_direction'), m.get('llm_confidence'), m.get('llm_reason'), m.get('edge'),
             grok.get('fair'), grok.get('direction'), gpt.get('fair'), gpt.get('direction')))
    
    db.commit()
    db.close()


def display_results(markets: List[Dict]):
    """Display analysis results."""
    print("\n" + "="*70)
    print("🎯 SHORT-TERM OPPORTUNITIES (24h - 7 days)")
    print("="*70)
    
    # Group by source and category
    by_source = {}
    by_cat = {}
    
    for m in markets:
        src = m.get('source', 'UNKNOWN')
        cat = m.get('category', 'OTHER')
        
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(m)
        
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(m)
    
    # Actionable trades (edge > 5%)
    actionable = [m for m in markets if m.get('llm_direction') in ['YES', 'NO'] and m.get('edge', 0) > 0.05]
    
    if actionable:
        print("\n🚀 HIGH-EDGE TRADES (>5% edge):")
        print("-"*70)
        
        for m in sorted(actionable, key=lambda x: x.get('edge', 0), reverse=True)[:15]:
            hrs = m['hours_left']
            time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
            src = m.get('source', 'UNK')[:4]
            
            if m['llm_direction'] == 'YES':
                entry = m['price']
            else:
                entry = 1 - m['price']
            
            payout = 5 / entry if entry > 0 else 0
            profit = payout - 5
            
            print(f"\n  [{src}][{m['category'][:6]}] {m['question'][:45]}")
            print(f"  → {m['llm_direction']} @ {entry:.0%} | Edge: +{m['edge']:.1%} | {m['llm_confidence']}")
            print(f"  → $5 → ${payout:.2f} (+${profit:.2f}) | Closes: {time_str}")
            print(f"  → {m.get('llm_reason', 'No reason')[:60]}")
    
    # Summary by source
    print("\n📊 BY SOURCE:")
    print("-"*70)
    for src, items in sorted(by_source.items()):
        actionable_count = len([m for m in items if m.get('llm_direction') in ['YES', 'NO'] and m.get('edge', 0) > 0.03])
        print(f"  {src:12}: {len(items):3} markets, {actionable_count:2} actionable")
    
    # Summary by category
    print("\n📊 BY CATEGORY:")
    print("-"*70)
    for cat, items in sorted(by_cat.items()):
        actionable_count = len([m for m in items if m.get('llm_direction') in ['YES', 'NO'] and m.get('edge', 0) > 0.03])
        avg_edge = sum(m.get('edge', 0) for m in items if m.get('edge', 0) > 0) / max(1, len([m for m in items if m.get('edge', 0) > 0]))
        print(f"  {cat:12}: {len(items):3} markets, {actionable_count:2} actionable, avg edge: {avg_edge:+.1%}")
    
    print("\n" + "="*70)


async def main():
    print("🔍 Multi-Source Scanner v6 - Maximum Coverage + Kalshi")
    print("="*70)
    print("Sources: Kalshi, Polymarket (2), Manifold (2), ESPN Sports, Crypto, Stocks, Weather")
    print("Focus: 24 hours - 30 days")
    print("="*70)
    
    max_hours = 720  # 30 days
    
    # Fetch from all sources in parallel
    print("\n📡 Fetching markets from all sources...")
    
    kalshi_task = fetch_kalshi(max_hours)
    poly_task = fetch_polymarket(max_hours)
    poly_events_task = fetch_polymarket_events(max_hours)
    manifold_task = fetch_manifold(max_hours)
    manifold_hot_task = fetch_manifold_search(max_hours)
    crypto_task = fetch_crypto_predictions(168)  # Crypto only 7 days (too volatile longer)
    espn_task = fetch_espn_sports(72)  # Sports games in next 3 days
    stock_task = fetch_stock_predictions(168)  # Stock predictions 7 days
    weather_task = fetch_weather_predictions(168)  # Weather 7 days
    
    kalshi, poly, poly_events, manifold, manifold_hot, crypto, espn, stocks, weather = await asyncio.gather(
        kalshi_task, poly_task, poly_events_task, manifold_task, manifold_hot_task, 
        crypto_task, espn_task, stock_task, weather_task
    )
    
    print(f"   Kalshi:           {len(kalshi)} markets")
    print(f"   Polymarket:       {len(poly)} markets")
    print(f"   Polymarket Events:{len(poly_events)} markets")
    print(f"   Manifold:         {len(manifold)} markets")
    print(f"   Manifold Hot:     {len(manifold_hot)} markets")
    print(f"   ESPN Sports:      {len(espn)} games")
    print(f"   Crypto Pred:      {len(crypto)} markets")
    print(f"   Stock Pred:       {len(stocks)} markets")
    print(f"   Weather:          {len(weather)} predictions")
    
    # Combine all markets (dedupe by question similarity)
    all_markets = kalshi + poly + poly_events + manifold + manifold_hot + crypto + espn + stocks + weather
    
    # Simple deduplication by question
    seen_questions = set()
    unique_markets = []
    for m in all_markets:
        q_key = m['question'][:50].lower()
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            unique_markets.append(m)
    
    all_markets = sorted(unique_markets, key=lambda x: x['hours_left'])
    
    print(f"   TOTAL (dedupe):   {len(all_markets)} markets")
    
    if not all_markets:
        print("\nNo markets found!")
        return
    
    # Show urgency breakdown
    urgent_24h = len([m for m in all_markets if m['hours_left'] <= 24])
    urgent_48h = len([m for m in all_markets if 24 < m['hours_left'] <= 48])
    urgent_7d = len([m for m in all_markets if 48 < m['hours_left'] <= 168])
    urgent_30d = len([m for m in all_markets if 168 < m['hours_left'] <= 720])
    
    print(f"\n⏰ URGENCY BREAKDOWN:")
    print(f"   < 24 hours:  {urgent_24h} markets")
    print(f"   24-48 hours: {urgent_48h} markets")
    print(f"   2-7 days:    {urgent_7d} markets")
    print(f"   7-30 days:   {urgent_30d} markets")
    
    # Show preview by category
    print("\n📋 PREVIEW BY CATEGORY:")
    by_cat = {}
    for m in all_markets:
        cat = m['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(m)
    
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat][:3]  # Show 3 per category
        print(f"\n  {cat} ({len(by_cat[cat])} total):")
        for m in items:
            hrs = m['hours_left']
            time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
            print(f"    [{m['source'][:4]}] {time_str}: {m['question'][:45]}...")
    
    # LLM analysis - analyze more for better category coverage
    print("\n\n🤖 Running LLM analysis (20 markets max)...")
    analyzed = await analyze_with_llm(all_markets, max_calls=20)
    
    # Save and display
    save_to_db(analyzed)
    display_results(analyzed)
    
    print("\n✅ Scan complete. Results saved to master_trades.db")
    print("   Run this scan regularly to track LLM performance by category!")


if __name__ == "__main__":
    asyncio.run(main())
