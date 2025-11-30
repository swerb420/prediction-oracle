"""
data_collector.py - Collects real-time market and event data from all sources
"""
import requests

def fetch_manifold(market_id):
    url = f'https://api.manifold.markets/v0/market/{market_id}'
    return requests.get(url, timeout=10).json()

def fetch_polymarket(market_id):
    url = f'https://gamma-api.polymarket.com/markets/{market_id}'
    return requests.get(url, timeout=10).json()

def fetch_kalshi(market_id):
    url = f'https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}'
    return requests.get(url, timeout=10).json()

def fetch_espn(league, game_id):
    sport_map = {'nfl': 'football', 'nba': 'basketball', 'nhl': 'hockey', 'mlb': 'baseball'}
    sport = sport_map.get(league.lower())
    if not sport:
        return None
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={game_id}"
    return requests.get(url, timeout=15).json()

# Add more sources as needed
