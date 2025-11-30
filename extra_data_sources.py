"""Supplementary data sources for enrichment and research.

This module adds lightweight, free-to-use data fetchers that can be used to
augment strategy signals with context from:
- GDELT news headlines
- Metaculus crowd forecasts
- Macroeconomic calendar feeds published via ICS
- Hacker News search (as a proxy for tech sentiment)

All functions are async and rely on httpx, matching the rest of the codebase's
non-blocking design.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

UserDict = dict[str, Any]


async def fetch_gdelt_news(
    query: str,
    max_records: int = 25,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[UserDict]:
    """Fetch headline-level news from the free GDELT Doc API.

    Args:
        query: Free-text query string (e.g., "election OR referendum").
        max_records: Maximum articles to return (GDELT caps at 250).
        client: Optional shared httpx.AsyncClient.

    Returns:
        A list of normalized article dictionaries with title, url, source, and
        publication date keys.
    """

    params = {
        "format": "json",
        "maxrecords": max_records,
        "query": query,
        "sort": "datedesc",
        "mode": "ArtList",
    }
    close_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)

    try:
        response = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
        response.raise_for_status()
        data = response.json()
    finally:
        if close_client:
            await client.aclose()

    results: list[UserDict] = []
    for article in data.get("articles", []):
        results.append(
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source"),
                "published": article.get("seendate"),
                "tone": article.get("tone"),
            }
        )
    return results


async def fetch_metaculus_questions(
    search: str,
    limit: int = 20,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[UserDict]:
    """Fetch community questions and median forecasts from Metaculus.

    Args:
        search: Search phrase (e.g., "Trump", "BTC", "inflation").
        limit: Maximum questions to return.
        client: Optional shared httpx.AsyncClient.

    Returns:
        A list of question dictionaries containing the title, community median
        probability, and URL.
    """

    params = {
        "search": search,
        "limit": limit,
        "order_by": "-activity",
        "status": "open",
    }
    close_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)

    try:
        response = await client.get("https://www.metaculus.com/api2/questions/", params=params)
        response.raise_for_status()
        data = response.json()
    finally:
        if close_client:
            await client.aclose()

    results: list[UserDict] = []
    for item in data.get("results", []):
        prediction = item.get("community_prediction") or {}
        results.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "url": f"https://www.metaculus.com/questions/{item.get('id')}/",
                "median": prediction.get("q2"),
                "crowd": prediction,
                "close_time": item.get("close_time") or item.get("close_time_utc"),
            }
        )
    return results


async def fetch_hackernews_mentions(
    query: str,
    limit: int = 20,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[UserDict]:
    """Query the free Hacker News Algolia API for sentiment and chatter.

    Args:
        query: Search phrase.
        limit: Maximum stories to return (Algolia caps at 1000 per query).
        client: Optional shared httpx.AsyncClient.

    Returns:
        List of dictionaries containing title, url, points, and created_at.
    """

    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": limit,
    }
    close_client = client is None
    client = client or httpx.AsyncClient(timeout=15.0)

    try:
        response = await client.get("https://hn.algolia.com/api/v1/search", params=params)
        response.raise_for_status()
        data = response.json()
    finally:
        if close_client:
            await client.aclose()

    results: list[UserDict] = []
    for hit in data.get("hits", []):
        results.append(
            {
                "title": hit.get("title"),
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                "points": hit.get("points", 0),
                "created_at": hit.get("created_at"),
            }
        )
    return results


def _parse_ics_events(ics_text: str) -> list[UserDict]:
    """Parse a minimal subset of ICS format into dictionaries."""

    events: list[UserDict] = []
    blocks = ics_text.split("BEGIN:VEVENT")
    for block in blocks[1:]:
        event: UserDict = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("SUMMARY:"):
                event["summary"] = line.removeprefix("SUMMARY:")
            elif line.startswith("DTSTART"):
                value = line.split(":", 1)[-1]
                try:
                    event["start"] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    event["start_raw"] = value
            elif line.startswith("DTEND"):
                value = line.split(":", 1)[-1]
                try:
                    event["end"] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    event["end_raw"] = value
            elif line.startswith("DESCRIPTION:"):
                event["description"] = line.removeprefix("DESCRIPTION:")
            elif line.startswith("URL:"):
                event["url"] = line.removeprefix("URL:")
        if event:
            events.append(event)
    return events


async def fetch_calendar_events(
    ics_url: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[UserDict]:
    """Pull macro calendars or holiday events from a public ICS feed.

    ICS feeds can be produced by FRED calendars, public holiday calendars, or
    community event trackers. This parser is intentionally minimal but robust
    enough for enrichment and tagging.
    """

    close_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)

    try:
        response = await client.get(ics_url)
        response.raise_for_status()
        ics_text = response.text
    finally:
        if close_client:
            await client.aclose()

    return _parse_ics_events(ics_text)
