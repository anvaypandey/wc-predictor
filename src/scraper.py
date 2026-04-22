"""
Fetches current FIFA world rankings from the official FIFA API.
Used at inference time to enrich predictions with up-to-date ranking features.
"""

import json
import time
import requests
from pathlib import Path

CACHE_FILE = Path("data/fifa_rankings.json")
CACHE_TTL  = 86400  # 24 hours

FIFA_RANKINGS_URL = "https://api.fifa.com/api/v3/rankings?gender=male&locale=en"

# FIFA official name → match results dataset name (only where they differ)
FIFA_TO_DATASET: dict[str, str] = {
    "Korea Republic":      "South Korea",
    "IR Iran":             "Iran",
    "USA":                 "United States",
    "Côte d'Ivoire":       "Ivory Coast",
    "Türkiye":             "Turkey",
    "Congo DR":            "DR Congo",
    "Bosnia-Herzegovina":  "Bosnia and Herzegovina",
    "Czechia":             "Czech Republic",
}

_HEADERS = {
    "Accept":     "application/json",
    "Referer":    "https://inside.fifa.com/",
    "Origin":     "https://inside.fifa.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0.0.0 Safari/537.36",
}


def _team_name(entry: dict) -> str:
    names = entry.get("TeamName", [])
    name = next((n["Description"] for n in names if n.get("Locale", "").startswith("en")),
                names[0]["Description"] if names else "")
    return FIFA_TO_DATASET.get(name, name)


def _parse_rankings(data: dict | list) -> dict[str, dict]:
    entries = data if isinstance(data, list) else data.get("Results", [])
    rankings = {}
    for entry in entries:
        team = _team_name(entry)
        rank = entry.get("Rank", 0)
        pts  = entry.get("DecimalTotalPoints", entry.get("TotalPoints", 0))
        if team and rank:
            rankings[team] = {"rank": int(rank), "points": float(pts)}
    return rankings


def fetch_fifa_rankings(force: bool = False) -> dict[str, dict]:
    """
    Returns {team: {rank, points}} for all ranked nations.
    Reads from cache if fresh (< 24 h); fetches from api.fifa.com otherwise.
    """
    if not force and CACHE_FILE.exists():
        if time.time() - CACHE_FILE.stat().st_mtime < CACHE_TTL:
            with open(CACHE_FILE) as f:
                return json.load(f)

    print("Fetching FIFA rankings from official FIFA API ...")
    try:
        resp = requests.get(FIFA_RANKINGS_URL, timeout=15, headers=_HEADERS)
        resp.raise_for_status()
        rankings = _parse_rankings(resp.json())
        if not rankings:
            raise ValueError("Empty rankings — API response structure may have changed")
        CACHE_FILE.parent.mkdir(exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(rankings, f, indent=2)
        print(f"  Cached {len(rankings)} teams to {CACHE_FILE}")
        return rankings
    except Exception as e:
        print(f"  Warning: could not fetch FIFA rankings ({e})")
        if CACHE_FILE.exists():
            print("  Falling back to cached rankings.")
            with open(CACHE_FILE) as f:
                return json.load(f)
        return {}
