"""
Fetches current FIFA world rankings and caches them locally.
Used at inference time to enrich predictions with up-to-date ranking features.
"""

import json
import time
import requests
from pathlib import Path

CACHE_FILE = Path("data/fifa_rankings.json")
CACHE_TTL  = 86400  # refresh after 24 hours

# ESPN unofficial API — no key required
ESPN_RANKINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/soccer/"
    "fifa.world/standings?season=2026"
)

# Name overrides: ESPN display name → our dataset name
ESPN_TO_DATASET: dict[str, str] = {
    "Türkiye":             "Turkey",
    "Czechia":             "Czech Republic",
    "Bosnia-Herzegovina":  "Bosnia and Herzegovina",
    "Congo DR":            "DR Congo",
    "USA":                 "United States",
    "IR Iran":             "Iran",
    "Korea Republic":      "South Korea",
    "Côte d'Ivoire":       "Ivory Coast",
    "Curaçao":             "Curaçao",
}


def _normalize(name: str) -> str:
    return ESPN_TO_DATASET.get(name, name)


def fetch_fifa_rankings(force: bool = False) -> dict[str, dict]:
    """
    Returns a dict mapping team name → {rank, points}.
    Reads from cache if fresh; fetches from ESPN API otherwise.
    """
    if not force and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL:
            with open(CACHE_FILE) as f:
                return json.load(f)

    print("Fetching FIFA rankings from ESPN ...")
    try:
        resp = requests.get(ESPN_RANKINGS_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rankings = _parse_espn_rankings(data)
        CACHE_FILE.parent.mkdir(exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(rankings, f, indent=2)
        print(f"  Cached {len(rankings)} teams to {CACHE_FILE}")
        return rankings
    except Exception as e:
        print(f"  Warning: could not fetch rankings ({e})")
        if CACHE_FILE.exists():
            print("  Falling back to cached rankings.")
            with open(CACHE_FILE) as f:
                return json.load(f)
        return {}


def _parse_espn_rankings(data: dict) -> dict[str, dict]:
    rankings = {}
    try:
        for group in data.get("standings", {}).get("entries", []):
            for entry in group.get("entries", []):
                team = _normalize(entry.get("team", {}).get("displayName", ""))
                rank  = entry.get("stats", [{}])[0].get("value", 0)
                pts   = entry.get("stats", [{}])[1].get("value", 0)
                if team:
                    rankings[team] = {"rank": int(rank), "points": float(pts)}
    except Exception:
        pass

    # Fallback: try alternate ESPN structure
    if not rankings:
        try:
            for item in data.get("groups", []):
                for entry in item.get("standings", {}).get("entries", []):
                    team = _normalize(entry.get("team", {}).get("displayName", ""))
                    stats = {s["name"]: s["value"] for s in entry.get("stats", [])}
                    rank = stats.get("rank", 0)
                    pts  = stats.get("points", 0)
                    if team:
                        rankings[team] = {"rank": int(rank), "points": float(pts)}
        except Exception:
            pass

    if not rankings:
        print("  Warning: could not parse ESPN rankings response — both parse strategies returned empty.")

    return rankings


def get_rank(team: str, rankings: dict) -> int:
    """Returns FIFA rank for a team, or 100 if unknown."""
    return rankings.get(team, {}).get("rank", 100)
