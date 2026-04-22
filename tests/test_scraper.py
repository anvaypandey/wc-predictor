"""
Tests for src/scraper.py — ranking parsing, name mapping, cache logic.
"""

import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import (
    _parse_rankings,
    _team_name,
    fetch_fifa_rankings,
    FIFA_TO_DATASET,
    CACHE_FILE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entry(rank: int, name: str, pts: float = 1500.0) -> dict:
    """Build a minimal FIFA API rankings entry."""
    return {
        "Rank": rank,
        "TeamName": [{"Locale": "en-GB", "Description": name}],
        "DecimalTotalPoints": pts,
    }


# ── _team_name ────────────────────────────────────────────────────────────────

def test_team_name_en_locale_used():
    entry = {
        "TeamName": [
            {"Locale": "fr-FR", "Description": "Brésil"},
            {"Locale": "en-GB", "Description": "Brazil"},
        ]
    }
    assert _team_name(entry) == "Brazil"

def test_team_name_en_prefix_match():
    # "en" prefix: "en-US" should also match startswith("en")
    entry = {"TeamName": [{"Locale": "en-US", "Description": "United States"}]}
    assert _team_name(entry) == "United States"

def test_team_name_falls_back_to_first_when_no_en():
    entry = {"TeamName": [{"Locale": "de-DE", "Description": "Brasilien"}]}
    assert _team_name(entry) == "Brasilien"

def test_team_name_applies_dataset_mapping():
    for fifa_name, dataset_name in FIFA_TO_DATASET.items():
        entry = {"TeamName": [{"Locale": "en-GB", "Description": fifa_name}]}
        assert _team_name(entry) == dataset_name

def test_team_name_no_mapping_returns_as_is():
    entry = {"TeamName": [{"Locale": "en-GB", "Description": "Brazil"}]}
    assert _team_name(entry) == "Brazil"

def test_team_name_empty_team_name_list():
    entry = {"TeamName": []}
    assert _team_name(entry) == ""


# ── _parse_rankings ───────────────────────────────────────────────────────────

def test_parse_rankings_from_results_key():
    data = {"Results": [_entry(1, "France", 1800.0), _entry(2, "Brazil", 1790.0)]}
    r = _parse_rankings(data)
    assert "France" in r
    assert r["France"]["rank"] == 1
    assert r["France"]["points"] == pytest.approx(1800.0)

def test_parse_rankings_from_list_directly():
    data = [_entry(1, "France", 1800.0)]
    r = _parse_rankings(data)
    assert "France" in r

def test_parse_rankings_empty_results():
    assert _parse_rankings({"Results": []}) == {}
    assert _parse_rankings([]) == {}
    assert _parse_rankings({}) == {}

def test_parse_rankings_applies_name_mapping():
    data = {"Results": [_entry(1, "Korea Republic", 1600.0)]}
    r = _parse_rankings(data)
    assert "South Korea" in r
    assert "Korea Republic" not in r

def test_parse_rankings_skips_zero_rank():
    data = {"Results": [_entry(0, "TestTeam", 1000.0)]}
    r = _parse_rankings(data)
    assert "TestTeam" not in r

def test_parse_rankings_skips_empty_team_name():
    entry = {"Rank": 1, "TeamName": [], "DecimalTotalPoints": 1500.0}
    r = _parse_rankings({"Results": [entry]})
    assert r == {}

def test_parse_rankings_multiple_entries():
    entries = [_entry(i, f"Team{i}", float(1600 - i)) for i in range(1, 211)]
    r = _parse_rankings({"Results": entries})
    assert len(r) == 210

def test_parse_rankings_fallback_to_total_points():
    entry = {
        "Rank": 1,
        "TeamName": [{"Locale": "en-GB", "Description": "France"}],
        "TotalPoints": 1750,   # no DecimalTotalPoints key
    }
    r = _parse_rankings({"Results": [entry]})
    assert r["France"]["points"] == pytest.approx(1750.0)

def test_parse_rankings_rank_is_int():
    data = {"Results": [_entry(1, "France")]}
    r = _parse_rankings(data)
    assert isinstance(r["France"]["rank"], int)

def test_parse_rankings_points_is_float():
    data = {"Results": [_entry(1, "France", 1800)]}
    r = _parse_rankings(data)
    assert isinstance(r["France"]["points"], float)


# ── fetch_fifa_rankings — cache logic ─────────────────────────────────────────

def test_fetch_returns_cached_when_fresh(tmp_path):
    cache = tmp_path / "fifa_rankings.json"
    cached = {"France": {"rank": 1, "points": 1800.0}}
    cache.write_text(json.dumps(cached))
    # mtime = now → cache is fresh
    with patch("src.scraper.CACHE_FILE", cache):
        result = fetch_fifa_rankings(force=False)
    assert result == cached

def test_fetch_ignores_stale_cache_and_hits_api(tmp_path):
    cache = tmp_path / "fifa_rankings.json"
    stale = {"OldTeam": {"rank": 1, "points": 1000.0}}
    cache.write_text(json.dumps(stale))
    # backdate mtime by 2 days
    old_time = time.time() - 2 * 86400
    import os; os.utime(cache, (old_time, old_time))

    api_response = {"Results": [_entry(1, "France", 1800.0)]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_response
    mock_resp.raise_for_status = MagicMock()

    with patch("src.scraper.CACHE_FILE", cache), \
         patch("src.scraper.requests.get", return_value=mock_resp):
        result = fetch_fifa_rankings(force=False)

    assert "France" in result
    assert "OldTeam" not in result

def test_fetch_force_bypasses_fresh_cache(tmp_path):
    cache = tmp_path / "fifa_rankings.json"
    cached = {"OldTeam": {"rank": 1, "points": 1000.0}}
    cache.write_text(json.dumps(cached))

    api_response = {"Results": [_entry(1, "France", 1800.0)]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_response
    mock_resp.raise_for_status = MagicMock()

    with patch("src.scraper.CACHE_FILE", cache), \
         patch("src.scraper.requests.get", return_value=mock_resp):
        result = fetch_fifa_rankings(force=True)

    assert "France" in result

def test_fetch_falls_back_to_stale_cache_on_http_error(tmp_path):
    cache = tmp_path / "fifa_rankings.json"
    stale = {"France": {"rank": 1, "points": 1800.0}}
    cache.write_text(json.dumps(stale))
    old_time = time.time() - 2 * 86400
    import os; os.utime(cache, (old_time, old_time))

    with patch("src.scraper.CACHE_FILE", cache), \
         patch("src.scraper.requests.get", side_effect=Exception("Network error")):
        result = fetch_fifa_rankings(force=False)

    assert result == stale

def test_fetch_returns_empty_dict_when_no_cache_and_api_fails(tmp_path):
    cache = tmp_path / "no_cache.json"   # does not exist
    with patch("src.scraper.CACHE_FILE", cache), \
         patch("src.scraper.requests.get", side_effect=Exception("Network error")):
        result = fetch_fifa_rankings(force=False)
    assert result == {}

def test_fetch_writes_cache_after_successful_fetch(tmp_path):
    cache = tmp_path / "fifa_rankings.json"

    api_response = {"Results": [_entry(1, "France", 1800.0)]}
    mock_resp = MagicMock()
    mock_resp.json.return_value = api_response
    mock_resp.raise_for_status = MagicMock()

    with patch("src.scraper.CACHE_FILE", cache), \
         patch("src.scraper.requests.get", return_value=mock_resp):
        fetch_fifa_rankings(force=True)

    assert cache.exists()
    written = json.loads(cache.read_text())
    assert "France" in written

def test_fetch_raises_on_empty_api_response(tmp_path):
    cache = tmp_path / "no_cache.json"

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"Results": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.scraper.CACHE_FILE", cache), \
         patch("src.scraper.requests.get", return_value=mock_resp):
        # empty rankings triggers ValueError → caught → returns {}
        result = fetch_fifa_rankings(force=True)
    assert result == {}
