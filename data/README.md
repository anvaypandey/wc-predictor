# Data

The training datasets are downloaded automatically via **kagglehub** when you run `train.py`. No manual steps required — kagglehub caches downloads so subsequent runs are instant.

## Files

| File | Source | Purpose |
|------|--------|---------|
| `results.csv` | Kaggle (auto-downloaded) | Match results 1872–present, used for rolling stats and ELO |
| `fifa_ranking.csv` | Kaggle (auto-downloaded) | Historical FIFA rankings 1993–2018, used as training features |
| `fifa_rankings.json` | Official FIFA API (auto-fetched, 24h TTL) | Live rankings for inference, cached locally |
| `wc2026_groups.json` | Committed | 2026 WC group draw — 48 teams across 12 groups |

## Kaggle credentials

You need a Kaggle account configured once:

```bash
# Option A — .env file in project root
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Option B — standard Kaggle config
~/.kaggle/kaggle.json  →  {"username": "...", "key": "..."}
```

Get your key: Kaggle → Account → API → **Create New Token**.

## Manual data override

If you already have the CSVs locally:

```bash
python train.py --data path/to/results.csv
```

The rankings CSV is looked up automatically from the Kaggle cache or downloaded fresh if missing.

---

## Dataset schemas

### `results.csv` — Match results

Kaggle: `martj42/international-football-results-from-1872-to-2017`

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Match date |
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `home_score` | int | Goals scored by home team |
| `away_score` | int | Goals scored by away team |
| `tournament` | string | Tournament name |
| `city` | string | Host city |
| `country` | string | Host country |
| `neutral` | bool | True if played at a neutral venue |

### `fifa_ranking.csv` — Historical FIFA rankings

Kaggle: `tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now`

| Column | Type | Description |
|--------|------|-------------|
| `rank_date` | date | Date of the ranking snapshot |
| `country_full` | string | Country name (mapped to results dataset names) |
| `rank` | int | FIFA world ranking position |
| `total_points` | float | FIFA ranking points |

Used to look up each team's ranking at the time of each training match. Several country names differ between the two datasets and are normalised via `RANKINGS_NAME_MAP` in `src/prepare.py`.

### `fifa_rankings.json` — Live rankings cache

Fetched from the official FIFA API (`api.fifa.com/api/v3/rankings`) at inference time, cached for 24 hours. Format:

```json
{
  "Brazil":    {"rank": 1, "points": 1896.15},
  "Argentina": {"rank": 2, "points": 1868.96}
}
```

Refreshed automatically on first page load after the cache expires. If the fetch fails the previous cached file is used as fallback. Source: [FIFA World Ranking (Men)](https://inside.fifa.com/fifa-world-ranking/men).

### `wc2026_groups.json` — 2026 World Cup groups

```json
{
  "groups": {
    "A": ["Mexico", "USA", "Canada", "..."],
    "B": [...],
    ...
  }
}
```

48 teams across 12 groups of 4, reflecting the official 2026 FIFA World Cup draw. Used by the Bracket Simulator page.
