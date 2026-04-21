# Data Setup

The dataset is downloaded automatically via **kagglehub** when you run `train.py`.
No manual steps required — kagglehub caches the download so subsequent runs are instant.

## First-time setup

You need a Kaggle account and API credentials configured once:

```bash
pip install kagglehub
```

Then either:
- Set environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`
- Or place `~/.kaggle/kaggle.json` with `{"username": "...", "key": "..."}`
  (download from Kaggle → Account → API → Create New Token)

## Run

```bash
pip install -r requirements.txt
python train.py          # auto-downloads dataset, trains model
streamlit run app.py
```

## Manual override

If you already have `results.csv` locally:

```bash
python train.py --data path/to/results.csv
```

## Dataset

**International football results from 1872 to 2017 (and beyond)**
`martj42/international-football-results-from-1872-to-2017` on Kaggle

| Column       | Description                        |
|--------------|------------------------------------|
| date         | Match date (YYYY-MM-DD)            |
| home_team    | Home team name                     |
| away_team    | Away team name                     |
| home_score   | Goals scored by home team          |
| away_score   | Goals scored by away team          |
| tournament   | Tournament name                    |
| city         | Host city                          |
| country      | Host country                       |
| neutral      | True if played at neutral venue    |
