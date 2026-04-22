# World Cup Match Predictor

A machine learning app that predicts the outcome of any international football match and simulates the 2026 FIFA World Cup bracket using historical data from 1872 to the present.

---

## App Pages

| Page | Description |
|------|-------------|
| **Match Predictor** | Win / Draw / Loss probabilities for any two national teams, with SHAP feature explanations, team stats, and head-to-head record |
| **Bracket Simulator** | Full 2026 WC Monte Carlo simulation — group stage standings, R32 bracket, knockout round probabilities, predicted champion |
| **Model Accuracy** | CV accuracy, WC back-test by year, confusion matrix, calibration curve, feature importance, per-tournament accuracy |

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Data | Pandas, kagglehub |
| Model | XGBoost, LightGBM, scikit-learn |
| Explanations | SHAP |
| UI | Streamlit, Plotly |
| Persistence | joblib |
| Live rankings | ESPN unofficial API (cached 24h) |

---

## Project Structure

```
WC Predictor/
├── app.py                    # Streamlit navigation entry point
├── train.py                  # Training pipeline (run once)
├── requirements.txt
├── .env                      # Kaggle credentials (not committed)
├── src/
│   ├── prepare.py            # Feature engineering + ELO ratings
│   ├── model.py              # XGBoost/LightGBM wrappers, CV, training
│   ├── bracket.py            # Monte Carlo group + knockout simulator
│   ├── scraper.py            # Live FIFA rankings via ESPN
│   └── flags.py              # Country flag emoji map
├── pages/
│   ├── 1_Match_Predictor.py
│   ├── 2_Bracket_Simulator.py
│   └── 3_Accuracy.py
├── data/
│   ├── results.csv           # Match results (auto-downloaded)
│   ├── fifa_ranking.csv      # Historical rankings (auto-downloaded)
│   ├── wc2026_groups.json    # 2026 WC group draw
│   └── README.md
└── artifacts/                # Generated after training
    ├── model.pkl
    ├── team_stats.pkl
    ├── h2h.pkl
    ├── elo_ratings.pkl
    ├── backtest.pkl
    └── metrics.json
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Kaggle credentials

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Get your API key from [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**.

### 3. Train the model

```bash
python train.py
```

This will:
- Download both datasets automatically (cached after first run)
- Compute rolling ELO ratings across 49,000+ matches
- Engineer 33 features per match with no data leakage
- Evaluate XGBoost vs LightGBM via 5-fold CV and select the better one
- Save model artifacts and a back-test dataset to `artifacts/`

Runtime: ~3–5 minutes.

### 4. Launch the app

```bash
streamlit run app.py
```

---

## How It Works

### Feature Engineering (33 features)

All features are computed from matches *prior* to the one being predicted — no leakage.

| Group | Features |
|-------|----------|
| Overall rolling stats | Win/draw/loss rate, avg goals scored/conceded, recent form (last 10) |
| Competitive-only stats | Win/draw rate in tier ≥ 2 matches (qualifications, tournaments) |
| Short-window form | Points rate over last 5 matches, avg goal difference over last 10 |
| Head-to-head | H2H win/draw/loss rate, total games played |
| Context | Neutral venue flag, tournament tier (1–5) |
| FIFA rankings | Home rank, away rank, rank difference |
| ELO ratings | Home ELO, away ELO, ELO difference, absolute ELO difference |

### ELO Rating System

ELO ratings are computed from scratch across the full match history (1872–present):

- Starting rating: 1500 for all teams
- K factor scales with tournament importance: 16 (friendly) → 50 (World Cup)
- Margin-of-victory adjustment: `K_eff = K × min(1 + 0.5 × |goal diff|, 2.5)`
- ELO difference and absolute ELO difference are the two most predictive features

### Tournament Tiers

| Tier | Tournaments |
|------|-------------|
| 5 | FIFA World Cup |
| 4 | UEFA Euro, Copa América, AFCON, AFC Asian Cup, Gold Cup |
| 3 | WC/Euro/AFCON qualification, UEFA/CONCACAF Nations League |
| 2 | Other competitive matches |
| 1 | Friendlies |

Friendlies are included for rolling stats and ELO updates but excluded from training.

### Model Selection

Both XGBoost and LightGBM are evaluated via 5-fold stratified CV on each training run. The model with higher mean CV accuracy is selected and retrained on the full dataset. Mild class weight balancing (35% of full correction) is applied to improve draw recall without sacrificing overall accuracy.

### Monte Carlo Simulation

The bracket simulator runs N independent simulations (default 500):
- Group stage: all 6 round-robin matches per group simulated; teams ranked by points then goal difference
- Best 8 third-place teams across 12 groups advance to R32
- Knockout rounds: draws resolved as 50/50 (penalty shootout)
- Final probabilities are frequencies across all simulations

---

## Data Sources

- **Match results**: [International football results 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle)
- **Historical FIFA rankings**: [FIFA rankings 1993–2018](https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now) (Kaggle)
- **Live FIFA rankings**: ESPN unofficial API (cached locally, refreshed every 24h)

---

## License

MIT
