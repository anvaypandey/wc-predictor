# World Cup Match Predictor

A machine learning app that predicts the outcome of any **men's international football** match and simulates the **2026 FIFA Men's World Cup** bracket using historical data from 1872 to the present. Women's football is not currently supported.

---

## Pages

| Page | Description |
|------|-------------|
| **Match Predictor** | Win / Draw / Loss probabilities for any two national teams, with confidence indicator, SHAP feature explanations, team stats, head-to-head record, and shareable prediction URLs |
| **Bracket Simulator** | Full 2026 WC Monte Carlo simulation — group stage standings, R32 bracket, knockout round probabilities, predicted champion |
| **Model Accuracy** | CV accuracy, WC back-test by year, confusion matrix, calibration curve, feature importance, per-tournament accuracy |

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Data | Pandas, kagglehub |
| Model | XGBoost, LightGBM, scikit-learn |
| Explanations | SHAP |
| Backend | FastAPI, uvicorn, sse-starlette |
| Frontend | React, Vite, TypeScript, Tailwind CSS, Plotly |
| Persistence | joblib |
| Live rankings | Official FIFA API — api.fifa.com (cached 24h) |

---

## Project Structure

```
WC Predictor/
├── backend/
│   ├── main.py               # FastAPI app + CORS + static file serving
│   ├── state.py              # Artifact loading (model, stats, ELO, rankings)
│   ├── schemas.py            # Pydantic request/response models
│   └── routers/
│       ├── predict.py        # GET /api/teams  |  POST /api/predict
│       ├── simulate.py       # GET /api/simulate/stream  (SSE)
│       └── accuracy.py       # GET /api/accuracy
├── frontend/
│   ├── vite.config.ts        # Vite + Tailwind plugin + /api proxy
│   └── src/
│       ├── App.tsx           # React Router with 3 routes
│       ├── api/client.ts     # fetch + EventSource helpers + TypeScript types
│       ├── components/       # Navbar, PlotlyChart, StatCard
│       └── pages/            # MatchPredictor, BracketSimulator, ModelAccuracy
├── src/
│   ├── prepare.py            # Feature engineering + ELO ratings
│   ├── model.py              # XGBoost/LightGBM wrappers, CV, training
│   ├── bracket.py            # Monte Carlo group + knockout simulator
│   ├── scraper.py            # Live FIFA rankings via official API
│   └── flags.py              # Country flag emoji map
├── data/
│   ├── results.csv           # Match results (auto-downloaded)
│   ├── fifa_ranking.csv      # Historical rankings (auto-downloaded)
│   ├── wc2026_groups.json    # 2026 WC group draw
│   └── README.md
├── tests/
│   ├── test_prepare.py       # 55 unit tests for feature engineering
│   ├── test_bracket.py       # 44 unit tests for bracket simulation
│   ├── test_scraper.py       # 31 unit tests for rankings scraper
│   └── test_model.py         # 24 unit tests for model wrappers
├── train.py                  # Training pipeline (run once)
├── requirements.txt
├── render.yaml               # Render deployment config
└── .github/workflows/
    ├── retrain.yml           # Monthly model retraining
    └── deploy.yml            # Push-to-main → Render deploy hook
```

Artifacts (`artifacts/`) are generated at training time and committed to the repo. The monthly retrain workflow updates them automatically.

> **Performance note:** `/api/accuracy` responses are cached in memory for 1 hour. The first request builds 5 Plotly figures from the backtest pickle (~1–2 s); subsequent requests within the hour return in <5 ms.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend && npm install
```

### 3. Add Kaggle credentials

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Get your API key from [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**.

### 4. Train the model

```bash
python train.py
```

This will:
- Download both datasets automatically (cached after first run)
- Compute rolling ELO ratings across 49,000+ matches
- Engineer 33 features per match with no data leakage
- Evaluate XGBoost vs LightGBM via 5-fold CV and select the better one
- Save model artifacts to `artifacts/` (gitignored)

Runtime: ~3–5 minutes.

### 5. Run locally

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload

# Terminal 2 — frontend dev server (proxies /api → localhost:8000)
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### 6. Production build

```bash
cd frontend && npm run build
uvicorn backend.main:app   # serves React from frontend/dist/
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

### Shareable Prediction URLs

Every prediction can be shared as a deep link:

```
/?home=Brazil&away=Argentina&neutral=true
```

The page pre-fills the team selectors and auto-runs the prediction on load. A **Copy link** button appears after each prediction.

### Confidence Indicator

A badge is shown alongside each predicted outcome based on the margin between the top two outcome probabilities:

| Badge | Condition |
|-------|-----------|
| **High confidence** | Top − 2nd ≥ 30 pp (e.g. 70 % / 20 % / 10 %) |
| **Medium confidence** | 15 pp ≤ margin < 30 pp |
| **Low confidence** | margin < 15 pp (e.g. 45 % / 35 % / 20 %) |

### Monte Carlo Simulation

The bracket simulator runs N independent simulations (default 500):
- Group stage: all 6 round-robin matches per group simulated; teams ranked by points then goal difference
- Best 8 third-place teams across 12 groups advance to R32
- Knockout rounds: draws resolved as 50/50 (penalty shootout)
- Final probabilities are frequencies across all simulations

---

## Deployment

The app is deployed on [Render](https://render.com) as a single Python web service. The React frontend is built at deploy time and served as static files from FastAPI.

```yaml
# render.yaml
buildCommand: pip install -r requirements.txt && cd frontend && npm install && npm run build
startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

A GitHub Actions workflow triggers a redeploy on every push to `main`, and a separate workflow retrains the model on the first of each month using Kaggle credentials stored as repository secrets.

---

## Data Sources

- **Match results**: [International football results 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle)
- **Historical FIFA rankings**: [FIFA rankings 1993–present](https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now) (Kaggle)
- **Live FIFA rankings**: Official FIFA API — `api.fifa.com` (cached locally, refreshed every 24h)

---

## License

MIT
