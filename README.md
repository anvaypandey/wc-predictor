# World Cup Match Predictor ⚽

A machine learning app that predicts the outcome of a FIFA World Cup match — **Win, Draw, or Loss** — using historical international football data.

Built with Python, Scikit-learn, and Streamlit.

---

## Demo

Select any two national teams, toggle neutral venue, and get an instant prediction with confidence percentages, team stats, and head-to-head history.

---

## Features

- Predicts Win / Draw / Loss with probability breakdown
- Rolling feature engineering — no data leakage (each match is predicted using only prior data)
- Head-to-head record between any two teams
- Team stats comparison: win rate, avg goals, recent form
- Auto-downloads dataset from Kaggle on first run

---

## Tech Stack

| Layer       | Library              |
|-------------|----------------------|
| Data        | Pandas, kagglehub    |
| Model       | Scikit-learn (RandomForest) |
| UI          | Streamlit, Plotly    |
| Persistence | joblib               |
| Config      | python-dotenv        |

---

## Project Structure

```
WC Predictor/
├── app.py              # Streamlit UI
├── train.py            # Training pipeline (run once)
├── requirements.txt
├── .env                # Kaggle credentials (not committed)
├── src/
│   ├── prepare.py      # Feature engineering
│   └── model.py        # Train, evaluate, save/load
├── data/
│   └── README.md       # Dataset info
└── artifacts/          # Saved model + team stats (generated)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd "WC Predictor"
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
- Download the dataset automatically (cached after first run)
- Engineer rolling features from 47,000+ historical matches
- Train a RandomForest classifier
- Print cross-validated accuracy and top feature importances
- Save model artifacts to `artifacts/`

### 4. Launch the app

```bash
streamlit run app.py
```

---

## How It Works

### Features (17 total)

For each match, the model uses statistics computed from all *prior* matches only:

| Feature group      | Features                                      |
|--------------------|-----------------------------------------------|
| Team performance   | Win rate, draw rate, loss rate                |
| Goals              | Avg goals scored, avg goals conceded          |
| Recent form        | Points per game over last 10 matches          |
| Head-to-head       | H2H win/draw/loss rate, total games played    |
| Venue              | Neutral ground flag                           |

### Model

- **Algorithm:** Random Forest (300 trees, balanced class weights)
- **Target:** Win / Draw / Loss from Team 1's perspective
- **Evaluation:** Stratified 5-fold cross-validation

### Data

[International football results from 1872 to 2017+](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — over 47,000 international matches across all tournaments.

---

## Usage Notes

- "Team 1" is treated as the home team; tick **Neutral venue** for World Cup-style predictions
- Teams with limited match history will have less reliable predictions
- The model is trained on all international matches, not World Cup matches only — this maximises training data

---

## License

MIT
