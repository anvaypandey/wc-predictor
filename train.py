"""
Training pipeline — run once before launching the Streamlit app.

Usage:
    python train.py
    python train.py --data path/to/results.csv   # skip auto-download
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads KAGGLE_USERNAME and KAGGLE_KEY from .env

sys.path.insert(0, str(Path(__file__).parent))

from src.prepare import (
    load_data, load_rankings, build_rank_index,
    build_features, serialize_stats, FEATURE_COLS,
)
from src.model import train, evaluate, save_artifacts

KAGGLE_RESULTS   = "martj42/international-football-results-from-1872-to-2017"
KAGGLE_RANKINGS  = "tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now"

CACHE_RESULTS  = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "martj42" / "international-football-results-from-1872-to-2017"
)
CACHE_RANKINGS = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "tadhgfitzgerald" / "fifa-international-soccer-mens-ranking-1993now"
)


def _resolve_csv(label: str, local_path: str, cache_dir: Path,
                 kaggle_slug: str, filename: str) -> str:
    """Generic helper: local → cache → download."""
    p = Path(local_path)
    if p.exists():
        print(f"  {label}: using local file {p}")
        return str(p)
    if cache_dir.exists():
        cached = list(cache_dir.rglob(filename))
        if cached:
            print(f"  {label}: using cache {cached[0]}")
            return str(cached[0])
    print(f"  {label}: downloading from Kaggle ({kaggle_slug}) ...")
    import kagglehub
    d = Path(kagglehub.dataset_download(kaggle_slug))
    found = d / filename
    if not found.exists():
        hits = list(d.rglob(filename))
        if not hits:
            raise FileNotFoundError(f"{filename} not found in {d}")
        found = hits[0]
    print(f"  {label}: saved to {found}")
    return str(found)


def resolve_datasets(results_path: str) -> tuple[str, str]:
    """Returns (results_csv_path, rankings_csv_path)."""
    print("Locating datasets ...")
    results  = _resolve_csv("Match results",  results_path,     CACHE_RESULTS,
                            KAGGLE_RESULTS,  "results.csv")
    rankings = _resolve_csv("FIFA rankings",  "data/fifa_ranking.csv", CACHE_RANKINGS,
                            KAGGLE_RANKINGS, "fifa_ranking.csv")
    return results, rankings


def main(data_path: str = "data/results.csv") -> None:
    results_path, rankings_path = resolve_datasets(data_path)

    print(f"\nLoading match results ...")
    df = load_data(results_path)
    print(f"  {len(df):,} matches  |  {df['date'].min().year}–{df['date'].max().year}"
          f"  |  {df['home_team'].nunique()} teams")

    print("Loading FIFA rankings ...")
    rk_df = load_rankings(rankings_path)
    rank_index = build_rank_index(rk_df)
    print(f"  {len(rk_df):,} ranking entries  |  "
          f"{rk_df['rank_date'].min().date()} – {rk_df['rank_date'].max().date()}")

    print("Building rolling features (no data leakage) ...")
    X, y, team_stats, h2h = build_features(df, rank_index=rank_index)
    print(f"  {len(X):,} total rows  |  {y.value_counts().to_dict()}")

    # Drop friendlies (tier 1) from training — kept for rolling stats only
    competitive = X["tournament_tier"] >= 2
    X_comp, y_comp = X[competitive], y[competitive]
    print(f"  {competitive.sum():,} competitive rows used for training "
          f"({(~competitive).sum():,} friendlies excluded)")

    print("Training XGBoost ...")
    model, X_fit, y_fit = train(X_comp, y_comp, warmup_frac=0.0)

    print("Running 5-fold cross-validation ...")
    metrics = evaluate(model, X_fit, y_fit, cv=5)
    print(f"  CV accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")

    print("\nTop feature importances:")
    pairs = sorted(zip(FEATURE_COLS, model.feature_importances_),
                   key=lambda x: x[1], reverse=True)
    for feat, imp in pairs[:8]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<30s} {imp:.4f}  {bar}")

    print("\nSaving artifacts ...")
    all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
    ts_serial, h2h_serial = serialize_stats(team_stats, h2h)
    save_artifacts(model, ts_serial, h2h_serial, all_teams, FEATURE_COLS)
    print("Done — artifacts saved to artifacts/")
    print("\nRun the app with:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/results.csv")
    args = parser.parse_args()
    main(args.data)
