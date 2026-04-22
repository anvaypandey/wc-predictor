"""
Training pipeline — run once before launching the Streamlit app.

Usage:
    python train.py
    python train.py --data path/to/results.csv   # skip auto-download
"""

import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.prepare import (
    load_data, load_rankings, build_rank_index,
    build_features, serialize_stats, FEATURE_COLS,
)
from src.model import train, save_artifacts
import joblib

KAGGLE_RESULTS  = "martj42/international-football-results-from-1872-to-2017"
KAGGLE_RANKINGS = "tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now"

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
    print("Locating datasets ...")
    results  = _resolve_csv("Match results", results_path,          CACHE_RESULTS,
                            KAGGLE_RESULTS,  "results.csv")
    rankings = _resolve_csv("FIFA rankings", "data/fifa_ranking.csv", CACHE_RANKINGS,
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
    X, y, team_stats, h2h, elo_ratings = build_features(df, rank_index=rank_index)
    print(f"  {len(X):,} total rows  |  {y.value_counts().to_dict()}")
    top_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top ELO: {', '.join(f'{t} {r:.0f}' for t, r in top_elo)}")

    # Drop friendlies — kept for rolling stats but not training signal
    competitive = X["tournament_tier"] >= 2
    X_comp, y_comp = X[competitive], y[competitive]
    print(f"  {competitive.sum():,} competitive rows used for training "
          f"({(~competitive).sum():,} friendlies excluded)\n")

    print("Training models (XGBoost vs LightGBM, balanced class weights, 5-fold CV) ...")
    model, metrics = train(X_comp, y_comp)
    print(f"\n  Best model: {metrics['model_type']}")
    print(f"  CV accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")

    print("\nTop feature importances:")
    pairs = sorted(zip(FEATURE_COLS, model.feature_importances_),
                   key=lambda x: x[1], reverse=True)
    for feat, imp in pairs[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<30s} {imp:.4f}  {bar}")

    # Build back-test dataframe from OOF predictions
    print("\nBuilding back-test dataset ...")
    oof_preds  = metrics["oof_predictions"]
    oof_probas = metrics["oof_probas"]
    classes    = metrics["classes"]

    df_meta = df.iloc[X_comp.index][
        ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    ].copy().reset_index(drop=True)
    df_meta["actual"]    = y_comp.values
    df_meta["predicted"] = oof_preds.values
    df_meta["correct"]   = df_meta["actual"] == df_meta["predicted"]
    for i, cls in enumerate(classes):
        df_meta[f"prob_{cls}"] = oof_probas[:, i]
    print(f"  Overall OOF accuracy: {df_meta['correct'].mean():.4f}")

    wc = df_meta[df_meta["tournament"] == "FIFA World Cup"]
    if len(wc):
        print(f"  WC-only OOF accuracy: {wc['correct'].mean():.4f}  ({len(wc)} matches)")
        wc_draw = wc[wc["actual"] == "Draw"]
        print(f"  WC Draw recall:       {(wc_draw['predicted']=='Draw').mean():.4f}  ({len(wc_draw)} draws)")

    ARTIFACTS = Path("artifacts")
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(df_meta, ARTIFACTS / "backtest.pkl")

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({
            "cv_accuracy_mean": metrics["cv_accuracy_mean"],
            "cv_accuracy_std":  metrics["cv_accuracy_std"],
            "cv_scores":        metrics["cv_scores"],
            "classes":          classes,
            "n_training_rows":  len(X_comp),
            "model_type":       metrics["model_type"],
        }, f, indent=2)

    print("\nSaving artifacts ...")
    all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
    ts_serial, h2h_serial = serialize_stats(team_stats, h2h)
    save_artifacts(model, ts_serial, h2h_serial, all_teams, FEATURE_COLS, elo_ratings)
    print("Done — artifacts saved to artifacts/")
    print("\nRun the app with:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/results.csv")
    args = parser.parse_args()
    main(args.data)
