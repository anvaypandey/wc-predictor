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

from src.prepare import load_data, build_features, serialize_stats, FEATURE_COLS
from src.model import train, evaluate, save_artifacts


KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"
KAGGLE_CACHE   = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "martj42" / "international-football-results-from-1872-to-2017"
)


def resolve_data(data_path: str) -> str:
    """
    Locate results.csv using three checks in order:
      1. Local path (data/results.csv or --data argument)
      2. kagglehub on-disk cache (~/.cache/kagglehub/...)
      3. Fresh download from Kaggle
    """
    # 1. Local file
    p = Path(data_path)
    if p.exists():
        print(f"Using local dataset: {p}")
        return str(p)

    # 2. kagglehub cache
    if KAGGLE_CACHE.exists():
        cached = list(KAGGLE_CACHE.rglob("results.csv"))
        if cached:
            print(f"Using cached dataset: {cached[0]}")
            return str(cached[0])

    # 3. Download
    print("Dataset not found locally or in cache — downloading from Kaggle ...")
    import kagglehub
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv = dataset_dir / "results.csv"
    if not csv.exists():
        matches = list(dataset_dir.rglob("results.csv"))
        if not matches:
            raise FileNotFoundError(
                f"results.csv not found in downloaded dataset at {dataset_dir}"
            )
        csv = matches[0]
    print(f"Downloaded to: {csv}")
    return str(csv)


def main(data_path: str = "data/results.csv") -> None:
    data_path = resolve_data(data_path)
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    print(f"  {len(df):,} matches  |  {df['date'].min().year}–{df['date'].max().year}"
          f"  |  {df['home_team'].nunique()} teams")

    print("Building rolling features (no data leakage) ...")
    X, y, team_stats, h2h = build_features(df)
    dist = y.value_counts().to_dict()
    print(f"  {len(X):,} rows  |  {dist}")

    print("Training RandomForest ...")
    model, X_fit, y_fit = train(X, y)

    print("Running 5-fold cross-validation ...")
    metrics = evaluate(model, X_fit, y_fit)
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
