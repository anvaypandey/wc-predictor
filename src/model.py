import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

ARTIFACTS_DIR = Path("artifacts")


def train(X: pd.DataFrame, y: pd.Series, warmup_frac: float = 0.05):
    """
    Train a RandomForest on the feature matrix.
    Skips the first `warmup_frac` of rows where teams have little history,
    which would otherwise pollute the model with near-zero stats.
    """
    warmup = int(len(X) * warmup_frac)
    X_fit = X.iloc[warmup:].copy()
    y_fit = y.iloc[warmup:].copy()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)
    return model, X_fit, y_fit


def evaluate(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return {
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std":  float(scores.std()),
        "cv_scores":        scores.tolist(),
    }


def save_artifacts(model, team_stats: dict, h2h: dict,
                   teams: list, feature_cols: list) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,        ARTIFACTS_DIR / "model.pkl")
    joblib.dump(team_stats,   ARTIFACTS_DIR / "team_stats.pkl")
    joblib.dump(h2h,          ARTIFACTS_DIR / "h2h.pkl")
    joblib.dump(teams,        ARTIFACTS_DIR / "teams.pkl")
    joblib.dump(feature_cols, ARTIFACTS_DIR / "feature_cols.pkl")


def load_artifacts() -> tuple:
    model        = joblib.load(ARTIFACTS_DIR / "model.pkl")
    team_stats   = joblib.load(ARTIFACTS_DIR / "team_stats.pkl")
    h2h          = joblib.load(ARTIFACTS_DIR / "h2h.pkl")
    teams        = joblib.load(ARTIFACTS_DIR / "teams.pkl")
    feature_cols = joblib.load(ARTIFACTS_DIR / "feature_cols.pkl")
    return model, team_stats, h2h, teams, feature_cols
