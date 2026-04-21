import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ARTIFACTS_DIR = Path("artifacts")


class XGBWrapper:
    """
    Thin wrapper so XGBClassifier accepts string labels (Win/Draw/Loss)
    and exposes the same interface as sklearn estimators.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._clf = XGBClassifier(**kwargs)
        self._le = LabelEncoder()

    def fit(self, X, y):
        y_enc = self._le.fit_transform(y)
        self._clf.fit(X, y_enc)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._clf.predict(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())

    @property
    def feature_importances_(self):
        return self._clf.feature_importances_


def _make_model() -> XGBWrapper:
    return XGBWrapper(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def train(X: pd.DataFrame, y: pd.Series, warmup_frac: float = 0.05):
    """
    Train XGBoost on the feature matrix.
    Skips the first `warmup_frac` rows where teams have little history.
    """
    warmup = int(len(X) * warmup_frac)
    X_fit = X.iloc[warmup:].copy()
    y_fit = y.iloc[warmup:].copy()

    model = _make_model()
    model.fit(X_fit, y_fit)
    return model, X_fit, y_fit


def evaluate(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        m = _make_model()
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(m.score(X.iloc[val_idx], y.iloc[val_idx]))
    return {
        "cv_accuracy_mean": float(np.mean(scores)),
        "cv_accuracy_std":  float(np.std(scores)),
        "cv_scores":        scores,
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
