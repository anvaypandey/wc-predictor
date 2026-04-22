import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ARTIFACTS_DIR = Path("artifacts")


class _BaseWrapper:
    """Shared interface: accepts string labels, exposes sklearn-compatible API."""

    def __init__(self):
        self._le = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y, balanced: bool = False):
        y_enc = self._le.fit_transform(y)
        if balanced:
            # Mild correction: partially rebalance toward Draw/Loss without overcorrecting.
            # Full balance hurts Win recall too much (Win is genuinely the most common outcome).
            raw_w = compute_sample_weight("balanced", y_enc)
            sw = 1.0 + (raw_w - 1.0) * 0.35   # blend 35% of the full correction
        else:
            sw = None
        self._fit_clf(X, y_enc, sw)
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


class XGBWrapper(_BaseWrapper):
    def __init__(self, **kwargs):
        super().__init__()
        self._clf = XGBClassifier(**kwargs)

    def _fit_clf(self, X, y_enc, sample_weight):
        self._clf.fit(X, y_enc, sample_weight=sample_weight)


class LGBMWrapper(_BaseWrapper):
    def __init__(self, **kwargs):
        super().__init__()
        self._clf = LGBMClassifier(**kwargs)

    def _fit_clf(self, X, y_enc, sample_weight):
        self._clf.fit(X, y_enc, sample_weight=sample_weight)


def _make_xgb() -> XGBWrapper:
    return XGBWrapper(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def _make_lgbm() -> LGBMWrapper:
    return LGBMWrapper(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        num_leaves=50,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _cv_score(make_fn, X: pd.DataFrame, y: pd.Series,
              cv: int, balanced: bool) -> tuple[float, float, list, np.ndarray, np.ndarray, list]:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    oof_preds  = pd.Series(index=X.index, dtype=object)
    oof_probas = np.zeros((len(X), 3))
    classes_ref = None

    for _, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        m = make_fn()
        m.fit(X.iloc[train_idx], y.iloc[train_idx], balanced=balanced)
        scores.append(m.score(X.iloc[val_idx], y.iloc[val_idx]))
        oof_preds.iloc[val_idx] = m.predict(X.iloc[val_idx])
        oof_probas[val_idx]     = m.predict_proba(X.iloc[val_idx])
        if classes_ref is None:
            classes_ref = list(m.classes_)

    return float(np.mean(scores)), float(np.std(scores)), scores, oof_preds, oof_probas, classes_ref


def train(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train both XGBoost and LightGBM with balanced class weights.
    Returns the better model and its CV metrics.
    """
    print("  Evaluating XGBoost ...")
    xgb_mean, xgb_std, xgb_scores, xgb_preds, xgb_probas, classes = \
        _cv_score(_make_xgb, X, y, cv=5, balanced=True)
    print(f"    XGBoost CV: {xgb_mean:.4f} ± {xgb_std:.4f}")

    print("  Evaluating LightGBM ...")
    lgbm_mean, lgbm_std, lgbm_scores, lgbm_preds, lgbm_probas, _ = \
        _cv_score(_make_lgbm, X, y, cv=5, balanced=True)
    print(f"    LightGBM CV: {lgbm_mean:.4f} ± {lgbm_std:.4f}")

    if lgbm_mean >= xgb_mean:
        print(f"  → LightGBM wins (+{lgbm_mean-xgb_mean:.4f}), using LightGBM")
        make_fn = _make_lgbm
        best_mean, best_std, best_scores = lgbm_mean, lgbm_std, lgbm_scores
        oof_preds, oof_probas = lgbm_preds, lgbm_probas
    else:
        print(f"  → XGBoost wins (+{xgb_mean-lgbm_mean:.4f}), using XGBoost")
        make_fn = _make_xgb
        best_mean, best_std, best_scores = xgb_mean, xgb_std, xgb_scores
        oof_preds, oof_probas = xgb_preds, xgb_probas

    # Train final model on full data
    model = make_fn()
    model.fit(X, y, balanced=True)

    metrics = {
        "cv_accuracy_mean": best_mean,
        "cv_accuracy_std":  best_std,
        "cv_scores":        best_scores,
        "oof_predictions":  oof_preds,
        "oof_probas":       oof_probas,
        "classes":          classes,
        "model_type":       "LightGBM" if lgbm_mean >= xgb_mean else "XGBoost",
    }
    return model, metrics


def save_artifacts(model, team_stats: dict, h2h: dict,
                   teams: list, feature_cols: list,
                   elo_ratings: dict | None = None) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model,        ARTIFACTS_DIR / "model.pkl")
    joblib.dump(team_stats,   ARTIFACTS_DIR / "team_stats.pkl")
    joblib.dump(h2h,          ARTIFACTS_DIR / "h2h.pkl")
    joblib.dump(teams,        ARTIFACTS_DIR / "teams.pkl")
    joblib.dump(feature_cols, ARTIFACTS_DIR / "feature_cols.pkl")
    if elo_ratings is not None:
        joblib.dump(elo_ratings, ARTIFACTS_DIR / "elo_ratings.pkl")
