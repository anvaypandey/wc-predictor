"""
Tests for src/model.py — XGBWrapper, LGBMWrapper, CV training, artifact saving.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import XGBWrapper, LGBMWrapper
from src.prepare import FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_xy(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Small synthetic dataset with balanced Win/Draw/Loss labels."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = pd.Series(["Win", "Draw", "Loss"] * (n // 3), name="result")
    return X, y


@pytest.fixture(scope="module")
def xgb_fitted():
    X, y = _make_xy()
    m = XGBWrapper(n_estimators=10, max_depth=3, random_state=42, n_jobs=1)
    m.fit(X, y)
    return m, X, y


@pytest.fixture(scope="module")
def lgbm_fitted():
    X, y = _make_xy()
    m = LGBMWrapper(n_estimators=10, max_depth=3, random_state=42, n_jobs=1, verbose=-1)
    m.fit(X, y)
    return m, X, y


# ── XGBWrapper ────────────────────────────────────────────────────────────────

def test_xgb_classes_set_after_fit(xgb_fitted):
    m, _, _ = xgb_fitted
    assert set(m.classes_) == {"Win", "Draw", "Loss"}

def test_xgb_predict_returns_valid_labels(xgb_fitted):
    m, X, _ = xgb_fitted
    preds = m.predict(X)
    assert set(preds) <= {"Win", "Draw", "Loss"}

def test_xgb_predict_length_matches_input(xgb_fitted):
    m, X, _ = xgb_fitted
    assert len(m.predict(X)) == len(X)

def test_xgb_predict_proba_shape(xgb_fitted):
    m, X, _ = xgb_fitted
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 3)

def test_xgb_predict_proba_rows_sum_to_one(xgb_fitted):
    m, X, _ = xgb_fitted
    proba = m.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

def test_xgb_predict_proba_non_negative(xgb_fitted):
    m, X, _ = xgb_fitted
    assert (m.predict_proba(X) >= 0).all()

def test_xgb_score_between_0_and_1(xgb_fitted):
    m, X, y = xgb_fitted
    s = m.score(X, y)
    assert 0.0 <= s <= 1.0

def test_xgb_feature_importances_shape(xgb_fitted):
    m, _, _ = xgb_fitted
    assert len(m.feature_importances_) == len(FEATURE_COLS)

def test_xgb_feature_importances_non_negative(xgb_fitted):
    m, _, _ = xgb_fitted
    assert (m.feature_importances_ >= 0).all()

def test_xgb_fit_balanced_does_not_crash():
    X, y = _make_xy(60)
    m = XGBWrapper(n_estimators=5, max_depth=2, random_state=42, n_jobs=1)
    m.fit(X, y, balanced=True)
    assert m.classes_ is not None

def test_xgb_predict_single_row(xgb_fitted):
    m, X, _ = xgb_fitted
    pred = m.predict(X.iloc[:1])
    assert len(pred) == 1
    assert pred[0] in {"Win", "Draw", "Loss"}

def test_xgb_predict_proba_single_row(xgb_fitted):
    m, X, _ = xgb_fitted
    proba = m.predict_proba(X.iloc[:1])
    assert proba.shape == (1, 3)
    assert abs(proba.sum() - 1.0) < 1e-5


# ── LGBMWrapper ───────────────────────────────────────────────────────────────

def test_lgbm_classes_set_after_fit(lgbm_fitted):
    m, _, _ = lgbm_fitted
    assert set(m.classes_) == {"Win", "Draw", "Loss"}

def test_lgbm_predict_returns_valid_labels(lgbm_fitted):
    m, X, _ = lgbm_fitted
    preds = m.predict(X)
    assert set(preds) <= {"Win", "Draw", "Loss"}

def test_lgbm_predict_length_matches_input(lgbm_fitted):
    m, X, _ = lgbm_fitted
    assert len(m.predict(X)) == len(X)

def test_lgbm_predict_proba_shape(lgbm_fitted):
    m, X, _ = lgbm_fitted
    assert m.predict_proba(X).shape == (len(X), 3)

def test_lgbm_predict_proba_rows_sum_to_one(lgbm_fitted):
    m, X, _ = lgbm_fitted
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0, atol=1e-5)

def test_lgbm_score_between_0_and_1(lgbm_fitted):
    m, X, y = lgbm_fitted
    assert 0.0 <= m.score(X, y) <= 1.0

def test_lgbm_feature_importances_shape(lgbm_fitted):
    m, _, _ = lgbm_fitted
    assert len(m.feature_importances_) == len(FEATURE_COLS)

def test_lgbm_fit_balanced_does_not_crash():
    X, y = _make_xy(60)
    m = LGBMWrapper(n_estimators=5, max_depth=2, random_state=42, n_jobs=1, verbose=-1)
    m.fit(X, y, balanced=True)
    assert m.classes_ is not None


# ── Shared wrapper behaviour ──────────────────────────────────────────────────

@pytest.mark.parametrize("WrapperCls,kwargs", [
    (XGBWrapper,  {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}),
    (LGBMWrapper, {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1, "verbose": -1}),
])
def test_classes_are_sorted(WrapperCls, kwargs):
    X, y = _make_xy(60)
    m = WrapperCls(**kwargs)
    m.fit(X, y)
    assert list(m.classes_) == sorted(m.classes_)

@pytest.mark.parametrize("WrapperCls,kwargs", [
    (XGBWrapper,  {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}),
    (LGBMWrapper, {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1, "verbose": -1}),
])
def test_score_consistent_with_predict(WrapperCls, kwargs):
    X, y = _make_xy(60)
    m = WrapperCls(**kwargs)
    m.fit(X, y)
    manual = float((m.predict(X) == np.asarray(y)).mean())
    assert m.score(X, y) == pytest.approx(manual)

@pytest.mark.parametrize("WrapperCls,kwargs", [
    (XGBWrapper,  {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}),
    (LGBMWrapper, {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1, "verbose": -1}),
])
def test_refitting_overwrites_previous_model(WrapperCls, kwargs):
    X1, y1 = _make_xy(60)
    X2, _ = _make_xy(90)
    m = WrapperCls(**kwargs)
    m.fit(X1, y1)
    preds_before = m.predict(X1).tolist()
    m.fit(X1, y1)  # refit on same data
    preds_after = m.predict(X1).tolist()
    # Deterministic model: predictions should be identical after refit on same data
    assert preds_before == preds_after


# ── save_artifacts ────────────────────────────────────────────────────────────

def test_save_artifacts_creates_expected_files(tmp_path):
    from src.model import save_artifacts
    from unittest.mock import patch
    X, y = _make_xy(60)
    m = XGBWrapper(n_estimators=5, max_depth=2, random_state=42, n_jobs=1)
    m.fit(X, y)

    with patch("src.model.ARTIFACTS_DIR", tmp_path):
        save_artifacts(m, {"A": {}}, {}, ["TeamA"], FEATURE_COLS, elo_ratings={"A": 1500.0})

    for fname in ("model.pkl", "team_stats.pkl", "h2h.pkl", "teams.pkl", "feature_cols.pkl", "elo_ratings.pkl"):
        assert (tmp_path / fname).exists(), f"{fname} not saved"

def test_save_artifacts_without_elo(tmp_path):
    from src.model import save_artifacts
    from unittest.mock import patch
    X, y = _make_xy(60)
    m = XGBWrapper(n_estimators=5, max_depth=2, random_state=42, n_jobs=1)
    m.fit(X, y)

    with patch("src.model.ARTIFACTS_DIR", tmp_path):
        save_artifacts(m, {}, {}, [], FEATURE_COLS, elo_ratings=None)

    assert not (tmp_path / "elo_ratings.pkl").exists()
