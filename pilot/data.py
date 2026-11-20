"""Hillstrom loading and preprocessing for the pilot.

Treatment: Women's e-mail (1) vs no e-mail (0); the men's e-mail arm is dropped.
Outcome: `visit` (binary). Conversion is too rare for small contexts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREATED, CONTROL = "Womens E-Mail", "No E-Mail"
CATEGORICAL = ["zip_code", "channel"]
DROP = ["history_segment"]  # binned copy of `history`, redundant


def load_hillstrom(data_home: str | None = None, target: str = "visit"):
    """Return (X: DataFrame of numeric features, T: int array, Y: int array)."""
    from sklift.datasets import fetch_hillstrom

    X, y, seg = fetch_hillstrom(target_col=target, data_home=data_home, return_X_y_t=True)
    keep = seg.isin([TREATED, CONTROL]).to_numpy()
    X, y, seg = X[keep].reset_index(drop=True), y[keep].to_numpy(), seg[keep].to_numpy()
    return preprocess(X), (seg == TREATED).astype(int), y.astype(int)


def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric-only features: drop redundant columns, one-hot encode categoricals."""
    X = X.drop(columns=[c for c in DROP if c in X.columns])
    X = pd.get_dummies(X, columns=[c for c in CATEGORICAL if c in X.columns], dtype=float)
    return X.astype(float)


def make_split(n: int, T: np.ndarray, Y: np.ndarray, seed: int, test_frac: float = 0.3):
    """Stratified (by T and Y) train-pool / test split of row indices."""
    from sklearn.model_selection import train_test_split

    strata = T * 2 + Y
    return train_test_split(np.arange(n), test_size=test_frac, stratify=strata, random_state=seed)


def subsample(pool: np.ndarray, T: np.ndarray, Y: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Stratified subsample of `size` rows from the training pool (all rows if size >= pool)."""
    if size >= len(pool):
        return pool
    from sklearn.model_selection import train_test_split

    strata = T[pool] * 2 + Y[pool]
    idx, _ = train_test_split(pool, train_size=size, stratify=strata, random_state=seed)
    return idx


def synthetic_hillstrom(n: int = 20000, seed: int = 0):
    """Stand-in with Hillstrom's columns and a known small uplift, for offline testing only."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": "x",
        "history": rng.gamma(2.0, 120.0, n),
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Phone", "Web", "Multichannel"], n),
    })
    T = rng.integers(0, 2, n)
    base = 0.10 + 0.04 * X["womens"] - 0.005 * X["recency"] + 0.0002 * X["history"].clip(0, 500)
    tau = 0.03 + 0.05 * X["womens"] * (X["recency"] < 6)  # uplift concentrated in one segment
    Y = (rng.random(n) < (base + T * tau).clip(0, 1)).astype(int)
    return preprocess(X), T, Y.to_numpy(), tau.to_numpy()
