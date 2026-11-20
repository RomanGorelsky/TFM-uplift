"""Uplift methods. Every method has one shape, as in Zhang et al.'s cate.py:

    tau_hat = method(y, T, X, X_test, seed=..., device=...)

X and X_test are float numpy arrays; y and T are 0/1 integer arrays.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- baselines

def random_scores(y, T, X, X_test, seed=0, **kw):
    """Sanity check: a random ranking should give Qini/AUUC near zero."""
    return np.random.default_rng(seed).normal(size=len(X_test))


def _lgbm(seed):
    from lightgbm import LGBMClassifier

    # Untuned defaults for the pilot; the full study tunes these (FLAML, as in Zhang et al.).
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)


def lgbm_t(y, T, X, X_test, seed=0, **kw):
    m1 = _lgbm(seed).fit(X[T == 1], y[T == 1])
    m0 = _lgbm(seed).fit(X[T == 0], y[T == 0])
    return m1.predict_proba(X_test)[:, 1] - m0.predict_proba(X_test)[:, 1]


def lgbm_s(y, T, X, X_test, seed=0, **kw):
    m = _lgbm(seed).fit(np.column_stack([T, X]), y)
    return _s_diff(m, X_test)

# ---------------------------------------------------------------- TabPFN

def _tabpfn(seed, device):
    from tabpfn import TabPFNClassifier

    # Model version is chosen with the TABPFN_MODEL_VERSION environment variable (default v3.5).
    return TabPFNClassifier(device=device, random_state=seed, ignore_pretraining_limits=True)


def tabpfn_t(y, T, X, X_test, seed=0, device="cuda", **kw):
    m1 = _tabpfn(seed, device).fit(X[T == 1], y[T == 1])
    m0 = _tabpfn(seed, device).fit(X[T == 0], y[T == 0])
    return m1.predict_proba(X_test)[:, 1] - m0.predict_proba(X_test)[:, 1]


def tabpfn_s(y, T, X, X_test, seed=0, device="cuda", **kw):
    m = _tabpfn(seed, device).fit(np.column_stack([T, X]), y)
    return _s_diff(m, X_test)

# ---------------------------------------------------------------- CausalPFN

def causalpfn(y, T, X, X_test, seed=0, device="cuda", max_ctx=4096, n_neigh=1024, **kw):
    import torch
    from causalpfn import CATEEstimator

    torch.manual_seed(seed)
    est = CATEEstimator(device=device, max_context_length=max_ctx, num_neighbours=n_neigh)
    est.fit(X.astype(np.float32), T.astype(np.float32), y.astype(np.float32))
    return np.asarray(est.estimate_cate(X_test.astype(np.float32))).ravel()

# ---------------------------------------------------------------- helpers

def _s_diff(model, X_test):
    one, zero = np.ones(len(X_test)), np.zeros(len(X_test))
    return (model.predict_proba(np.column_stack([one, X_test]))[:, 1]
            - model.predict_proba(np.column_stack([zero, X_test]))[:, 1])


REGISTRY = {
    "random": random_scores,
    "lgbm_t": lgbm_t,
    "lgbm_s": lgbm_s,
    "tabpfn_t": tabpfn_t,
    "tabpfn_s": tabpfn_s,
    "causalpfn": causalpfn,
}
