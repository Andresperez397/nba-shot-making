"""Expected-make (xFG) models for Q1: baselines, penalized logistic regression and gradient boosting.

Both learners are tuned the same way (ANALYSIS_PLAN.md): fit each setting on seasons before the
holdout season, score on the holdout season, then refit the chosen setting on all training seasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import data
from .metrics import logloss

C_GRID = np.logspace(-3, 2, 10)
HGB_GRID = [dict(max_leaf_nodes=m, learning_rate=lr, min_samples_leaf=msl)
            for m in (15, 31, 63) for lr in (0.05, 0.1) for msl in (50, 500)]
HGB_ITERS = list(range(100, 1001, 100))


class NaturalSpline:
    """Natural cubic spline basis (ESL eq. 5.4-5.5) with knots at training quantiles.

    `df` basis columns: x itself plus df - 1 truncated-power terms; linear beyond the boundary knots.
    """

    def __init__(self, df: int):
        self.df = df

    def fit(self, x: np.ndarray) -> NaturalSpline:
        self.knots = np.quantile(x, np.linspace(0, 1, self.df + 1))
        self.knots = np.unique(self.knots)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        k = self.knots
        last = k[-1]

        def d(j):
            return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - last, 0, None) ** 3) / (last - k[j])

        cols = [x] + [d(j) - d(len(k) - 2) for j in range(len(k) - 2)]
        return np.column_stack(cols)


def block_columns(block: str) -> list[str]:
    cols = list(data.LOCATION)
    if block in ("M3", "M4"):
        cols += data.TYPE
    if block == "M4":
        cols += data.CONTEXT
    return cols


class LogisticBasis:
    """Fixed design for the logistic model; spline knots and scaling are learned on training data."""

    def __init__(self, block: str):
        self.block = block

    def fit(self, d: pd.DataFrame) -> LogisticBasis:
        self.sp = {"distance_ft": NaturalSpline(6).fit(d["distance_ft"].to_numpy()),
                   "angle_deg": NaturalSpline(4).fit(d["angle_deg"].to_numpy()),
                   "x_ft": NaturalSpline(4).fit(d["x_ft"].to_numpy())}
        if self.block == "M4":
            self.sp["seconds_left_period"] = NaturalSpline(4).fit(d["seconds_left_period"].to_numpy())
            self.sp["margin25"] = NaturalSpline(4).fit(d["score_margin"].clip(-25, 25).to_numpy())
        raw = self._raw(d)
        self.mu = raw.mean(axis=0)
        self.sd = raw.std(axis=0)
        self.sd[self.sd == 0] = 1
        return self

    def _raw(self, d: pd.DataFrame) -> np.ndarray:
        dist = self.sp["distance_ft"].transform(d["distance_ft"].to_numpy())
        parts = [dist, self.sp["angle_deg"].transform(d["angle_deg"].to_numpy()),
                 self.sp["x_ft"].transform(d["x_ft"].to_numpy()),
                 d[["is_three", "corner_three"]].to_numpy(float)]
        if self.block in ("M3", "M4"):
            fam = d[[f"fam_{f}" for f in data.FAMILIES]].to_numpy(float)
            parts += [fam, (fam[:, :, None] * dist[:, None, :]).reshape(len(d), -1),
                      d[[f"mod_{m}" for m in data.MODIFIERS]].to_numpy(float)]
        if self.block == "M4":
            per = np.column_stack([(d["period_n"] == p).to_numpy(float) for p in (2, 3, 4, 5)])
            parts += [per, self.sp["seconds_left_period"].transform(d["seconds_left_period"].to_numpy()),
                      self.sp["margin25"].transform(d["score_margin"].clip(-25, 25).to_numpy()),
                      d[["home"]].to_numpy(float)]
        return np.column_stack(parts).astype(np.float64)

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        return ((self._raw(d) - self.mu) / self.sd).astype(np.float32)


TUNE_N = 300_000  # tuning-fit sample size, same for both learners (DEVIATIONS.md)


def inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tuning data: a fixed random sample of the seasons before the holdout, and the holdout season."""
    hold = train["season"].max()
    fit = train[train["season"] < hold]
    if len(fit) > TUNE_N:
        fit = fit.sample(n=TUNE_N, random_state=0)
    return fit, train[train["season"] == hold]


def fit_predict_m0(train, test):
    return np.full(len(test), train["made"].mean())


def fit_predict_m1(train, test):
    sp = NaturalSpline(6).fit(train["distance_ft"].to_numpy())
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(sp.transform(train["distance_ft"].to_numpy()), train["made"])
    return m.predict_proba(sp.transform(test["distance_ft"].to_numpy()))[:, 1]


def fit_predict_logit(train, test, block):
    fit, hold = inner_split(train)
    b = LogisticBasis(block).fit(fit)
    xf, xh = b.transform(fit), b.transform(hold)
    scores = []
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=2000).fit(xf, fit["made"])
        scores.append(logloss(hold["made"].to_numpy(), m.predict_proba(xh)[:, 1]))
    c = C_GRID[int(np.argmin(scores))]
    b = LogisticBasis(block).fit(train)
    m = LogisticRegression(C=c, max_iter=2000).fit(b.transform(train), train["made"])
    return m.predict_proba(b.transform(test))[:, 1], {"C": float(c), "holdout_logloss": float(min(scores)),
                                                      "C_index": int(np.argmin(scores))}


def hgb(params: dict, max_iter: int, warm_start: bool = False) -> HistGradientBoostingClassifier:
    # early_stopping=False: scikit-learn otherwise stops silently on an internal split for n > 10,000.
    return HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=False, warm_start=warm_start,
                                          random_state=0, **params)


def fit_predict_hgb(train, test, block):
    cols = block_columns(block)
    fit, hold = inner_split(train)
    xf, xh = fit[cols].to_numpy(np.float32), hold[cols].to_numpy(np.float32)
    rows = []
    for i, params in enumerate(HGB_GRID):
        m = hgb(params, HGB_ITERS[0], warm_start=True)
        for it in HGB_ITERS:
            m.set_params(max_iter=it)
            m.fit(xf, fit["made"])
            rows.append({"setting": i, "iters": it,
                         "holdout_logloss": logloss(hold["made"].to_numpy(), m.predict_proba(xh)[:, 1])})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["holdout_logloss"].idxmin()]
    m = hgb(HGB_GRID[int(best["setting"])], int(best["iters"]))
    m.fit(train[cols].to_numpy(np.float32), train["made"])
    tuning = {**HGB_GRID[int(best["setting"])], "iters": int(best["iters"]),
              "holdout_logloss": float(best["holdout_logloss"])}
    return m.predict_proba(test[cols].to_numpy(np.float32))[:, 1], tuning


def fit_predict(train, test, block, learner):
    if learner == "const":
        return fit_predict_m0(train, test), {}
    if learner == "distance":
        return fit_predict_m1(train, test), {}
    if learner == "logit":
        return fit_predict_logit(train, test, block)
    if learner == "hgb":
        return fit_predict_hgb(train, test, block)
    raise ValueError(learner)
