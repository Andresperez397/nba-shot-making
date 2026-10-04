"""Shot-making (points above expected), empirical-Bayes shrinkage and stability tests (Q2-Q4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_N_TAU = 50  # player-seasons used to estimate the between-player variance


def pae_table(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: attempts, raw points above expected per shot, its sampling variance, xPPS and PPS."""
    x = d.assign(
        pts=d["shot_value"] * d["made"],
        xpts=d["shot_value"] * d["xfg"],
        var=d["shot_value"] ** 2 * d["xfg"] * (1 - d["xfg"]),
    )
    g = x.groupby(by).agg(n=("made", "size"), pts=("pts", "sum"), xpts=("xpts", "sum"), var=("var", "sum"))
    g["pps"] = g["pts"] / g["n"]
    g["xpps"] = g["xpts"] / g["n"]
    g["pae"] = g["pps"] - g["xpps"]
    g["v"] = g["var"] / g["n"] ** 2
    return g.drop(columns=["pts", "xpts", "var"]).reset_index()


def tau2(t: pd.DataFrame, min_n: int = MIN_N_TAU) -> float:
    """Method-of-moments between-group variance: weighted variance of raw PAE minus mean sampling variance."""
    u = t[t["n"] >= min_n]
    w = u["n"] / u["n"].sum()
    mean = (w * u["pae"]).sum()
    total = (w * (u["pae"] - mean) ** 2).sum()
    return float(max(0.0, total - (w * u["v"]).sum()))


def shrink(t: pd.DataFrame, by_season: bool = True) -> pd.DataFrame:
    """Normal-normal shrinkage of raw PAE toward zero (the league), with posterior SD."""
    out = []
    groups = t.groupby("season") if by_season else [(None, t)]
    for _, g in groups:
        g = g.copy()
        t2 = tau2(g)
        g["tau2"] = t2
        g["k"] = t2 / (t2 + g["v"]) if t2 > 0 else 0.0
        g["pae_shrunk"] = g["k"] * g["pae"]
        g["post_sd"] = np.sqrt(t2 * g["v"] / (t2 + g["v"])) if t2 > 0 else 0.0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r)


def split_half(d: pd.DataFrame, min_n: int = 200) -> pd.DataFrame:
    """Odd vs even game halves of each player-season with at least min_n attempts."""
    keep = d.groupby(["season", "person_id"])["made"].transform("size") >= min_n
    x = d[keep].assign(half=lambda z: np.where(z["game_n"] % 2 == 1, "odd", "even"))
    t = pae_table(x, ["season", "person_id", "half"])
    wide = t.pivot_table(index=["season", "person_id"], columns="half", values=["pae", "xpps", "pps", "n"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.dropna().reset_index()


def consecutive(t: pd.DataFrame, key: str, min_n: int = 200) -> pd.DataFrame:
    """Pair each group-season with the same group's next season (both with at least min_n attempts)."""
    u = t[t["n"] >= min_n]
    nxt = u.assign(season=u["season"] - 1)
    return u.merge(nxt, on=[key, "season"], suffixes=("", "_next"))


def wmse(y, yhat, w) -> float:
    w = np.asarray(w, float)
    return float(np.sum(w * (np.asarray(y) - np.asarray(yhat)) ** 2) / w.sum())


def cluster_boot(df: pd.DataFrame, cluster: str, stat, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap a statistic over resampled clusters (e.g. players or teams)."""
    rng = np.random.default_rng(seed)
    ids = df[cluster].unique()
    groups = {c: g for c, g in df.groupby(cluster)}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        out.append(stat(pd.concat([groups[c] for c in pick], ignore_index=True)))
    return np.asarray(out)
