from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shots import data, metrics, models, skill  # noqa: E402

HAS_DATA = (data.RAW / "nba_play_by_play_2025.parquet").exists()


def fake_shots(n=6000, seasons=(2016, 2017, 2018), seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "season": rng.choice(seasons, n), "game_id": rng.integers(0, 300, n).astype(str),
        "person_id": rng.integers(0, 80, n), "x_ft": rng.uniform(-24, 24, n), "y_ft": rng.uniform(-4, 30, n),
        "period_n": rng.integers(1, 5, n), "seconds_left_period": rng.uniform(0, 720, n),
        "score_margin": rng.integers(-20, 20, n), "home": rng.integers(0, 2, n),
    })
    d["distance_ft"] = np.hypot(d["x_ft"], d["y_ft"])
    d["angle_deg"] = np.degrees(np.arctan2(d["x_ft"].abs(), d["y_ft"]))
    d["is_three"] = (d["distance_ft"] > 23.75).astype(int)
    d["corner_three"] = ((d["is_three"] == 1) & (d["y_ft"] < 9.25)).astype(int)
    d["shot_value"] = 2 + d["is_three"]
    fam = rng.choice(data.FAMILIES, n)
    for f in data.FAMILIES:
        d[f"fam_{f}"] = (fam == f).astype(int)
    for m in data.MODIFIERS:
        d[f"mod_{m}"] = rng.integers(0, 2, n)
    d["made"] = rng.binomial(1, 1 / (1 + np.exp(-(0.8 - 0.06 * d["distance_ft"]))))
    return d


def test_no_leaky_field_is_an_input():
    assert not any(data.is_leaky_name(c) for c in data.FEATURES)
    for block in ("M2", "M3", "M4"):
        assert not set(models.block_columns(block)) & set(data.LEAKY)
    assert "person_id" not in data.FEATURES and "player_name" not in data.FEATURES


def test_shot_families():
    labels = {"Tip Layup Shot": "tip_putback", "Putback Dunk Shot": "tip_putback",
              "Driving Floating Bank Jump Shot": "floater", "Turnaround Hook Shot": "hook",
              "Cutting Finger Roll Layup Shot": "layup", "Step Back Jump shot": "jumper",
              "Alley Oop Dunk Shot": "dunk", "Jump Bank Shot": "jumper", "": "other"}
    assert data.shot_family(pd.Series(list(labels))).tolist() == list(labels.values())


def test_pre_shot_score_never_includes_the_shot_itself():
    g = pd.DataFrame({"score_home": ["0", "", "3", "", "3", "5"], "score_away": ["0", "", "0", "", "2", "2"]})
    s = data.pre_shot_score(g)
    # Row 2 is a made home three (score 3-0 on its own row); its pre-shot score must be 0-0.
    assert s["pre_home"].tolist() == [0, 0, 0, 3, 3, 3]
    assert s["pre_away"].tolist() == [0, 0, 0, 0, 0, 2]


def test_natural_spline_is_linear_beyond_boundary_knots():
    x = np.linspace(0, 30, 400)
    sp = models.NaturalSpline(6).fit(x)
    assert sp.transform(x).shape[1] == 6
    out = sp.transform(np.linspace(31, 60, 50))
    second_diff = np.diff(out, n=2, axis=0)
    assert np.allclose(second_diff, 0, atol=1e-8)


def test_test_season_outcomes_cannot_change_its_predictions():
    d = fake_shots()
    train, test = d[d["season"] < 2018], d[d["season"] == 2018]
    flipped = test.assign(made=1 - test["made"])
    old_grid, old_iters = models.HGB_GRID, models.HGB_ITERS
    models.HGB_GRID, models.HGB_ITERS = models.HGB_GRID[:1], [100, 200]
    try:
        for block, learner in [(None, "distance"), ("M4", "logit"), ("M4", "hgb")]:
            a, _ = models.fit_predict(train, test, block, learner)
            b, _ = models.fit_predict(train, flipped, block, learner)
            assert np.array_equal(a, b), learner
    finally:
        models.HGB_GRID, models.HGB_ITERS = old_grid, old_iters


def test_tuning_holds_out_the_latest_season_and_caps_the_fit_sample():
    d = fake_shots()
    fit, hold = models.inner_split(d[d["season"] < 2018])
    assert set(hold["season"]) == {2017} and fit["season"].max() < 2017
    old = models.TUNE_N
    models.TUNE_N = 500
    try:
        fit, _ = models.inner_split(d[d["season"] < 2018])
        assert len(fit) == 500
    finally:
        models.TUNE_N = old


def test_boosting_never_stops_early_and_warm_start_matches_fresh_fit():
    d = fake_shots()
    x, y = d[models.block_columns("M4")].to_numpy(np.float32), d["made"]
    a = models.hgb(models.HGB_GRID[0], 50, warm_start=True).fit(x, y)
    a.set_params(max_iter=100).fit(x, y)
    b = models.hgb(models.HGB_GRID[0], 100).fit(x, y)
    assert b.n_iter_ == 100 and not b.early_stopping
    assert np.allclose(a.predict_proba(x), b.predict_proba(x))


def test_metrics_and_bootstrap():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 3000), rng.uniform(size=3000)
    assert metrics.auc(y, p) == pytest.approx(roc_auc_score(y, p))
    q = rng.uniform(0.05, 0.95, 200000)
    cal = metrics.calibration(rng.binomial(1, q), q)
    assert abs(cal["cal_slope"] - 1) < 0.03 and abs(cal["cal_intercept"]) < 0.03
    yy = rng.binomial(1, q[:5000])
    series = pd.Series(q[:5000], index=np.arange(5000) + 10**6)  # pandas input with an arbitrary index
    assert metrics.calibration(yy, series) == metrics.calibration(yy, q[:5000])
    assert np.isnan(metrics.calibration(yy, np.full(5000, 0.4))["cal_slope"])
    game = np.repeat(np.arange(30), 5)
    season = np.repeat([1, 2, 3], 50)
    w = metrics.boot_weights(game, season, n_boot=10)
    season_of = pd.Series(season, index=game).groupby(level=0).first().reindex(w.index).to_numpy()
    assert (w.groupby(season_of).sum() == 10).all().all()
    ones = pd.DataFrame(np.ones((len(w), 1), int), index=w.index)
    vals = np.arange(150) / 10
    assert metrics.boot_mean(vals, game, ones)[0] == pytest.approx(vals.mean())


def test_shrinkage_recovers_between_player_variance():
    rng = np.random.default_rng(3)
    true_tau = 0.05
    rows = []
    for pid in range(400):
        n = int(rng.integers(100, 900))
        skill_ = rng.normal(0, true_tau)
        xfg = rng.uniform(0.35, 0.6, n)
        made = rng.binomial(1, np.clip(xfg + skill_ / 2, 0, 1))
        rows.append(pd.DataFrame({"season": 2020, "person_id": pid, "made": made, "xfg": xfg,
                                  "shot_value": 2}))
    d = pd.concat(rows)
    t = skill.shrink(skill.pae_table(d, ["season", "person_id"]))
    assert np.sqrt(t["tau2"].iloc[0]) == pytest.approx(true_tau, rel=0.25)
    assert ((t["k"] > 0) & (t["k"] < 1)).all()
    assert (t["pae_shrunk"].abs() <= t["pae"].abs() + 1e-12).all()


@pytest.mark.skipif(not HAS_DATA, reason="raw data not downloaded")
def test_loader_rules_on_real_data():
    d, logs = data.load([2025], drop_heaves=True)
    assert set(d["made"].unique()) <= {0, 1}
    assert d["distance_ft"].max() <= data.HEAVE_FT
    assert logs[2025]["regular_season_games"] == 1230
    assert not d.duplicated(["game_id", "order_index"]).any()
