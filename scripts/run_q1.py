"""Q1 (ANALYSIS_PLAN.md): expected-make models on held-out seasons, rolling origin.

Each test season's predictions are checkpointed in data/derived/ so an interrupted run resumes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shots import data, metrics, models  # noqa: E402

TEST_SEASONS = list(range(2020, 2027))
SPECS = [("M0 constant", None, "const"), ("M1 distance", None, "distance"),
         ("M2 logit", "M2", "logit"), ("M2 hgb", "M2", "hgb"),
         ("M3 logit", "M3", "logit"), ("M3 hgb", "M3", "hgb"),
         ("M4 logit", "M4", "logit"), ("M4 hgb", "M4", "hgb")]
MODELS = [s[0] for s in SPECS]
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "reports" / "tables"
KEEP = ["game_id", "order_index", "season", "person_id", "player_name", "team_id", "made", "shot_value",
        "family"]


def run_season(d: pd.DataFrame, s: int) -> None:
    ck = DERIVED / f"q1_season_{s}.parquet"
    if ck.exists():
        return
    t0 = time.time()
    train, test = d[d["season"] < s], d[d["season"] == s]
    preds = test[KEEP].copy()
    preds["new_player"] = (~test["person_id"].isin(train["person_id"])).astype(int)
    tuning = {}
    for name, block, learner in SPECS:
        p, tun = models.fit_predict(train, test, block, learner)
        preds[name] = p
        tuning[name] = tun
        print(s, name, round(metrics.logloss(test["made"], p), 4), flush=True)
    preds.to_parquet(ck, index=False)
    with open(DERIVED / f"q1_tuning_{s}.json", "w") as f:
        json.dump({"tuning": tuning, "minutes": (time.time() - t0) / 60}, f, indent=2)


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d, logs = data.load(drop_heaves=True)
    with open(OUT / "q1_load_log.json", "w") as f:
        json.dump(logs, f, indent=2)
    for s in TEST_SEASONS:
        run_season(d, s)

    P = pd.concat([pd.read_parquet(DERIVED / f"q1_season_{s}.parquet") for s in TEST_SEASONS],
                  ignore_index=True)
    tuning = {s: json.load(open(DERIVED / f"q1_tuning_{s}.json"))["tuning"] for s in TEST_SEASONS}
    with open(OUT / "q1_tuning.json", "w") as f:
        json.dump(tuning, f, indent=2)

    y = P["made"].to_numpy()
    by_season = pd.DataFrame([{"season": s, "model": m, "n": int((P["season"] == s).sum()),
                               **metrics.score(y[P["season"] == s], P.loc[P["season"] == s, m])}
                              for s in TEST_SEASONS for m in MODELS])
    by_season.to_csv(OUT / "q1_by_season.csv", index=False)

    w = metrics.boot_weights(P["game_id"], P["season"], n_boot=1000, seed=0)
    ll = {m: metrics.logloss_i(y, P[m]) for m in MODELS}
    bs = {m: metrics.boot_mean(ll[m], P["game_id"], w) for m in MODELS}
    rows = []
    for m in MODELS:
        br = metrics.boot_mean(metrics.brier_i(y, P[m]), P["game_id"], w)
        rows.append({"model": m, "n": len(P), **metrics.score(y, P[m]),
                     "logloss_lo": metrics.ci(bs[m])[0], "logloss_hi": metrics.ci(bs[m])[1],
                     "brier_lo": metrics.ci(br)[0], "brier_hi": metrics.ci(br)[1]})
    pooled = pd.DataFrame(rows)

    def best_at(b):
        cand = [f"{b} logit", f"{b} hgb"]
        return min(cand, key=lambda m: pooled.set_index("model").loc[m, "logloss"])

    best = {"M0": "M0 constant", "M1": "M1 distance", "M2": best_at("M2"), "M3": best_at("M3"),
            "M4": best_at("M4")}

    def gain(label, m, ref):
        dlt = bs[ref] - bs[m]
        pt = ll[ref].mean() - ll[m].mean()
        return {"comparison": label, "model": m, "reference": ref, "logloss_reduction": pt,
                "ci_lo": metrics.ci(dlt)[0], "ci_hi": metrics.ci(dlt)[1],
                "adds_information": bool(metrics.ci(dlt)[0] > 0)}

    gains = pd.DataFrame([
        gain("distance over constant", best["M1"], best["M0"]),
        gain("full location over distance", best["M2"], best["M1"]),
        gain("shot type over location", best["M3"], best["M2"]),
        gain("context over shot type", best["M4"], best["M3"]),
        gain("hgb minus logit at M2 (positive = hgb better)", "M2 hgb", "M2 logit"),
        gain("hgb minus logit at M3 (positive = hgb better)", "M3 hgb", "M3 logit"),
        gain("hgb minus logit at M4 (positive = hgb better)", "M4 hgb", "M4 logit"),
    ])
    gains.to_csv(OUT / "q1_block_gains.csv", index=False)

    # xFG model for Q2-Q4: highest block that adds information; logistic unless hgb is clearly better.
    top = "M2"
    for b, lab in (("M3", "shot type over location"), ("M4", "context over shot type")):
        if gains.set_index("comparison").loc[lab, "adds_information"] and top == {"M3": "M2", "M4": "M3"}[b]:
            top = b
    g = gains.set_index("comparison").loc[f"hgb minus logit at {top} (positive = hgb better)"]
    xfg = f"{top} hgb" if g["ci_lo"] > 0 else f"{top} logit"

    # Shots by players the model has never seen.
    new = P["new_player"] == 1
    pooled["logloss_new_players"] = [metrics.logloss(y[new], P.loc[new, m]) for m in MODELS]
    pooled["logloss_returning_players"] = [metrics.logloss(y[~new], P.loc[~new, m]) for m in MODELS]
    pooled["n_new_players_shots"] = int(new.sum())
    pooled.to_csv(OUT / "q1_pooled.csv", index=False)
    with open(OUT / "q1_summary.json", "w") as f:
        json.dump({"best_by_block": best, "xfg_model": xfg, "top_block": top}, f, indent=2)
    xfg_table = P[KEEP + ["new_player", xfg]].rename(columns={xfg: "xfg"})
    xfg_table.to_parquet(DERIVED / "xfg.parquet", index=False)
    print(pooled[["model", "logloss", "logloss_lo", "logloss_hi", "auc", "cal_slope", "ece",
                  "logloss_new_players"]].round(4).to_string())
    print(gains.round(5).to_string())
    print("xFG model:", xfg)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
