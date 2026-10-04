"""Q2-Q4 (ANALYSIS_PLAN.md): shot-making, its stability, and team shot diets. Run after run_q1.py.

Expected makes (xFG) for each season come from the Q1 model trained on earlier seasons only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shots import data, metrics, skill  # noqa: E402

DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "reports" / "tables"
MIN_SPLIT = 200
N_BOOT = 1000


def corr_ci(df, a, b, cluster):
    r = float(np.corrcoef(df[a], df[b])[0, 1])
    bs = skill.cluster_boot(df, cluster, lambda x: np.corrcoef(x[a], x[b])[0, 1], n_boot=N_BOOT)
    return {"r": r, "ci": list(metrics.ci(bs)), "n": int(len(df))}


def main() -> None:
    x = pd.read_parquet(DERIVED / "xfg.parquet")
    x["game_n"] = data.game_number(x).to_numpy()
    out: dict = {}

    # Q2: player-season shot-making, raw and shrunk.
    ps = skill.shrink(skill.pae_table(x, ["season", "person_id"]))
    names = x.groupby("person_id")["player_name"].agg(lambda s: s.mode().iloc[0])
    ps["player_name"] = ps["person_id"].map(names)
    ps.to_csv(OUT / "q2_player_seasons.csv", index=False)
    out["tau_by_season"] = ps.groupby("season")["tau2"].first().pow(0.5).round(4).to_dict()
    big = ps[ps["n"] >= 500]
    out["players_500_plus"] = int(len(big))
    out["share_of_raw_pae_variance_that_is_signal_500_plus"] = float(
        (big["tau2"] / (big["tau2"] + big["v"])).mean())
    lead = ps[ps["n"] >= 300].sort_values("pae_shrunk", ascending=False)
    lead.head(15).to_csv(OUT / "q2_top_shotmakers.csv", index=False)
    lead.tail(15).to_csv(OUT / "q2_bottom_shotmakers.csv", index=False)

    # Q3a: split-half reliability of making, selection and results.
    sh = skill.split_half(x, MIN_SPLIT)
    sh["person_season"] = sh["person_id"].astype(str) + "_" + sh["season"].astype(str)
    q3a = {}
    for k in ("pae", "xpps", "pps"):
        c = corr_ci(sh, f"{k}_odd", f"{k}_even", "person_id")
        c["spearman_brown"] = skill.spearman_brown(c["r"])
        q3a[k] = c
    out["q3a_split_half"] = q3a

    # Q3b: next season's shot-making from this season's (zero, raw, shrunk).
    pair = skill.consecutive(ps, "person_id", MIN_SPLIT)
    def mse_all(df):
        w = df["n_next"]
        return {k: skill.wmse(df["pae_next"], df[c] if c else 0, w)
                for k, c in (("zero", None), ("raw", "pae"), ("shrunk", "pae_shrunk"))}

    point = mse_all(pair)
    bs = skill.cluster_boot(pair, "person_id", lambda d: list(mse_all(d).values()), n_boot=N_BOOT)
    bs = pd.DataFrame(bs, columns=list(point))
    out["q3b_year_to_year"] = {
        "n_pairs": int(len(pair)), "n_players": int(pair["person_id"].nunique()), "wmse": point,
        "zero_minus_shrunk": {"est": point["zero"] - point["shrunk"],
                              "ci": list(metrics.ci(bs["zero"] - bs["shrunk"]))},
        "raw_minus_shrunk": {"est": point["raw"] - point["shrunk"],
                             "ci": list(metrics.ci(bs["raw"] - bs["shrunk"]))},
        "r_shrunk_vs_next": float(np.corrcoef(pair["pae_shrunk"], pair["pae_next"])[0, 1]),
        "r_xpps_vs_next": float(np.corrcoef(pair["xpps"], pair["xpps_next"])[0, 1]),
    }

    # Q3c: next season's points per shot: raw PPS vs diet + shrunk making (leave-one-season-pair-out WLS).
    def loso(df, cols):
        pred = np.empty(len(df))
        for s in df["season"].unique():
            tr, te = df["season"] != s, df["season"] == s
            X = np.column_stack([np.ones(tr.sum())] + [df.loc[tr, c] for c in cols])
            w = df.loc[tr, "n_next"].to_numpy()
            sw = np.sqrt(w)
            beta = np.linalg.lstsq(X * sw[:, None], df.loc[tr, "pps_next"] * sw, rcond=None)[0]
            Xt = np.column_stack([np.ones(te.sum())] + [df.loc[te, c] for c in cols])
            pred[te.to_numpy()] = Xt @ beta
        return pred

    def q3c_stat(df):
        a = skill.wmse(df["pps_next"], loso(df, ["pps"]), df["n_next"])
        b = skill.wmse(df["pps_next"], loso(df, ["xpps", "pae_shrunk"]), df["n_next"])
        return a, b

    a, b = q3c_stat(pair)
    bs3 = skill.cluster_boot(pair, "person_id", lambda d: np.subtract(*q3c_stat(d)), n_boot=N_BOOT)
    out["q3c_next_pps"] = {"wmse_raw_pps": a, "wmse_diet_plus_making": b, "raw_minus_split": a - b,
                           "ci": list(metrics.ci(bs3))}

    # Q4: team shot diets on offense and defense.
    games = x[["game_id", "team_id"]].drop_duplicates()
    opp = games.merge(games, on="game_id", suffixes=("", "_opp"))
    opp = opp[opp["team_id"] != opp["team_id_opp"]].set_index(["game_id", "team_id"])["team_id_opp"]
    x["def_team"] = opp.reindex(pd.MultiIndex.from_frame(x[["game_id", "team_id"]])).to_numpy()
    off = skill.pae_table(x, ["season", "team_id"]).rename(columns={"team_id": "team"})
    dfn = skill.pae_table(x, ["season", "def_team"]).rename(columns={"def_team": "team"})
    teams = off.merge(dfn, on=["season", "team"], suffixes=("_off", "_def"))
    teams.to_csv(OUT / "q4_team_seasons.csv", index=False)
    tp = teams.merge(teams.assign(season=teams["season"] - 1), on=["team", "season"], suffixes=("", "_next"))
    q4 = {k: corr_ci(tp, k, f"{k}_next", "team") for k in ("xpps_off", "pae_off", "xpps_def", "pae_def")}

    def diff(df):
        return (np.corrcoef(df["xpps_def"], df["xpps_def_next"])[0, 1]
                - np.corrcoef(df["pae_def"], df["pae_def_next"])[0, 1])

    q4["def_selection_minus_making"] = {"est": float(diff(tp)),
                                        "ci": list(metrics.ci(skill.cluster_boot(tp, "team", diff, N_BOOT)))}
    q4["sd"] = {k: float(teams[k].std()) for k in ("xpps_off", "pae_off", "xpps_def", "pae_def")}
    out["q4_team_year_to_year"] = q4

    # Added robustness (DEVIATIONS.md): the same stability tests with location-only expected makes
    # (Q1 M2 boosting), which use no scorer-assigned shot-type label.
    cols = ["game_id", "order_index", "M2 hgb"]
    q1 = pd.concat([pd.read_parquet(DERIVED / f"q1_season_{s}.parquet", columns=cols)
                    for s in sorted(x["season"].unique())])
    xl = x.drop(columns="xfg").merge(q1.rename(columns={"M2 hgb": "xfg"}), on=["game_id", "order_index"])
    assert len(xl) == len(x)
    psl = skill.shrink(skill.pae_table(xl, ["season", "person_id"]))
    shl = skill.split_half(xl, MIN_SPLIT)
    pl = skill.consecutive(psl, "person_id", MIN_SPLIT)
    both = ps.merge(psl, on=["season", "person_id"], suffixes=("", "_loc"))
    out["robustness_location_only_xfg"] = {
        "split_half_r_pae": float(np.corrcoef(shl["pae_odd"], shl["pae_even"])[0, 1]),
        "year_to_year_r_shrunk": float(np.corrcoef(pl["pae_shrunk"], pl["pae_next"])[0, 1]),
        "wmse_zero": skill.wmse(pl["pae_next"], 0, pl["n_next"]),
        "wmse_shrunk": skill.wmse(pl["pae_next"], pl["pae_shrunk"], pl["n_next"]),
        "corr_with_main_pae_shrunk_500_plus": float(np.corrcoef(
            both.loc[both["n"] >= 500, "pae_shrunk"], both.loc[both["n"] >= 500, "pae_shrunk_loc"])[0, 1]),
    }

    with open(OUT / "skill_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
