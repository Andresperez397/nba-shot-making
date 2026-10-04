"""Data audit (run before any outcome model). Writes reports/tables/data_audit.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shots import data  # noqa: E402

out: dict = {}

# 1. Completeness: every regular-season game on the official schedule has play-by-play.
sched = {}
for s in data.SEASONS:
    sc = pd.read_parquet(data.RAW / f"nba_schedule_{s}.parquet")
    pbp_games = pd.read_parquet(data.RAW / f"nba_play_by_play_{s}.parquet", columns=["game_id"])["game_id"]
    reg = set(sc.loc[sc["season_type"] == "regular-season", "game_id"])
    pbp_reg = set(pbp_games[pbp_games.str[:3] == "002"])
    sched[s] = {"scheduled": len(reg), "with_pbp": len(reg & pbp_reg), "missing": sorted(reg - pbp_reg),
                "pbp_not_scheduled": sorted(pbp_reg - reg)}
out["schedule"] = sched

# 2. Shots, cleaning log and outcome.
d, logs = data.load()
out["load_log"] = logs
out["shots"] = int(len(d))
out["make_rate_by_season"] = d.groupby("season")["made"].mean().round(4).to_dict()
out["players"] = int(d["person_id"].nunique())

# 3. Locations: shots placed exactly at the basket, by family and season (tips/putbacks from 2020-21).
at_rim0 = (d["x_ft"] == 0) & (d["y_ft"] == 0)
out["exact_basket_location_share_by_season"] = at_rim0.groupby(d["season"]).mean().round(4).to_dict()
out["exact_basket_location_by_family_2021_on"] = (
    d.loc[at_rim0 & (d["season"] >= 2021), "family"].value_counts().to_dict())
out["three_inside_22ft"] = int(((d["is_three"] == 1) & (d["distance_ft"] < 21.5)).sum())
out["two_beyond_24ft"] = int(((d["is_three"] == 0) & (d["distance_ft"] > 24.5)).sum())
out["beyond_half_court_40ft"] = int((d["distance_ft"] > 40).sum())

# 4. Shot families: shares by season (label drift) and make rates (no family is all-make or all-miss).
out["family_share_by_season"] = (pd.crosstab(d["season"], d["family"], normalize="index").round(4)
                                 .to_dict(orient="index"))
fam_rate = d.groupby("sub_type")["made"].agg(["size", "mean"])
out["sub_types_all_make_or_all_miss_n50"] = fam_rate[(fam_rate["size"] >= 50) &
                                                     ((fam_rate["mean"] < 0.02) | (fam_rate["mean"] > 0.98))
                                                     ].index.tolist()
out["blank_sub_type"] = int((d["sub_type"].fillna("").str.strip() == "").sum())

# 5. Pre-shot score: the margin must start at zero and be the negative of the opponent's view.
first = d.sort_values(["game_id", "order_index"]).groupby("game_id").head(1)
out["first_shot_margin_zero_share"] = float((first["score_margin"] == 0).mean())
out["score_margin_range"] = [float(d["score_margin"].min()), float(d["score_margin"].max())]

# 6. Volume per player-season (sample sizes for skill estimates).
ps = d.groupby(["season", "person_id"]).size()
out["player_seasons"] = int(len(ps))
out["player_season_fga_quantiles"] = ps.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_dict()
out["player_seasons_200plus"] = int((ps >= 200).sum())

out["leaky_columns_never_inputs"] = data.LEAKY
assert not any(data.is_leaky_name(c) for c in data.FEATURES)

(ROOT / "reports" / "tables").mkdir(parents=True, exist_ok=True)
with open(ROOT / "reports" / "tables" / "data_audit.json", "w") as f:
    json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
for k in ["shots", "players", "make_rate_by_season", "exact_basket_location_share_by_season",
          "exact_basket_location_by_family_2021_on", "three_inside_22ft", "two_beyond_24ft",
          "beyond_half_court_40ft", "sub_types_all_make_or_all_miss_n50", "blank_sub_type",
          "first_shot_margin_zero_share", "score_margin_range", "player_seasons",
          "player_season_fga_quantiles", "player_seasons_200plus"]:
    print(k, out[k])
for s, v in sched.items():
    print(s, v["scheduled"], v["with_pbp"], len(v["missing"]), len(v["pbp_not_scheduled"]))
print(pd.DataFrame(out["family_share_by_season"]).T)
