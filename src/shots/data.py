"""Load NBA play-by-play into one row per regular-season field-goal attempt with pre-shot inputs only.

Outcome: `made` (1 = field goal made). Leakage traps in the source, none of which are inputs:
the description text ("MISS", "(2 PTS)", "(... AST)", "BLOCK"), action_type ("Made Shot" /
"Missed Shot"), points_total, is_made_shot, and the score fields on the shot's own row (a made
shot's row already includes its points). The pre-shot score is carried forward from earlier events.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SEASONS = list(range(2018, 2027))  # 2017-18 through 2025-26: one shot-type taxonomy and one rim geometry

PBP_COLS = ["game_id", "order_index", "period", "seconds_remaining", "team_id", "person_id",
            "player_name", "x_legacy", "y_legacy", "shot_distance", "shot_result", "is_field_goal",
            "score_home", "score_away", "location", "sub_type", "shot_value"]
LEAKY = ["description", "action_type", "points_total", "is_made_shot", "is_missed_shot", "shot_result",
         "score_home", "score_away"]

# Shot families, harmonized across label changes (bank variants fold into their base type).
FAMILY_RULES = [
    ("tip_putback", r"\b(?:tip|putback)\b"),
    ("dunk", r"\bdunk\b"),
    ("hook", r"\bhook\b"),
    ("floater", r"\bfloat"),
    ("layup", r"\blayup|finger roll"),
    ("jumper", r"jump|fadeaway|turnaround|bank shot|pull-?up|step back"),
]
MODIFIERS = {
    "drive": r"\bdriving\b",
    "running": r"\brunning\b",
    "cutting": r"\bcutting\b",
    "pullup": r"pull-?up",
    "stepback": r"step back",
    "fadeaway": r"fadeaway",
    "turnaround": r"turnaround",
    "alley_oop": r"alley oop",
    "reverse": r"\breverse\b",
}
FAMILIES = [f for f, _ in FAMILY_RULES]

CONTEXT = ["period_n", "seconds_left_period", "score_margin", "home"]
LOCATION = ["x_ft", "y_ft", "distance_ft", "angle_deg", "is_three", "corner_three"]
TYPE = [f"fam_{f}" for f in FAMILIES] + [f"mod_{m}" for m in MODIFIERS]
FEATURES = LOCATION + TYPE + CONTEXT


def shot_family(sub_type: pd.Series) -> pd.Series:
    s = sub_type.fillna("").str.lower()
    out = pd.Series("other", index=s.index)
    for fam, pat in reversed(FAMILY_RULES):  # earlier rules take precedence
        out[s.str.contains(pat, regex=True)] = fam
    return out


def pre_shot_score(g: pd.DataFrame) -> pd.DataFrame:
    """Score before each event: carry the last recorded score forward, then shift one event."""
    home = pd.to_numeric(g["score_home"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    away = pd.to_numeric(g["score_away"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    home = home.ffill().shift(1).fillna(0)
    away = away.ffill().shift(1).fillna(0)
    return pd.DataFrame({"pre_home": home, "pre_away": away}, index=g.index)


def load_season(season: int) -> tuple[pd.DataFrame, dict]:
    p = pd.read_parquet(RAW / f"nba_play_by_play_{season}.parquet", columns=PBP_COLS)
    log = {"events": len(p)}
    p = p[p["game_id"].str[:3] == "002"]  # regular season only
    log["regular_season_events"] = len(p)
    log["regular_season_games"] = int(p["game_id"].nunique())
    p = p.sort_values(["game_id", "order_index"])
    scores = p.groupby("game_id", group_keys=False)[["score_home", "score_away"]].apply(pre_shot_score)
    p = p.join(scores)
    fg = p[p["is_field_goal"] == 1].copy()
    log["fga"] = len(fg)
    bad = ~fg["shot_result"].isin(["Made", "Missed"]) | ~fg["location"].isin(["h", "v"])
    log["dropped_missing_result_or_side"] = int(bad.sum())
    fg = fg[~bad]

    fg["made"] = (fg["shot_result"] == "Made").astype(int)
    fg["season"] = season
    fg["home"] = (fg["location"] == "h").astype(int)
    own = np.where(fg["home"] == 1, fg["pre_home"], fg["pre_away"])
    opp = np.where(fg["home"] == 1, fg["pre_away"], fg["pre_home"])
    fg["score_margin"] = own - opp
    fg["period_n"] = fg["period"].clip(upper=5)  # every overtime as 5
    fg["seconds_left_period"] = fg["seconds_remaining"]
    # Legacy coordinates are tenths of a foot with the basket at (0, 0); y grows toward half court.
    fg["x_ft"] = fg["x_legacy"] / 10
    fg["y_ft"] = fg["y_legacy"] / 10
    fg["distance_ft"] = np.hypot(fg["x_ft"], fg["y_ft"])
    fg["angle_deg"] = np.degrees(np.arctan2(fg["x_ft"].abs(), fg["y_ft"]))
    fg["is_three"] = (fg["shot_value"] == 3).astype(int)
    fg["corner_three"] = ((fg["is_three"] == 1) & (fg["y_ft"] < 9.25)).astype(int)
    fg["family"] = shot_family(fg["sub_type"])
    for f in FAMILIES:
        fg[f"fam_{f}"] = (fg["family"] == f).astype(int)
    st = fg["sub_type"].fillna("").str.lower()
    for m, pat in MODIFIERS.items():
        fg[f"mod_{m}"] = st.str.contains(pat, regex=True).astype(int)
    keep = ["game_id", "order_index", "season", "team_id", "person_id", "player_name", "sub_type",
            "family", "made", "shot_value"] + FEATURES
    return fg[keep].reset_index(drop=True), log


HEAVE_FT = 40  # attempts from beyond this distance are end-of-period heaves (ANALYSIS_PLAN.md)


def load(seasons: list[int] | None = None, drop_heaves: bool = False) -> tuple[pd.DataFrame, dict]:
    parts, logs = [], {}
    for s in seasons or SEASONS:
        d, log = load_season(s)
        if drop_heaves:
            heave = d["distance_ft"] > HEAVE_FT
            log["dropped_heaves"] = int(heave.sum())
            d = d[~heave].reset_index(drop=True)
        log["analysis_shots"] = len(d)
        parts.append(d)
        logs[s] = log
    return pd.concat(parts, ignore_index=True), logs


def game_number(d: pd.DataFrame) -> pd.Series:
    """Order of each game within a season (by game id), used for odd/even split halves."""
    ids = d[["season", "game_id"]].drop_duplicates().sort_values(["season", "game_id"])
    ids["game_n"] = ids.groupby("season").cumcount() + 1
    return d[["season", "game_id"]].merge(ids, on=["season", "game_id"], how="left")["game_n"]


def is_leaky_name(name: str) -> bool:
    return name in LEAKY or bool(re.search(r"desc|assist|block|pts", name))
