"""Report figures from reports/tables/ and data/derived/ (run after run_q1.py and run_skill.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG, DERIVED = ROOT / "reports" / "tables", ROOT / "reports" / "figures", ROOT / "data" / "derived"
BLUE, ORANGE, GREY = "#1d6fb8", "#e8590c", "#6b7480"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
LABELS = {"M0": "Constant", "M1": "Distance only", "M2": "Full location", "M3": "+ shot type",
          "M4": "+ game context"}


def ladder():
    t = pd.read_csv(TAB / "q1_pooled.csv")
    t["block"] = t["model"].str[:2]
    t["learner"] = np.where(t["model"].str.contains("hgb"), "Gradient boosting",
                            np.where(t["model"].str.contains("logit"), "Logistic", "Baseline"))
    order = list(LABELS)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    styles = (("Baseline", GREY, 0), ("Logistic", BLUE, 0.15), ("Gradient boosting", ORANGE, -0.15))
    for learner, c, off in styles:
        s = t[t["learner"] == learner]
        y = np.array([order.index(b) for b in s["block"]]) + off
        ax.hlines(y, s["logloss_lo"], s["logloss_hi"], color=c, lw=2)
        ax.plot(s["logloss"], y, "o", color=c, ms=7, label=learner)
    ax.set_yticks(range(len(order)), [LABELS[b] for b in order])
    ax.set_xlabel("Log loss on unseen seasons, 2017-18 to 2025-26 (lower is better)")
    ax.set_title("What it takes to predict a make")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_information_ladder.png", dpi=200)
    plt.close(fig)


def calibration():
    x = pd.read_parquet(DERIVED / "xfg.parquet")
    b = pd.qcut(x["xfg"].rank(method="first"), 20, labels=False)
    g = x.groupby(b).agg(p=("xfg", "mean"), y=("made", "mean"))
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], ls="--", color=GRID)
    ax.plot(g["p"], g["y"], "o", color=ORANGE)
    ax.set_xlabel("Expected make probability")
    ax.set_ylabel("Observed make rate")
    ax.set_title("Expected makes are calibrated")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_calibration.png", dpi=200)
    plt.close(fig)


def stability():
    s = json.load(open(TAB / "skill_summary.json"))
    sh = s["q3a_split_half"]
    names = {"xpps": "Shot selection\n(expected pts/shot)", "pps": "Results\n(pts/shot)",
             "pae": "Shot-making\n(pts above expected)"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, k in enumerate(("xpps", "pps", "pae")):
        r, (lo, hi) = sh[k]["r"], sh[k]["ci"]
        ax.barh(i, r, color=[BLUE, GREY, ORANGE][i], height=0.55)
        ax.hlines(i, lo, hi, color=INK, lw=1.5)
        ax.text(hi + 0.02, i, f"r = {r:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(3), [names[k] for k in ("xpps", "pps", "pae")])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Split-half correlation (odd vs even games, same season)")
    ax.set_title("A shooter's diet is far more stable than their making")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_split_half.png", dpi=200)
    plt.close(fig)


def year_to_year():
    ps = pd.read_csv(TAB / "q2_player_seasons.csv")
    u = ps[ps["n"] >= 200]
    pair = u.merge(u.assign(season=u["season"] - 1), on=["person_id", "season"], suffixes=("", "_next"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, col, title, c in ((axes[0], "pae", "Raw points above expected", GREY),
                              (axes[1], "pae_shrunk", "Shrunk points above expected", ORANGE)):
        ax.scatter(100 * pair[col], 100 * pair["pae_next"], s=6, alpha=0.35, color=c, edgecolor="none")
        b = np.polyfit(pair[col], pair["pae_next"], 1, w=pair["n_next"])
        xs = np.linspace(pair[col].min(), pair[col].max(), 50)
        ax.plot(100 * xs, 100 * np.polyval(b, xs), color=INK, lw=1.5)
        ax.axhline(0, color=GRID)
        ax.axvline(0, color=GRID)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("This season (points per 100 shots)")
    axes[0].set_ylabel("Next season's raw points above expected\n(points per 100 shots)")
    fig.suptitle("Shot-making carries over, but only once it is shrunk", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_year_to_year.png", dpi=200)
    plt.close(fig)


def teams():
    s = json.load(open(TAB / "skill_summary.json"))["q4_team_year_to_year"]
    keys = [("xpps_off", "Offense: shot selection"), ("pae_off", "Offense: shot-making"),
            ("xpps_def", "Defense: selection allowed"), ("pae_def", "Defense: making allowed")]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (k, _lab) in enumerate(keys):
        r, (lo, hi) = s[k]["r"], s[k]["ci"]
        ax.barh(i, r, color=[BLUE, ORANGE, BLUE, ORANGE][i], height=0.55, alpha=0.9 if i < 2 else 0.55)
        ax.hlines(i, lo, hi, color=INK, lw=1.5)
        ax.text(max(hi, 0) + 0.02, i, f"r = {r:.2f}", va="center", fontsize=9)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(4), [lab for _, lab in keys])
    ax.invert_yaxis()
    ax.set_xlim(min(-0.2, ax.get_xlim()[0]), 1.05)
    ax.set_xlabel("Same team, consecutive seasons: correlation")
    ax.set_title("Teams control which shots they take and allow more than whether they go in")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_teams.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ladder()
    calibration()
    stability()
    year_to_year()
    teams()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
