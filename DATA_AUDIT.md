# Data audit

Written 2026-10-03, before any model of makes and misses was fit. Updated 2026-10-04 for the season window (see "Rim geometry" below and `DEVIATIONS.md`). Every number here comes from `scripts/data_audit.py` (saved to `reports/tables/data_audit.json`).

## Source and license

- **Play-by-play and schedules:** NBA Stats (stats.nba.com), as republished by the sportsdataverse project (`nba_stats_pbp` and `nba_stats_schedules` releases). The sportsdataverse data repository is MIT-licensed.
- **Who owns the data:** the NBA. This project uses it for non-commercial research. It commits no raw data, only derived summary tables and figures.
- **Pinning:** `scripts/fetch_data.py` downloads the files and checks each one against the SHA-256 in `data/manifest.csv`, because the source updates files in place.

## Seasons

The analysis covers 2017-18 through 2025-26, regular season only. Seasons are named by their ending year (2018 = 2017-18), as in the source.

Two breaks in the source set where the window starts:
- **Shot-type labels were recoded in 2015-16.** "Tip Shot", "Slam Dunk", "Jump Hook" and "Running Bank" disappear, and "Tip Layup", "Cutting Layup", "Driving Floating Jump Shot" and others appear. Mixing taxonomies would let a model learn label changes instead of shooting.
- **Rim geometry changed in 2017-18.** This was found after the plan froze; see the dated section below. Before it, rim attempts sit closer to the basket in the coordinates.
- **What is excluded:** playoffs, play-in, preseason, All-Star and Cup-final games. They are different contexts, and only the regular season has a balanced schedule.

## Completeness

- **Games:** every regular-season game on the official schedule has play-by-play, and no play-by-play game is missing from the schedule. That is 1,230 games a season, except 1,059 in 2019-20 and 1,080 in 2020-21, when COVID shortened both seasons.
- **Shots:** 1,901,593 field-goal attempts by 1,384 players. Every attempt has a result (made or missed), a shooting side (home or away) and a shot-type label.

## The outcome and what cannot be an input

The outcome is `made` (1 = field goal made). Make rates run from 46.0% (2017-18) to 47.5% (2022-23).

The source has several fields that reveal the result, so none of them is an input:
- **Description text**, which contains "MISS", "(2 PTS)", "(... AST)" and "BLOCK".
- **Assists and blocks:** an assister exists only on makes and a blocker only on misses.
- **Outcome fields:** `action_type` ("Made Shot" / "Missed Shot"), `points_total` and `is_made_shot`.
- **The score on the shot's own row:** a made shot's row already includes its points.

**Pre-shot score.** The score before each shot is rebuilt from earlier events only: the last recorded score is carried forward, then shifted by one event. On a test game, the margin after a made three by the home team is −3 for the next away shot, and so on through the sequence. 96.4% of games start their first shot at a margin of zero. The rest had free throws before the first field goal.

## Inputs

| Group | Fields |
|---|---|
| Location | x and y (feet, basket at the origin), distance, angle, 3-point attempt, corner 3 |
| Shot type | family (dunk, layup, tip or putback, hook, floater, jumper) and modifiers (driving, running, cutting, pull-up, step-back, fadeaway, turnaround, alley-oop, reverse) |
| Context | period (all overtimes pooled), seconds left in the period, score margin before the shot (shooting team's view), home |

**How families are built.** Families come from the shot-type label by rules that fold bank-shot variants into their base type, because bank labels come and go between 2017-18 and 2021-22. No label with at least 50 attempts is all makes or all misses.

**Shot-type labels are assigned by scorers after the play.** That is standard in public shot models, but it is a known softness: for example, a blocked dunk attempt may be scored as a layup. The README states this limitation.

## Findings that affect the analysis

1. **Tip-ins and putbacks sit exactly at the basket from 2020-21.** About 2.7% of attempts a season have coordinates of exactly (0, 0) from 2020-21 on, against 0.03–0.04% before. 34,343 of the 35,023 such attempts are tips or putbacks, which are evidently no longer charted for location. For that family, location carries no information after 2020-21. The shot-type family carries it instead.
2. **The shot mix drifts.** Floaters rise from 5.7% of attempts (2017-18) to 8.4–9.5% (2022-23 to 2025-26), and jumpers fall from 58% to 55–56%. Some of this is real, some is labelling. Temporal validation shows whether a model trained on earlier seasons still works.
3. **Heaves:** 4,220 attempts come from beyond 40 feet, mostly end-of-quarter heaves. They are desperation shots outside the normal offense, so they are excluded from the analysis (the rule is fixed in the plan).
4. **Line consistency:** only 6 three-point attempts are inside 21.5 feet and no two-point attempt is beyond 24.5 feet, so the coordinates and the 3-point flag agree.
5. **Sample sizes:** there are 4,978 player-seasons. The median player-season has 284 attempts (10th percentile 18, 90th percentile 904), and 2,952 have at least 200. Shot-making estimates for low-volume players need shrinkage.

## Leakage risks carried into the analysis plan

1. **Fields that reveal the result:** the description text, assists, blocks and the shot row's own score are never inputs. A test enforces this.
2. **Expected makes must be out of sample** for the season they score. If the expected-make model saw a player's own shots, it would absorb part of that player's skill. Seasons used for shot-making are scored only by models trained on earlier seasons.
3. **Random splits:** they would put the same player-season and game on both sides. Validation is by season, and is also reported for players the model has never seen.

## Rim geometry (found 2026-10-04, after the plan froze)

**How it was found.** The first Q1 test season (2017-18) gave an odd result. Gradient boosting with full location scored worse than a distance-only model (log loss 0.670 against 0.656), although it beat distance-only on 2016-17 (0.650 against 0.654). Dropping one location input at a time showed the y coordinate was the cause.

**What changed.** Rim attempts are charted about 0.4 feet further from the basket from 2017-18 on:

| Season | Median y of non-tip attempts within 4 ft | Make rate, 1–2 ft | Share of attempts 1–2 ft |
|---|---|---|---|
| 2015-16 | 0.3 ft | 58.3% | 10.9% |
| 2016-17 | 0.7 ft | 58.0% | 10.6% |
| 2017-18 | 1.1 ft | 68.2% | 11.9% |
| 2018-19 | 1.1 ft | 68.1% | 12.3% |
| 2019-20 | 1.1 ft | 68.6% | 11.6% |
| 2020-21 to 2025-26 | 1.1–1.2 ft | 70.3–73.5% | 8.7–9.4% |

The make rate at 1–2 feet jumps by 10 points in one season, and nothing about shooting explains that. It is a change in where shots are placed.

**Why it matters.** Trees trained on the earlier geometry put sharp cutoffs at the wrong places. The geometry is stable from 2017-18 (the 2020-21 drop in the 1–2 ft share is tips moving to the basket point, finding 1).

**Response.** The analysis window now starts in 2017-18. That was preferred over shifting coordinates by hand, because the size of the shift is not documented. The plan change is logged in `DEVIATIONS.md`, and the table above is produced by `scripts/data_audit.py`.
