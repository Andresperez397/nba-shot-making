# Data audit

Written 2026-10-03, before any model of makes and misses was fit. Every number here comes from `scripts/data_audit.py` (saved to `reports/tables/data_audit.json`).

## Source and license

- **Play-by-play and schedules:** NBA Stats (stats.nba.com), as republished by the sportsdataverse project (`nba_stats_pbp` and `nba_stats_schedules` releases). The sportsdataverse data repository is MIT-licensed.
- **Who owns the data:** the NBA. This project uses it for non-commercial research. It commits no raw data, only derived summary tables and figures.
- **Pinning:** `scripts/fetch_data.py` downloads the files and checks each one against the SHA-256 in `data/manifest.csv`, because the source updates files in place.

## Seasons

The analysis covers 2015-16 through 2025-26, regular season only. Seasons are named by their ending year (2016 = 2015-16), as in the source.

- **Why the window starts in 2015-16:** the NBA recoded its shot-type labels that season. "Tip Shot", "Slam Dunk", "Jump Hook" and "Running Bank" disappear, and "Tip Layup", "Cutting Layup", "Driving Floating Jump Shot" and others appear. Mixing taxonomies would let a model learn label changes instead of shooting.
- **What is excluded:** playoffs, play-in, preseason, All-Star and Cup-final games. They are different contexts, and only the regular season has a balanced schedule.

## Completeness

- **Games:** every regular-season game on the official schedule has play-by-play, and no play-by-play game is missing from the schedule. That is 1,230 games a season, except 1,059 in 2019-20 and 1,080 in 2020-21, when COVID shortened both seasons.
- **Shots:** 2,319,756 field-goal attempts by 1,544 players.
  - Every attempt has a result (made or missed) and a shooting side (home or away).
  - 81 attempts have a blank shot-type label.

## The outcome and what cannot be an input

The outcome is `made` (1 = field goal made). Make rates run from 45.2% (2015-16) to 47.5% (2022-23).

The source has several fields that reveal the result, so none of them is an input:
- **Description text**, which contains "MISS", "(2 PTS)", "(... AST)" and "BLOCK".
- **Assists and blocks:** an assister exists only on makes and a blocker only on misses.
- **Outcome fields:** `action_type` ("Made Shot" / "Missed Shot"), `points_total` and `is_made_shot`.
- **The score on the shot's own row:** a made shot's row already includes its points.

**Pre-shot score.** The score before each shot is rebuilt from earlier events only: the last recorded score is carried forward, then shifted by one event. On a test game, the margin after a made three by the home team is −3 for the next away shot, and so on through the sequence. 96.6% of games start their first shot at a margin of zero. The rest had free throws before the first field goal.

## Inputs

| Group | Fields |
|---|---|
| Location | x and y (feet, basket at the origin), distance, angle, 3-point attempt, corner 3 |
| Shot type | family (dunk, layup, tip or putback, hook, floater, jumper) and modifiers (driving, running, cutting, pull-up, step-back, fadeaway, turnaround, alley-oop, reverse) |
| Context | period (all overtimes pooled), seconds left in the period, score margin before the shot (shooting team's view), home |

**How families are built.** Families come from the shot-type label by rules that fold bank-shot variants into their base type, because bank labels come and go between 2017-18 and 2021-22. No label with at least 50 attempts is all makes or all misses.

**Shot-type labels are assigned by scorers after the play.** That is standard in public shot models, but it is a known softness: for example, a blocked dunk attempt may be scored as a layup. The README states this limitation.

## Findings that affect the analysis

1. **Tip-ins and putbacks sit exactly at the basket from 2020-21.** About 2.7% of attempts a season have coordinates of exactly (0, 0) from 2020-21 on, against 0.0–0.04% before. 34,343 of the 35,023 such attempts are tips or putbacks, which are evidently no longer charted for location. For that family, location carries no information after 2020-21. The shot-type family carries it instead.
2. **The shot mix drifts.** Floaters rise from 3.8% of attempts (2015-16) to 8.7–9.5% (2022-23 to 2025-26), and jumpers fall from 62% to 55–56%. Some of this is real, some is labelling. Temporal validation shows whether a model trained on earlier seasons still works.
3. **Heaves:** 5,417 attempts come from beyond 40 feet, mostly end-of-quarter heaves. They are desperation shots outside the normal offense, so they are excluded from the analysis (the rule is fixed in the plan).
4. **Line consistency:** only 7 three-point attempts are inside 21.5 feet and 1 two-point attempt is beyond 24.5 feet, so the coordinates and the 3-point flag agree.
5. **Sample sizes:** there are 5,938 player-seasons. The median player-season has 298 attempts (10th percentile 19, 90th percentile 913), and 3,603 have at least 200. Shot-making estimates for low-volume players need shrinkage.

## Leakage risks carried into the analysis plan

1. **Fields that reveal the result:** the description text, assists, blocks and the shot row's own score are never inputs. A test enforces this.
2. **Expected makes must be out of sample** for the season they score. If the expected-make model saw a player's own shots, it would absorb part of that player's skill. Seasons used for shot-making are scored only by models trained on earlier seasons.
3. **Random splits:** they would put the same player-season and game on both sides. Validation is by season, and is also reported for players the model has never seen.
