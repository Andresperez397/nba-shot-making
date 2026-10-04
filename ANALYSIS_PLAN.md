# Analysis plan (frozen before any outcome model is fit)

Written 2026-10-03 after the data audit (`DATA_AUDIT.md`) and before any model of makes and misses was fit. Any later change is logged in `DEVIATIONS.md` with a date and reason.

**Unit:** one regular-season field-goal attempt, 2015-16 to 2025-26. Attempts from beyond 40 feet (heaves) are excluded.
**Outcome:** `made`.
**Inputs:** only the location, shot-type and context fields listed in `DATA_AUDIT.md`. Player identity is never an input to the expected-make model.

## Q1. How well can a make be predicted from where and how the shot was taken?

**Validation:** rolling origin by season.
- **Test seasons:** 2017-18 to 2025-26, nine in all.
- **Training:** for test season *s*, every earlier season from 2015-16.
- **Tuning:** each setting is fit on seasons before *s*−1 and scored on season *s*−1. The chosen setting is then refit on all seasons before *s*. Both learners are tuned this way.
- **Unseen players:** results are also reported on test-season shots by players with no attempts in that model's training seasons.

**Nested information blocks**

| Block | Adds | Learners |
|---|---|---|
| M0 | Nothing: the training-set make rate | constant |
| M1 | Distance only | logistic regression on a natural spline of distance (df 6) |
| M2 | Full location: x, y, distance, angle, 3-point, corner 3 | logistic, gradient boosting |
| M3 | + shot type (family and modifiers) | logistic, gradient boosting |
| M4 | + context (period, seconds left, score margin, home) | logistic, gradient boosting |

**Logistic regression:** L2-penalized, with C tuned on the holdout season over 10 log-spaced values from 1e-3 to 1e2. It uses this fixed basis:
- natural splines of distance (df 6) and angle (df 4), and x as a natural spline (df 4)
- 3-point and corner-3 flags
- family indicators interacted with the distance spline
- modifier indicators
- context: period as a factor, a seconds-left spline (df 4), the score margin clipped to ±25 as a spline (df 4), and home.

Inputs are standardized.

**Gradient boosting:** scikit-learn `HistGradientBoostingClassifier` with `early_stopping=False`.
- **Grid:** 12 settings, max leaf nodes {15, 31, 63} × learning rate {0.05, 0.1} × min samples per leaf {50, 500}.
- **Iterations:** 100 to 1,000, chosen on the holdout season by scoring every 100 iterations. The chosen number is reused when refitting.
- **Seed:** fixed.

**Metrics**, per test season and pooled (shot-weighted):
- **Primary:** log loss.
- **Secondary:** Brier score, AUC, and calibration (logistic recalibration intercept and slope, expected calibration error over 10 equal-count bins).
- **Clipping:** probabilities are clipped to [0.001, 0.999].

**Uncertainty:** game-cluster bootstrap, 1,000 resamples within each test season, paired across models.

**Decision rules (pre-specified)**
- **Information gain:** a block adds information if its pooled log-loss reduction against the previous block has a 95% interval above zero. The better learner is used at each block.
- **The expected-make model (xFG)** used in Q2–Q4 is the better learner at the highest block that adds information. If boosting's advantage over logistic at that block has an interval that includes zero, logistic is used.

## Q2. Shot-making: how many points does each shooter add beyond what their shots should yield?

**Points above expected (PAE):** for each test-season player-season, PAE = mean over the player's attempts of shot value × (made − xFG), in points per shot. xFG for season *s* always comes from the model trained on earlier seasons only, never from one that saw the player's own shots that season.

**Empirical-Bayes shrinkage** (normal–normal, estimated separately for each season):
- **Sampling variance** of a player's raw PAE: *v* = Σ value² · xFG(1 − xFG) / *n*².
- **Between-player variance:** τ² = max(0, weighted variance of raw PAE − weighted mean of *v*). The weights are attempts, among player-seasons with at least 50 attempts.
- **Shrunk PAE** = raw PAE × τ² / (τ² + *v*), with posterior SD √(τ² *v* / (τ² + *v*)).
- **Reporting:** shooters are ranked by shrunk PAE with 95% intervals. Raw rankings are shown alongside to illustrate what shrinkage changes.

## Q3. Is shot-making a skill, and is it more or less stable than shot selection?

**(a) Split-half reliability within a season:** for player-seasons with at least 200 attempts.
- Each player's shots are split by odd and even game number within the season.
- **Correlations across player-seasons:** raw PAE (making), mean xFG points per shot (selection, "xPPS") and raw points per shot (results).
- **Correction:** Spearman–Brown.
- **Intervals:** bootstrapped by player.

**(b) Year to year:** for players with at least 200 attempts in both seasons *s* and *s*+1, with *s* from 2017-18 to 2024-25.
- **Target:** next season's raw PAE, weighted by next season's attempts.
- **Predictors compared:**
  1. zero (no skill; the league)
  2. raw PAE in *s*
  3. shrunk PAE in *s*
- **Metric:** weighted mean squared error.
- **Intervals:** the differences use a player-cluster bootstrap with 1,000 resamples.
- **Decision rule:** shot-making is a predictable skill if shrunk PAE beats zero with a 95% interval above zero. Shrinkage helps if shrunk beats raw.

**(c) Efficiency next season:** predicting next season's points per shot (PPS) from this season.
- **Model A:** raw PPS in *s*.
- **Model B:** xPPS in *s* plus shrunk PAE in *s*.
- **Fitting:** both are fit by weighted least squares with leave-one-season-pair-out cross-validation.
- **Comparison:** weighted MSE, with the same bootstrap.
- **What it asks:** whether splitting efficiency into diet and making predicts better than raw efficiency.

## Q4. Team shot diets: selection versus making, on offense and defense

For each team-season in 2017-18 to 2025-26, four quantities:
- **Offense:** xPPS (selection) and PAE per shot (making).
- **Defense:** opponents' xPPS allowed and opponents' PAE allowed.

**Stability:** the year-to-year correlation of each across consecutive seasons for the same team (8 season pairs × 30 teams), with bootstrap intervals by team.

**Pre-stated hypothesis:** defensive making-allowed is less stable year to year than defensive selection-allowed. That is, a defense controls which shots it allows more than whether they go in. This is supported if the correlation for selection-allowed exceeds that for making-allowed, and the bootstrap interval for the difference is above zero.

## Deliverables

- **README:** figures and the pooled results.
- **Two-page PDF summary.**
- **Every claim traced to a saved table by script.**

## Software

- **Python 3.11:** pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib.
- **Versions** are pinned in `requirements.txt`.
- **Seeds:** all random steps are seeded.
