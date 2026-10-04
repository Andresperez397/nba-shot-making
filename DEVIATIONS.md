# Deviations from ANALYSIS_PLAN.md

Every change made after the plan was frozen (commit 8c2a03d) is logged here, dated, with its reason.

## Analysis

- **2026-10-04, season window (data-quality break found during the first Q1 run):** the analysis now covers 2017-18 to 2025-26, not 2015-16 to 2025-26.
  - **How it was found:** on the first test season (2017-18), full-location gradient boosting scored worse than distance alone (log loss 0.670 against 0.656). A check traced this to the y coordinate.
  - **What changed in the source:** rim attempts are charted about 0.4 feet further out from 2017-18, and the 1–2 ft make rate jumps from 58% to 68% in one season (`DATA_AUDIT.md`, "Rim geometry").
  - **Results already seen:** that run was stopped. Its only outputs were the 2017-18 log losses of the constant, distance-only, full-location logistic and full-location boosting models. Nothing from it is reported as a result.
  - **New test seasons:** Q1 tests on 2019-20 to 2025-26 (seven seasons, each trained on two or more seasons from 2017-18). Q2–Q4 use the same seven seasons, so Q3b and Q4 have six consecutive season pairs instead of eight.
  - **What is unchanged:** every model, grid, metric and decision rule is as planned.

- **2026-10-04, added robustness check (Q3):** the shot-making stability tests repeated with location-only expected makes (Q1 M2 boosting).
  - **Why:** shot-type labels are assigned by scorers after the play and gave the largest gain in Q1. Using location-only expected makes tests whether the conclusions depend on those labels.
  - **Result:** they do not. Split-half r is 0.47 (main 0.44), year-to-year r of shrunk shot-making is 0.56 (main 0.55), and the two shot-making estimates correlate 0.96 for player-seasons with at least 500 attempts.

- **2026-10-03, before any Q1 result (computational):** hyperparameter tuning fits each setting on a fixed random sample of 300,000 shots from the seasons before the holdout season, instead of all of them.
  - **Why:** a timing test showed the full gradient-boosting grid would take about 10 hours. The sample cuts that to about 3.5.
  - **What is unchanged:** both learners use the same sample, so tuning stays equally fair. Holdout scoring still uses the whole holdout season, and every final model is refit on all training seasons.
  - **Possible effect:** iteration counts chosen on 300,000 shots may be lower than the full data would support, which can only understate gradient boosting.
