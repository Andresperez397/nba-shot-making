# Deviations from ANALYSIS_PLAN.md

Every change made after the plan was frozen (commit 8c2a03d) is logged here, dated, with its reason.

## Analysis

- **2026-10-03, before any Q1 result (computational):** hyperparameter tuning fits each setting on a fixed random sample of 300,000 shots from the seasons before the holdout season, instead of all of them.
  - **Why:** a timing test showed the full gradient-boosting grid would take about 10 hours. The sample cuts that to about 3.5.
  - **What is unchanged:** both learners use the same sample, so tuning stays equally fair. Holdout scoring still uses the whole holdout season, and every final model is refit on all training seasons.
  - **Possible effect:** iteration counts chosen on 300,000 shots may be lower than the full data would support, which can only understate gradient boosting.
