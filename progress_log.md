# Project Progress Log

This file records code changes, experiment adjustments, and analysis updates for the last-mile logistics experiments. New entries should be appended chronologically.

## 2026-05-12 Current Retrospective Summary

### 1. Initial PPO Notebook Review

- Reviewed `test_ppo.ipynb` against the paper model and identified several implementation risks in the heuristics and DRL experiment setup.
- Key issues discussed included demand realization consistency across policies, the `is_ride_phase` logic, training/evaluation seed handling, and whether heuristic baselines were comparable to DRL under the same episode realizations.
- Clarified that using the same demand realization across policies is appropriate for fair paired comparison, but repeated evaluation updates should use new seed sets if the goal is to measure learning progress across training.

### 2. Seed and Evaluation Logic

- Discussed why repeated update logs could remain identical when evaluation seeds were fixed across updates.
- Proposed using a changing deterministic seed set per evaluation update, while still keeping the same seeds across policies within the same update.
- Clarified the distinction between:
  - shared seeds across policies for fair comparison;
  - changing seed batches across updates for less repetitive evaluation curves.

### 3. Pure Delivery Baseline

- Reviewed the original pure delivery logic and clarified that it follows a nearest-package delivery policy after each completed package.
- Discussed adding an OR-Tools based pure delivery baseline to solve or approximate a TSP-like package route.
- Reviewed `baseline_pure_ortools` in `test_ppo.ipynb` and checked whether it functions as an effective TSP heuristic.
- Main conceptual distinction recorded:
  - nearest-neighbor pure delivery is a simple online-style heuristic;
  - OR-Tools pure delivery is closer to an offline route optimization baseline.

### 4. Parameter Sweep and Alpha Interpretation

- Reviewed how `ALPHA` affects the pickup visibility radius and therefore the switching policy's access to ride requests.
- Clarified that DRL still uses fixed environment parameters such as `r_pick`, but unlike heuristic baselines it can decide whether to accept visible rides or continue package delivery at each decision point.
- Discussed how high ride-request density and package density can cause heuristic switching baselines to alternate more rigidly between package delivery and rides, while DRL has more flexible action selection.

### 5. Voronoi Visualization Work

- Added or iteratively refined a Voronoi-style package region visualization in `experiment.ipynb`.
- Adjustments included:
  - removing plot borders;
  - generating a clean diamond-domain visualization;
  - forcing `n = 20` packages for the figure;
  - switching from filled continuous regions to a grid of small diamond-shaped cells;
  - adding spacing between the small diamond cells to better match the reference style;
  - hiding package marker points for the final style.
- Final target style: a large diamond composed of many small solid diamond cells, with cell color determined by nearest package region.

### 6. Training Log Instrumentation

- Added training log recording to `experiment.ipynb` so PPO rollout-level metrics are saved instead of relying only on `evaluate_all`.
- Logged metrics included:
  - PPO update index;
  - train/eval type;
  - rollout episode reward and rate;
  - terminal time;
  - number of completed episodes;
  - step-level average reward/rate for incomplete rollout-aware convergence analysis;
  - parameter-combination metadata.
- Corrected reward/rate scale handling so logged training rates are comparable to evaluation rates.
- Clarified that `evaluate_all` produces evaluation data rather than direct training convergence data.

### 7. PPO Update, Episode, and Step Interpretation

- Clarified terminology:
  - one environment step corresponds to `DT = 0.5` minutes;
  - one episode can run up to `HORIZON_MIN / DT = 11520` steps;
  - PPO update means one rollout batch followed by PPO optimization epochs;
  - episodes naturally terminate inside rollouts when all packages are delivered.
- For a typical average terminal time around 2963 minutes, the average episode length is about 5926 steps.
- With `PPO_STEPS = 4096` and `20` updates, estimated total training exposure is about `4096 * 20 / 5926 ~= 14` full episodes per parameter combination.
- Suggested replacing the paper-style phrase "20 consecutive episodes of a maximum of 11,520 epochs" with wording closer to:
  - "The policy is trained for 20 PPO updates, each using rollout batches of fixed step length; each episode has a maximum of 11,520 environment steps but often terminates earlier when all packages are delivered."

### 8. Convergence Analysis in `analysis.ipynb`

- Added convergence visualization based on `Results/training_log.csv`.
- Iteratively improved the plots:
  - added rolling mean smoothing;
  - grouped curves by `LAMBDA` and `GAMMA_PACK`;
  - changed x-axis from update index to episode index where appropriate;
  - switched median aggregation to mean aggregation when requested;
  - separated train rollout statistics from evaluation statistics.
- Identified that older mixed-format `training_log.csv` files caused parser errors and noisy convergence plots.
- Created a cleaned log file approach using `training_log_clean.csv` to recover usable train/eval rows without rerunning long experiments.
- Concluded from the cleaned `step_avg_rate` curve that PPO appeared mostly converged under the longer run setting, with a fast early rise and later plateau.

### 9. Terminal Time Analysis in `analysis.ipynb`

- Added analysis of average terminal time grouped by expected package count `n` and arrival rate `LAMBDA`.
- First implemented line plots by fixed lambda and fixed n.
- Replaced or supplemented line plots with grouped bar charts for more direct comparison across algorithms.
- Adjusted `n_round` so package counts round directly to target values `{20, 30, 40, 50, 60}` rather than displaying approximate values such as `30.2`.
- Added scatter plots requested by the advisor:
  - x-axis: terminal time;
  - y-axis: revenue rate;
  - policies: pure delivery, switching, and DRL;
  - selected settings: `n = 30, 50` and `lambda = 10, 40`.

### 10. Non-Stationary Demand Scenario

- Added non-stationary arrival-rate logic in `NonStationary/experiment2.ipynb`.
- New Uber NYC data fitting step:
  - reads `fhvhv_tripdata_2021-*.parquet` under an `Uber_NYC` data directory;
  - extracts `pickup_datetime`;
  - computes each hour's share of daily trip volume;
  - averages hourly shares across days;
  - caches the result to `hourly_alpha_2021.csv`.
- Defined:
  - `HOURLY_ALPHA`: 24-hour trip share profile;
  - `HOURLY_ALPHA_MEAN = 1 / 24`;
  - `HOURLY_MULTIPLIER = HOURLY_ALPHA / HOURLY_ALPHA_MEAN`.
- Modified `CoModalEnv` so ride arrivals are sampled from a time-varying rate:
  - `lambda_eff(t) = lambda * HOURLY_MULTIPLIER[hour_of_day]`;
  - `k ~ Poisson(lambda_eff(t) * DT)`.
- Clarified interpretation:
  - `lambda` still controls the overall demand scale;
  - `alpha(t)` controls the within-day demand shape.
- Identified a portability issue when running on a server:
  - if `hourly_alpha_2021.csv` is not found at the expected cache path, the notebook falls back to searching for parquet files;
  - if neither cache nor parquet files are found, `FileNotFoundError` is raised.
- Recommended checking `Path.cwd()`, `UBER_PARQUET_DIR.exists()`, `HOURLY_ALPHA_CACHE.exists()`, and using absolute paths on the server.

### 11. Non-Stationary Analysis Notebook Fixes

- Inspected `NonStationary/analysis2.ipynb` after a `NameError` in `paper_axes`.
- Found that CSV-reading logic had accidentally been placed inside `paper_axes(ax)`, where `path` was undefined.
- Identified parser errors in `NonStationary/Results/training_log.csv` caused by inconsistent row lengths:
  - header had 17 columns;
  - eval rows had an extra trailing `eval` marker.
- Added preprocessing logic inside the convergence-analysis cell to:
  - read the raw CSV using Python's `csv` module;
  - normalize each row to a consistent column count;
  - add a new `type` column;
  - label normal rows as `train`;
  - label legacy extra-marker rows as `eval`;
  - write `Results/training_log_clean.csv`.
- Clarified that the existing train/eval plotting logic was conceptually correct, but pandas failed before reaching it because the raw CSV could not be parsed.

### 12. Ratio Plot Interpretation

- Investigated why a ratio plot showed a point to the left of `lambda = 5`.
- Found the plot was using:
  - `lambda_vals = sorted(df["LAMBDA"].unique())`
- The underlying result CSV included `lambda = 2`, so the leftmost point was `lambda = 2`, not a spurious value left of 5.
- Suggested filtering:
  - `df = df[df["LAMBDA"] >= 5]`
  - or `lambda_vals = [lam for lam in sorted(df["LAMBDA"].unique()) if lam >= 5]`

## Open Items

- Decide whether non-stationary experiments should always rely on the cached `hourly_alpha_2021.csv` or whether server runs should also support recomputing from parquet.
- Confirm whether evaluation rows should be plotted as a separate convergence curve or used only as checkpoint markers.
- Revisit the ratio plots after filtering out `lambda = 2` if the final paper figure should start at `lambda = 5`.
- Consider making `Results/training_log_clean.csv` generation a reusable helper function rather than cell-local preprocessing.

