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

## 2026-06-01 Debug Pass: Notebook Logic and Analysis Cleanup

### 1. Debug Tracking

- Created `debug_log.md` as the dedicated debug record for the project.
- Recorded 28 reviewed issues with status labels, including code-level bugs, modeling deviations, analysis inconsistencies, and documentation cleanup items.
- Agreed to keep `debug_log.md` and `progress_log.md` synchronized for subsequent debug passes.

### 2. First-Round Experiment Notebook Fixes

- Updated both `experiment.ipynb` and `NonStationary/experiment2.ipynb`.
- Disabled the stale `main()` training entry because the active training path is now `train_policy_brief()` / `run_param_sweep()`.
- Fixed `HEUR_VOR` ride selection so the filtered Voronoi candidate keeps its original visible-ride action index.
- Fixed `FOUR_ZONE` so it actually follows the planned zone route rather than falling back to global nearest-package delivery.
- Moved new ride arrivals to the end of `step()` so action indices correspond to the ride set shown in the previous observation/mask.
- Removed PPO dropout from `ActorCritic` to keep PPO old/current log-probability comparisons deterministic under unchanged parameters.
- Added environment seed support so seeded rollout behavior is reproducible.

### 2.1 Issue 1-7 Status Mapping

- Issue 1 (`main()` stale training entry): fixed by disabling the obsolete `main()` cell and directing users to `train_policy_brief()` / `run_param_sweep()`.
- Issue 2 (`analysis.ipynb` `paper_axes()` pollution): fixed by removing the stray CSV-reading block from `paper_axes(ax)`.
- Issue 3 (`HEUR_VOR` filtered ride index): fixed by preserving the original `visible` ride index when converting a Voronoi-filtered candidate into an action.
- Issue 4 (`FOUR_ZONE` route ignored): fixed by temporarily overriding `env._nearest_package()` inside `baseline_four_zone()` so package delivery follows the planned zone route.
- Issue 5 (action/mask ride-set mismatch): fixed by resolving the current action before appending newly sampled ride requests.
- Issue 6 (PPO dropout instability): fixed by removing dropout layers from `ActorCritic`.
- Issue 7 (training rollout RNG reproducibility): fixed by adding optional `seed` support to `CoModalEnv`, deriving default environment seeds from the global seeded RNG, and seeding sweep environments.

### 3. Analysis Notebook Fix

- Fixed `analysis.ipynb` by removing stray CSV-reading code accidentally embedded inside `paper_axes(ax)`.
- Verified `paper_axes(ax)` can now be called without the previous `NameError`.

### 4. Modeling Decisions for Issues 8-12

- Confirmed the project will keep the code geometry convention: feasible region `|x| + |y| <= R` and area `2 * R^2`.
- Deferred README/comment updates that still refer to the paper's `R^2` area convention.
- Confirmed the pre-grace switching heuristic is intentional and should be documented as a modified switching heuristic, not treated as a bug.
- Marked the current-location ride visibility rule as unresolved pending advisor discussion.
- Decided not to constrain the experiment parameter ranges to the paper's numerical figure ranges.

### 5. Logging and Analysis Fixes for Issues 13-16

- Updated `append_training_log()` in both experiment notebooks with a `values_are_unscaled` flag.
- Eval rows now pass `values_are_unscaled=REPORT_UNSCALED`, preventing double descaling when `evaluate_all()` already returns unscaled metrics.
- Added fixed `TRAIN_LOG_COLUMNS` so future train/eval rows use the same CSV schema.
- Updated revenue-rate-vs-n cells in both analysis notebooks to filter by `RT_FILTER = 6.0` when figures are labeled with `r_l=6`.
- Updated the `HEUR_VOR / DRL` ratio cells in both analysis notebooks so both algorithms use best-alpha rates under each `(lambda, gamma)` setting.

### 6. Follow-Up Decisions for Issues 17-19

- Fixed the optimal-alpha plotting helper in both experiment notebooks so only ride-aware algorithms (`DRL`, `HEUR`, `HEUR_VOR`) use best-alpha selection.
- Pure delivery baselines (`PURE`, `PURE_OR`) are now averaged across alpha in the optimal-alpha plots because they do not use ride visibility.
- Recorded the evaluation-size recommendation: keep `EVAL_EPISODES = 5` for quick/debug sweeps, use `20` for final full sweeps, and use `30` for smaller final confirmation runs when runtime allows.
- Fixed issue 19 by redefining the main aggregate `avg_rate` / CSV `rate` as `sum(reward) / sum(terminal_time)`.
- Added diagnostic `avg_ep_rate` / CSV `ep_rate`, defined as `mean(reward_i / terminal_time_i)`, and added the corresponding training-log column.
- Updated the main analysis notebook plots to continue using CSV `rate` as the primary metric, with inline comments documenting that `ep_rate` is diagnostic only.

### 7. Non-Stationary Path Fixes for Issues 20-21

- Added project-root and non-stationary path helpers to `NonStationary/experiment2.ipynb`.
- Removed the hard-coded local Uber NYC absolute path. The notebook now resolves Uber parquet data from `UBER_PARQUET_DIR`, a sibling `Uber_NYC` directory next to the project root, `Last-mile Logistics/Uber_NYC`, or `NonStationary/Uber_NYC`.
- Moved the hourly alpha cache to `NonStationary/Results/hourly_alpha_2021.csv`, so fitting can cache inside the project instead of writing into the external data directory.
- Routed non-stationary training logs, sweep CSVs, and plot outputs in `experiment2.ipynb` through `ns_results_path()` or `ns_path()`.
- Added the same project-root helpers to `NonStationary/analysis2.ipynb`, so `CSV_PATH`, cleaned training logs, and non-stationary analysis figures resolve to `NonStationary/Results` or `NonStationary` even when the notebook is run from the project root.

### 8. Non-Stationary Observation Fix for Issue 22

- Added cyclic hour-of-day features to the non-stationary environment observation in `NonStationary/experiment2.ipynb`.
- The observation core now keeps `time_frac` and additionally includes `hour_sin = sin(2*pi*(t mod 1440)/1440)` and `hour_cos = cos(2*pi*(t mod 1440)/1440)`.
- Increased `CoModalEnv.obs_dim` by 2 so the PPO policy input dimension matches the expanded observation vector.

### 9. Deferred Non-Stationary Start-Time Issue 23

- Marked issue 23 as deferred by user decision.
- Current non-stationary episodes still start at `t = 0`; randomizing or sweeping episode start time is treated as a future robustness/design extension, not a current debug blocker.

### 10. Deferred Uber Spatial Filtering Issue 24

- Marked issue 24 as deferred by user decision.
- The project keeps the paper-style abstract diamond service region rather than mapping NYC geography into the simulation.
- The Uber NYC data is currently used only to calibrate a city-wide hour-of-day demand profile for non-stationary arrivals.

### 11. README and Comment Cleanup for Issues 25-27

- Rewrote README Quick Start to use the active `run_param_sweep()` workflow instead of the disabled `main()` entry.
- Updated README defaults to match the current experiment notebooks: `V = 0.19`, `HORIZON_MIN = 5760.0`, `GAMMA_PACK = 0.075`, `RT = 5.5`, `RP = 2.0`, and `PPO_STEPS = 8192`.
- Aligned README geometry text with the code convention: feasible region `|x| + |y| <= R` and package sampling area `2 * R^2`.
- Marked issues 8 and 9 as resolved through this documentation/comment cleanup: the project keeps the code's L1 radius convention even though the paper uses a different `R` area convention.
- Updated environment comments in both experiment notebooks from `Poisson(gamma * R^2)` to `Poisson(gamma * 2R^2)`.
- Added an inline comment to `baseline_nearby_rule()` in both experiment notebooks noting that `pickup_alpha` and `drop_bias` are legacy parameters and are not used by the current switching heuristic.

### 12. Deferred Movement Geometry Issue 28

- Marked issue 28 as deferred by user decision.
- Preferred future fix, if needed: replace x-first/y-first `step_towards()` movement with L1 geodesic interpolation along the segment from current position to target.
- Rationale: the diamond feasible region is convex, so interpolating between two feasible endpoints stays inside `|x| + |y| <= R`; the interpolation also preserves L1 step length as `min(max_dist, manhattan(from_pt, to_pt))`.

### 13. Training Log Path Cleanup

- Archived legacy training-log CSVs under `archive/training_logs/` instead of keeping them in active result directories.
- Updated `analysis.ipynb` to read `Results/training_log.csv` directly for convergence plots.
- Updated `NonStationary/analysis2.ipynb` to read `NonStationary/Results/training_log.csv` directly through `ns_results_path()`.
- Removed the current analysis dependency on `training_log_clean.csv`; clean files are now treated only as archived legacy recovery artifacts.
- Converted the convergence-cell comments and error messages to English.
