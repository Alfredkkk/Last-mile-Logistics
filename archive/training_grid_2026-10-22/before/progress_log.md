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

### 14. Issue 11 Delivery-Location Ride Visibility

- Updated ride visibility in both experiment notebooks so pickup radius and TTL feasibility are evaluated from the active package delivery target instead of the vehicle's current position.
- Added a visibility-reference helper that falls back to vehicle position only when no package delivery target exists.
- Aligned switching, Voronoi, and four-zone heuristic ride selection with the same delivery-reference point used by `_visible_rides()`.
- Updated README visibility documentation to state that visible rides are screened and sorted from the delivery target, while ride observation features remain encoded relative to the vehicle position.

### 15. 2026-08-07 Re-Review Decisions

- Confirmed that the active CSV/PNG files predate the June fixes because the experiments have not yet been rerun; this is expected and is not treated as a current code issue.
- Confirmed that revenue rate should still be reported for horizon-truncated episodes; no additional unfinished-package penalty will be added, and `finish_rate` remains the separate completion diagnostic.
- Added debug issue 29 as deferred: later investigate whether PPO's discounted cumulative-reward objective helps explain low package completion rates or differs materially from the evaluated revenue-rate objective. No PPO objective or reward change was made in this pass.

## 2026-09-19 Debug Inventory Reconciliation

- Rechecked all 29 debug-log issues against the current four main notebooks, README, and prior decisions.
- Corrected stale status labels for issues 1 and 3-7: the original fixes are present; issue 4's route-following fix does not resolve the separate FOUR_ZONE defects in CODE_REVIEW_2026-09-05.md.
- Reclassified issue 17 as partially fixed: both experiment plotting helpers average pure-delivery results across alpha, but both analysis notebooks still optimize PURE_OR over alpha.
- Added a per-issue assessment to debug_log.md separating resolved bugs, incomplete fixes, final-evaluation requirements, accepted modeling choices, and deferred research questions.
- Reran the September 5 environment/policy probes; their findings still reproduce. These are small behavioral/control-flow checks, not full PyTorch training or OR-Tools solver validation.
- Added numerical evidence for deferred issue 28: for R=1 and travel from (0,1) to (1,0) with a 0.095 step budget, projected positions travel total L1 distance 2, while the movement function reports approximately 2.845647 distance units for revenue and uses 30 rather than 22 steps. Overall experiment impact has not been measured.
- Preserved all accepted/deferred modeling decisions, including issue 28; no experiment notebook, analysis notebook, reward, objective, or historical result was changed by this reconciliation.

## 2026-09-20 TTL Definition Decision

- Confirmed with the user that TTL is the display/availability lifetime of an unaccepted ride request, not a deadline for physically picking up the passenger. Keep the current TTL behavior.
- Closed the definition question in CODE_REVIEW_2026-09-05.md B3 under this interpretation; its late-pickup example remains a semantic illustration rather than a required pickup-deadline fix.
- Left the additional delivery-reference ETA visibility filter unchanged. Its alpha-saturation effect in B5 remains a separate observation.
- FOUR_ZONE timing, order commitment, zone transitions, and route-cost issues remain under discussion. Updated review records only; no simulation or analysis code changed.

## 2026-09-21 Issue 17 and FOUR_ZONE Implementation

- Completed issue 17 first in both analysis notebooks: PURE/PURE_OR average run rates across alpha; ride policies still optimize the per-alpha mean. Verified unequal replicate counts and empty selections.
- Reworked FOUR_ZONE in both experiment notebooks to screen during the final t0 minutes of the whole zone route, using the final delivery location. It accepts a specific request immediately, completes the current zone, starts pickup, and then services the passenger's destination zone.
- Added pending_ride plus shared acceptance/start-pickup helpers. A committed request is removed from the buffer, counted once, and survives its former display TTL. Reset, action masking, and completion detection handle reservations. Normal immediate ride actions keep their existing timing and observation dimensions.
- Kept the additional ETA visibility filter, visibility cap, geometry/radius convention, HEUR/HEUR_VOR behavior, and deferred movement/objective decisions. The optional FOUR_ZONE alpha override now affects its screening; the default uses the environment's radius.
- Predicted completion uses a bounded dry-run of existing delivery motion, including step rounding and boundary projection, with no future demand sampling. Screening includes the completion instant; short zones screen from service start, and unmatched drivers do not wait after clearing the zone.
- Scaled FOUR_ZONE route distances by 1000. Both FOUR_ZONE and PURE_OR now optimize open routes by zeroing only the artificial terminal return arc while preserving departure costs.
- Added tests/test_notebook_regressions.py. All 14 tests passed across both notebook variants, including actual OR-Tools objective checks and full FOUR_ZONE runs with random stationary/non-stationary demand. The policy tests additionally isolate transitions with deterministic route orders.
- Validation command in this workspace: `tmp/fixes_2026-09-21/.venv/Scripts/python.exe -m unittest discover -s tests -v`. The isolated environment contains OR-Tools 9.15.6755 and uses the bundled runtime's NumPy/pandas.
- Final compatibility check: 200 ordinary-action steps per experiment notebook exactly matched pre-change observations, masks, rewards, and info. All four notebooks parse, retain their saved outputs/metadata, and have matching shared route/FZ definitions across stationary and non-stationary variants. Added coverage for nominal last stops already delivered along an earlier route leg; screening uses the actual final delivery location.
- Updated README, debug inventory, review status, and handoff notes. Preserved stored notebook outputs and historical CSV/PNG; no full parameter sweep, PPO training, or GPU verification was performed.

## 2026-09-21 Remaining-Issues Recheck

- Rechecked current notebooks after the issue 17/Four-Zone fixes, without changing their code.
- Reproduced open A5/A6/A8/A9, B2 and B5; statically reconfirmed B1, evaluation/logging/checkpoint limitations, and retained radius mapping. Saved fresh evidence in tmp/fixes_2026-09-21/remaining_results.json.
- A8 uses the actual trainer's control flow with lightweight collaborators, not a PyTorch training run. Performance consequences of the remaining training/model risks are still unmeasured.
- Updated debug_log.md with a current remaining-work inventory and corrected stale #4/#11 rows. Kept A1-A4/A7 closed, TTL's accepted definition, and all prior preserve/defer decisions.

## 2026-09-21 A5/A6/A8/A9 Implementation

- Applied all four follow-up fixes approved by the user, in both notebook variants where applicable.
- A5: classify each step by the executed activity. Pickup and passenger travel include the final dropoff step; advance-accepted FOUR_ZONE requests still count as delivery time while deliveries continue. Movement, reward and step duration are unchanged.
- A6: select only gamma=.50/.83 for the approximately 30/50-package scatter panels, using absolute tolerance 1e-8. Titles identify expected package counts; other densities are no longer rounded into these groups. Existing policy/lambda/RT choices are retained.
- A8: evaluate and log the final model through the same helper as periodic evaluations, with no duplicate evaluation when the two coincide. Disabling periodic evaluation still evaluates the final model; zero updates evaluates/logs the initialized model at update 0. Returned metrics and model now refer to the same training state.
- A9: non-stationary evaluation receives an independent copy of the actual hourly_multiplier, including custom profiles; all 24 hourly arrival intensities match the source environment.
- Added six regression tests in tests/test_remaining_a_fixes.py. All 20 tests passed in the isolated test environment, including the prior real OR-Tools/Four-Zone integration checks. Final-evaluation tests execute real trainer control flow and CSV logging with lightweight training/evaluation substitutes; scatter tests record actual cell plotting inputs without rendering figures.
- Compared both environments with the immediate pre-edit snapshot for 1000 steps each: positions, observations, masks, rewards and termination agree; 12 final dropoff steps per scenario correctly move from delivery time to ride-service time, with unchanged total time. All four notebooks parse, retain outputs/metadata, and match in shared environment-step/trainer/routing definitions. Verification script: tmp/fixes_2026-09-21/verify_a_class.py.
- Marked A1-A9 closed in the review/debug records and updated README/handoff notes. B/C open items and existing accepted/deferred decisions remain. No full PPO training, GPU validation, parameter sweep or historical CSV/PNG regeneration was performed.

## 2026-09-21 B1/B5 Implementation

- B1: the trainer now owns RolloutState across updates, preserving unfinished episode observations/masks and accumulated return/length. True episode endings reset once; batch boundaries bootstrap without being marked terminal. Completed-episode statistics include all fragments and use the environment's actual dt. Evaluation remains isolated in its cloned environment.
- B5: added experiment_support.py for shared state and equivalence accounting. Sweep planning recomputes the alpha saturation threshold from actual R/V/DT and discretized TTL, groups aliases only within otherwise identical settings, and uses the smallest requested alpha as representative. Default grids now have 240 stationary and 180 non-stationary effective configurations per seed. Existing display TTL and per-request ETA screening remain unchanged.
- Added explicit paired TRAIN_SEEDS/train_seeds, defaulting to one seed per effective setting. Raw results and logs record seeds, replicate, alpha aliases/effective value/threshold and physical/demand/training metadata. A companion alpha_summary CSV reports run-level mean, sample SD and count. Single-run SD remains undefined; fixed-seed baselines do not become independent evaluation replicates.
- Updated every best-alpha revenue/ratio helper and the alpha trend plot. Equivalent runs are averaged before selecting a group; distinct scenarios stay separate, with broad overview plots averaging their separately selected results. Pure delivery preserves issue 17's averaging rule. Historical CSVs use an explicit LEGACY_GEOMETRY fallback with warnings only where metadata is absent; new results always use their own metadata. Original nominal alpha columns are preserved.
- Added a training-log header check because the schema gained columns; old logs are not silently corrupted by wider appended rows. This does not implement general run IDs, checkpoint/resume, or B8's independent final evaluation.
- Added ten tests in tests/test_b1_b5.py. All 30 tests passed in 28.0 seconds, including actual environment sampling beyond 4096 minutes to the 5760-minute horizon, real CPU ActorCritic/GAE/PPO updates, real evaluation isolation, boundary statistics/bootstrap, parameter changes, visibility equivalence, seeds, plot inputs and CSV summaries. The PPO smoke check substitutes only the two costly routing baselines; separate existing tests still execute real OR-Tools.
- CPU PyTorch initially hit Windows path-length limits under the project venv. The working isolated PyTorch 2.14.0+cpu installation is in %TEMP%/ll-b1-torch-20260921, prepended to sys.path for these tests; reading it required authorized execution outside the sandbox. OR-Tools remains in the existing project test venv. This is not a completed CUDA/project-training setup.
- Verified all four notebooks parse and import shared helpers from root or NonStationary cwd; stored outputs/metadata are preserved. Environment, visibility, FOUR_ZONE, cloning and evaluation definitions are unchanged. The rollout/trainer/logger definitions match across variants. Script: tmp/fixes_2026-09-21/verify_b1_b5.py.
- Closed B1 and B5, preserving B2/B4/B6/B7/B8 and remaining C items and prior deferrals. No full sweep, GPU training or historical CSV/PNG regeneration was performed; performance impact requires new experiments.

## 2026-09-21 B2/B4 Implementation

- Implemented the approved B2 observation additions in both experiment notebooks: active pickup/dropoff relative coordinates and L1 distances, a pending-commitment flag, and each visible request's normalized remaining TTL. Request identity preserves the TTL association after sorting and with duplicate coordinates. Pickup/dropoff features clear at boarding/dropoff and reset; FOUR_ZONE advance commitments remain observable during delivery.
- Default observation widths are now 79 stationary and 81 non-stationary (previously 67/69), computed from actual package/visible-list limits. Existing network creation automatically uses the new widths. Nearest-10 package compression and limited visibility remain, so this is still partial observation; old-width model weights require retraining or explicit migration.
- Applied B4 option 1 without changing physical behavior: R_paper=sqrt(2)*R_code and alpha_paper=alpha_code/sqrt(2). The nominal area fraction before boundary/TTL filtering is alpha_code^2/2=alpha_paper^2. Existing radius, TTL, ETA screening, B5 saturation/groups and sweep inputs stay unchanged.
- Added OBS_DIM and code-to-paper geometry/radius/alpha metadata to run results and training logs. Group summaries record effective paper alpha and paper region size, while retaining nominal alias lists and distinguishing recorded observation widths. Alpha plots explicitly label code units. Legacy data retain nominal code alpha; no historical files are rewritten. The existing log-header guard prevents appending new-schema rows to old logs.
- Added eight tests in tests/test_b2_b4.py; all 38 tests passed in 26.3 seconds with existing CPU PyTorch and OR-Tools runtimes. Coverage includes accepted-trip lifecycle, pending reservations, zero-distance targets, TTL identity/order/truncation, variable dimensions, retained partial observability, physical conversion and actual sweep/logger integration. Existing actual CPU PPO tests pass with the new observations.
- verify_b2_b4.py compares the immediate pre-change snapshot: 1000 fixed-action steps per environment (12 dropoffs each) retain motion, original observation fields, masks, rewards and termination. All existing non-observation environment methods, training/evaluation definitions and B5 grouping/seeds/numeric summaries are unchanged. All four notebooks parse and preserve saved outputs/metadata; analysis notebooks only gained a units comment.
- Updated README, review/debug status and handoff notes. B2's approved information additions and B4's parameter-definition decision are complete; B6/B7/B8, open C items and prior deferrals remain. No full training/sweep, CUDA verification or historical-output regeneration was performed.

## 2026-09-21 C1 Persistence and Training Progress

- Added training_persistence.py and synchronized both experiment notebooks. Each completed parameter/seed group commits six result rows and atomically refreshes raw/summary CSVs. Independent timestamp/unique-suffix run folders retain the plan, progress, results and per-group logs/checkpoints. Original requested CSV paths remain convenience exports; result.json records allow interrupted exports to be rebuilt.
- Training saves complete-update checkpoints every five updates by default, at initialization, after scheduled evaluations and at the final update. latest.pt/final.pt contain model, optimizer, global/local RNG states, environment data, unfinished rollout state and evaluation status. Notebook-defined ride requests are serialized as fields rather than class objects.
- resume_dir validates the original sweep/configuration, skips committed groups and resumes pending training. The group's own log is verified against the checkpoint and uncommitted records are rolled back before replay. Interrupted final evaluation does not repeat completed optimization; zero-update and fully completed resumes avoid duplicate evaluation/logging.
- Added optional progress bars with text fallback plus run/group progress.json files. Reports group counts, PPO update fraction, stage, elapsed time, estimated remaining update time and checkpoint status. No continuously scheduled monitor is created; status changes at training phase/update boundaries.
- New scans use separate group logs instead of the common TRAIN_LOG_PATH. Existing analysis convergence cells still require log_path to be pointed to a chosen new group log. General row-level run_id tracking, historical-log migration and cross-run aggregation were explained but not implemented; inaccurate eval episode counts and scaled axis labels remain open.
- Added seven tests; all 45 passed in 35.7 seconds. Actual CPU interruption/recovery exactly reproduces uninterrupted model/optimizer/environment/RNG/log states; tests also cover failed final evaluation, settings mismatches, zero updates, accepted requests, atomic write failures, rebuilding exports, skipping completed groups and progress fallback. These new tests use small deterministic evaluation substitutes; existing real evaluation and OR-Tools tests still pass.
- verify_c1.py confirms only trainer/sweep functions changed, preserving notebook saved outputs and all other definitions. Before/after real CPU training produces identical weights and metrics. No full sweep, CUDA recovery, historical-output rewrite or concurrent-writer/OneDrive conflict validation was performed.

## 2026-09-21 Directory Audit, Run Identity and Archival

- Added project_paths.py and updated all four notebook entry points, data/CSV/figure references and convergence cells. Stationary paths always use root Results; non-stationary paths use NonStationary/Results independently of kernel cwd. Default sweep filenames now match the analysis filename. Relative explicit paths are scenario-rooted; absolute custom paths are retained. Shared helper modules must migrate with the notebooks.
- New analyses automatically locate one identified run/group log, print its run/group/seed/path and permit explicit selectors. Missing selections do not fall back to historical common logs. Identity and duplicate updates are validated before plotting; smoothing never averages different training runs. Revenue units now follow the saved scaling setting.
- Added run_id to raw logs/results, summaries, manifests, checkpoints, group identity and progress files. One sweep retains its ID across groups/resume; new runs and standalone training get distinct IDs. Alpha summaries separate independent runs while keeping paired-seed replicates within one run. Checkpoint format is now 2; historical unlabelled checkpoints are not silently migrated.
- Uber discovery skips empty candidate directories, anchors relative overrides to the project root, and permits cached hourly demand without an unused PyArrow import. Confirmed 12 matching monthly files in the sibling Uber_NYC folder; did not recompute the demand fit.
- Added eight path/identity/analysis integration tests, bringing the suite to 53. Preserved saved notebook outputs/metadata and all environment, policy, evaluation and PPO sampling/update definitions. No historical result CSV/PNG regeneration or GPU/full-sweep execution.
- All five historical training logs were already under archive/training_logs; active Results folders contained no common training logs. After completing validation, archived tests and temporary fixes/review material under archive/validation_2026-09-21, retaining evidence and the isolated test environment. Tests discover the project root from their ancestors so their archived location remains usable. The archive includes a relocation manifest and validation output.
- Completion: the full 53 tests passed in 29.8 seconds before relocation. Archived 13 test files, 68 review files and 8,531 fix/runtime files; count/size and source/evidence hash verification passed. Post-archive discovery still finds 53 tests and the eight path/identity checks passed again. Removed generated root bytecode and the now-empty tmp directory.
