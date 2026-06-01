# Debug Log

## 2026-05-31 Notebook and Experiment Logic Review

### Goal

Record the issues found in the May 31, 2026 review of the stationary and non-stationary last-mile logistics notebooks, then track the first debug pass for issues 1-7.

### User Decisions

- Create a separate root-level `debug_log.md` instead of appending to `progress_log.md`.
- Apply first-round code fixes to both `experiment.ipynb` and `NonStationary/experiment2.ipynb`.
- Keep the current code geometry convention: diamond feasible region `|x| + |y| <= R` with area `2 * R^2`.
- Align documentation and comments with the existing `2 * R^2` convention later.
- Treat `main()` as an unused/obsolete training entry and disable it rather than maintaining it as a supported path.
- Handle the remaining model-alignment and analysis issues in later stages.
- Treat issue 10 as an intentional modified switching heuristic: the pre-grace window is kept because it improved training/evaluation behavior relative to the stricter paper timing.
- Treat issue 11 as unresolved pending advisor discussion.
- Treat issue 12 as out of scope: the project does not need to restrict parameter ranges to the paper's numerical figures.
- For issue 19, use `rate` / `avg_rate` as the main time-weighted aggregate metric and keep `ep_rate` / `avg_ep_rate` as a diagnostic mean of per-episode rates.
- For issues 20-21, replace fragile path strings with notebook-local path resolvers that locate the project root and non-stationary results directory.
- For issue 22, keep `time_frac` and add cyclic hour-of-day observation features so the policy can learn the 24-hour demand cycle.
- Treat issue 23 as deferred: fixed start time may matter for robustness studies, but it is not a current blocking bug.
- Treat issue 24 as deferred: the simulation keeps the paper-style abstract diamond region and does not attempt to spatially map NYC geography into the model.
- For issues 25-26, align README and notebook comments with the active `run_param_sweep()` workflow and the code geometry convention `area = 2 * R^2`.
- For issue 27, keep the legacy baseline parameters but explicitly document that they are not active in the current switching heuristic.

### First-Round Scope

- In scope now: issues 1, 3, 4, 5, 6, and 7.
- Completed in this pass: issue 2 in `analysis.ipynb`.
- Completed in the 2026-06-01 follow-up pass: issues 13, 14, 15, and 16.
- Not in scope now: full historical CSV/PNG regeneration, broad README cleanup beyond issue-specific metric definitions, Uber data path cleanup, or full paper-level policy redesign beyond the concrete bugs below.

## Issue Inventory

### High Priority

1. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `main()` is stale and would crash because `collect_rollout()` now returns four values while `main()` still treats it as a single PPO batch. Status: in first-round scope; disable obsolete entry.
2. `analysis.ipynb`: `paper_axes()` contained stray CSV-reading code with undefined `path` / `r`, so later plotting cells could fail. Status: fixed on 2026-05-31 by removing the stray CSV block from `paper_axes()` while keeping `read_training_log(path)` separate.
3. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `baseline_nearby_rule_voronoi()` filters `visible` rides but converts the filtered candidate index directly into an action, so it can accept the wrong ride. Status: in first-round scope.
4. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `baseline_four_zone()` computes a zone route but still lets `env.step(0)` use global nearest-package routing. Status: in first-round scope.
5. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: `step()` samples new rides before resolving the action from the previous observation/mask, so ride action indices can refer to a different visible set than the policy observed. Status: in first-round scope.
6. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: PPO `ActorCritic` uses dropout, so old and current log probabilities can differ even without parameter changes. Status: in first-round scope.
7. `experiment.ipynb` and `NonStationary/experiment2.ipynb`: training rollouts are not fully reproducible because `CoModalEnv` initializes its local RNG without a seed. Status: in first-round scope.

### Paper Model and Experiment Definition

8. The code uses `|x| + |y| <= R` with area `2 * R^2`, while the paper defines edge-length `R` with area `R^2`. The project will keep the current code convention and document it clearly.
9. `readme.md` and code comments disagree about diamond area (`R^2` vs `2 * R^2`). Status: deferred documentation cleanup.
10. `baseline_nearby_rule()` uses a pre-grace window that may accept a ride before actually completing the package delivery, which is more aggressive than the paper's switching description. Status: accepted intentional modeling change; keep current behavior and document as a modified/pre-grace switching heuristic.
11. Ride visibility is computed from current vehicle position, not the final delivery location used in the paper's screening logic. Status: unresolved; defer until advisor discussion.
12. Project sweeps use arrival-rate and TTL settings that are extensions, not a direct reproduction of the paper's Figure 7/8 calibration. Status: accepted; no action needed because the project is not constrained to the paper's plotted parameter range.

### Results and Analysis

13. `append_training_log()` double-descales eval rows because `evaluate_all()` already reports unscaled metrics. Status: fixed on 2026-06-01 by adding `values_are_unscaled` to `append_training_log()` and passing it for eval rows.
14. Training log schema is unstable because appended rows can have changing field names. Status: fixed on 2026-06-01 by adding fixed `TRAIN_LOG_COLUMNS` and writing all train/eval rows with that schema.
15. `analysis.ipynb` revenue-vs-n plots label `r_l=6` without filtering `RT == 6.0`. Status: fixed on 2026-06-01 by adding `RT_FILTER = 6.0` to the revenue-vs-n cells in both analysis notebooks.
16. `ratio_hvor_drl` uses best alpha for `HEUR_VOR` but average alpha for `DRL`, making the ratio definition inconsistent. Status: fixed on 2026-06-01 by comparing `HEUR_VOR` best alpha against `DRL` best alpha in both analysis notebooks.
17. "Optimal alpha" plots apply alpha optimization to pure delivery baselines, where alpha has no model meaning. Status: fixed on 2026-06-01 by applying best-alpha selection only to ride-aware algorithms (`DRL`, `HEUR`, `HEUR_VOR`) and averaging `PURE` / `PURE_OR` across alpha.
18. Evaluation uses only `EVAL_EPISODES = 5`, so reported comparisons may have high Monte Carlo variance. Status: recommendation recorded on 2026-06-01. Keep `EVAL_EPISODES = 5` for quick/debug sweeps; use `20` for final full sweeps; use `30` for small final confirmation runs when runtime is manageable.
19. Aggregation uses mean of per-episode rates instead of `E[Reward] / E[Time]`, while the paper's objective is the latter. Status: fixed on 2026-06-01 by redefining the main `avg_rate` / CSV `rate` metric as `sum(reward) / sum(terminal_time)` and adding diagnostic `avg_ep_rate` / CSV `ep_rate` as `mean(reward_i / terminal_time_i)`.

### Metric Definitions Adopted for Issue 19

- `avg_rate` / CSV `rate`: primary time-weighted revenue rate, `sum_i reward_i / sum_i terminal_time_i`. This is the metric used by the main analysis plots.
- `avg_ep_rate` / CSV `ep_rate`: diagnostic mean of per-episode rates, `mean_i(reward_i / terminal_time_i)`. This remains useful for spotting episode-level variability but is not the primary comparison metric.

### Non-Stationary Scenario

20. `NonStationary/experiment2.ipynb` hard-codes a local absolute Uber data path. Status: fixed on 2026-06-01 by adding project-root path helpers and resolving Uber data from `UBER_PARQUET_DIR`, `../Uber_NYC`, `Uber_NYC`, or `NonStationary/Uber_NYC` without hard-coded user paths.
21. `NonStationary/analysis2.ipynb` uses relative `"Results/..."` paths that can point to the stationary results when run from the project root. Status: fixed on 2026-06-01 by routing non-stationary CSV/log reads and key figure outputs through `ns_results_path()` / `ns_path()`.
22. Non-stationary demand is represented only through `time_frac`, not explicit cyclic hour-of-day features. Status: fixed on 2026-06-01 by adding `hour_sin` and `hour_cos` to the `NonStationary/experiment2.ipynb` observation core and increasing `obs_dim` by 2.
23. Every non-stationary episode starts at `t = 0`, making results potentially start-time sensitive. Status: deferred by user decision on 2026-06-01; not considered urgent for the current debug pass.
24. The Uber NYC hourly profile is not spatially filtered to the Manhattan/diamond service region. Status: deferred by user decision on 2026-06-01; the project intentionally keeps the paper-style abstract diamond region and uses Uber NYC only for city-wide hour-of-day demand variation.

### Documentation and Code Hygiene

25. `readme.md` defaults are stale (`V`, `HORIZON_MIN`, `RT`, `RP`, and Quick Start do not match current code). Status: fixed on 2026-06-01 by rewriting Quick Start around `run_param_sweep()`, removing `main()` usage, and updating current defaults.
26. Environment comments still describe package count as `Poisson(gamma * R^2)` even though code uses `Poisson(gamma * 2R^2)`. Status: fixed on 2026-06-01 by aligning README and experiment notebook comments to the `2 * R^2` L1 diamond area convention.
27. `baseline_nearby_rule()` accepts `pickup_alpha` and `drop_bias` parameters that are not used. Status: fixed on 2026-06-01 by adding an inline code comment that these are legacy parameters and the current baseline uses `env.r_pick` plus nearest-pickup selection directly.
28. `step_towards()` always moves x-first then y and projects to the diamond if needed, which may distort the intended L1 route/reward near boundaries. Status: deferred modeling discussion.

## First-Round Verification Checklist

- Static checks should confirm `main()` is disabled, `HEUR_VOR` preserves original visible indices, `FOUR_ZONE` forces zone-route package selection, ride arrivals are sampled after action resolution, PPO dropout is removed, and `CoModalEnv` accepts an optional seed.
- Smoke checks should verify a small seeded environment is reproducible and both experiment notebooks can execute the core definitions without running full sweeps.
