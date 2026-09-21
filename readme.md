# DRL for Joint Ride-Hailing & Package Delivery

This project simulates a **single vehicle** operating in a diamond-shaped city (L1 metric) that must decide, at each step, whether to **deliver packages** or **accept a passenger ride**. Passenger requests arrive as a **spatially homogeneous Poisson process**; packages are drawn from a **spatial Poisson field**. We compare a **Deep RL (PPO)** policy against a **switching heuristic** (paper’s $N=n$ case), a **pure delivery** and a **heuristic zoning policy** baseline.

The project and part of the analysis is in the `experiment.ipynb` file.

Formal analysis in `analysis.ipynb` file.

This project is based on paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4565248.

For non-stationary scenario, the data we prepare to use is https://www.kaggle.com/datasets/shuhengmo/uber-nyc-forhire-vehicles-trip-data-2021

This README reflects the current notebook entry points and metric definitions used in the debug pass.

## Quick Start

`main()` is intentionally disabled. The active experiment entry point is the parameter sweep cell that trains/evaluates each setting and writes the CSV used by the analysis notebooks.

Stationary sweep example from `experiment.ipynb`:

```python
LAMBDA_list            = [2, 5, 10, 20, 30, 40]
R_PICK_ALPHA_list      = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
GAMMA_PACK_list        = [0.33, 0.50, 0.67, 0.83, 1.00]
RIDE_TTL_MINUTES_list  = [5]
MAX_VISIBLE_RIDES_list = [5]
SWITCH_GRACE_STEPS_list= [6]
RT_list                = [5.5, 6.0]

CSV_PATH = results_path("param_sweep_results_2.csv")

df = run_param_sweep(
    LAMBDA_list,
    R_PICK_ALPHA_list,
    GAMMA_PACK_list,
    RIDE_TTL_MINUTES_list,
    MAX_VISIBLE_RIDES_list,
    SWITCH_GRACE_STEPS_list,
    RT_list,
    train_updates_per_combo=50,      # Use 20-50 for short runs; 200 for full training.
    csv_path=CSV_PATH,
    seed_offset=0
)

plot_from_csv(
    CSV_PATH,
    vary_keys=["LAMBDA", "GAMMA_PACK", "R_PICK_ALPHA", "RT"],
    metrics=("rate","reward","accepted")
)
```

For the non-stationary notebook, use the analogous cell in `NonStationary/experiment2.ipynb`; it writes to `NonStationary/Results/param_sweep_results_2.csv`.

All four notebooks import `project_paths.py`. Stationary CSVs/figures belong to the root `Results/`; non-stationary CSVs/figures belong to `NonStationary/Results/`, whether Jupyter starts in the root or `NonStationary/`. `results_path(name)` constructs a scenario path. Relative arguments such as `Results/name.csv` are anchored to that notebook's scenario; explicit absolute paths remain supported. The default sweep output now matches the analyses: `param_sweep_results_2.csv`. If the kernel starts outside this project, set `LAST_MILE_PROJECT_ROOT` to the project folder before initialization.

Uber data discovery prefers directories containing `fhvhv_tripdata_2021-*.parquet`, skipping empty folders. A relative `UBER_PARQUET_DIR` override is relative to the project root. The hourly cache is `NonStationary/Results/hourly_alpha_2021.csv`; reading a prepared cache does not require importing PyArrow or having the raw files present. The current machine has all 12 monthly parquet files in the sibling `Uber_NYC` folder; the full demand fit is not part of regression testing.

## Environment & Dynamics

### City geometry
- **Diamond region** (L1 ball): feasible $(x,y)$ satisfy $|x|+|y|\le R$.
- **Distance**: Manhattan $d_1(a,b)=|a_x-b_x|+|a_y-b_y|$.

### Time & motion
- **Step size**: `DT` minutes per step.
- **Vehicle speed**: `V` distance units per minute.
- **Horizon**: `HORIZON_MIN` minutes max per episode; steps per episode `STEPS_PER_EP = HORIZON_MIN / DT`.

### Stochastic primitives
- **Packages**: on reset, sample $N\sim \text{Poisson}(\gamma \cdot 2R^2)$; locations IID uniform in the diamond.  
  (`GAMMA_PACK = γ` is **intensity per unit area**; the L1-ball diamond area used by the code is $2R^2$.)
- **Rides**: each step draws $\text{Poisson}(\lambda \cdot DT)$ new requests with pickup & dropoff IID uniform in the diamond (`LAMBDA = λ` per minute).

### Request visibility & TTL
- **Visible rides** must satisfy both:
  1) **Pickup within radius** $r_{\text{pick}}=\alpha \cdot \frac{R}{\sqrt 2}$ (`R_PICK_ALPHA = α`) of the current delivery target, not the vehicle's current position.
  2) **Additional ETA filter**: ETA from that delivery target to pickup $\le$ remaining display TTL. This does not guarantee an actual pickup deadline.
     Code: `eta_min = manhattan(visibility_ref, pickup) / V` and require `eta_min <= ttl * DT`.
- The default visibility reference is the package delivery target currently used by `_nearest_package()`; if no package target exists, the environment falls back to the vehicle position. FOUR_ZONE explicitly screens from the final delivery point of its current zone route.
- At most `MAX_VISIBLE_RIDES` **closest** pickups to that visibility reference are exposed each step.
- Unaccepted rides expire when their TTL reaches 0 (TTL is stored in **steps**; decremented by 1 per step).
- TTL is the lifetime of an **unaccepted** request. An accepted FOUR_ZONE request is removed from the buffer and remains committed while the current zone's deliveries finish.

### Mapping geometry and alpha to the paper
The existing physical model is preserved. Write the code's L1 radius as `R_code` and the edge length of the same diamond in the paper as `R_paper`. Then `R_paper = sqrt(2) * R_code` and `alpha_paper = alpha_code / sqrt(2)`, so both conventions give the same pickup radius:

`r_pick = alpha_code * R_code / sqrt(2) = alpha_paper * R_paper / sqrt(2)`.

Before boundary clipping or TTL filtering, the nominal pickup-area fraction is `alpha_code^2 / 2 = alpha_paper^2`. For example, code alpha `0.2` corresponds to paper alpha approximately `0.1414`. This conversion does not remove the simulation's boundary effects, additional ETA filter or visible-request cap, and does not by itself justify applying the paper's arrival-rate formula. Sweep inputs and alpha plot axes continue to use **code units**; output metadata also records the paper convention.

### Termination
An episode ends when **either**:
- **All packages delivered** and there is no accepted request awaiting pickup, pickup trip, or passenger onboard (`packages_done`), **or**
- **Horizon reached** (`horizon_reached`).

## Observation & Action

### Observation vector (normalized, flattened)
- Core: $[x/R,\, y/R,\, \text{time\_frac}=t/\text{HORIZON\_MIN},\, \mathbf{1}_{\text{to\_pickup}},\, \mathbf{1}_{\text{with\_pass}},\, \text{pkg\_count\_norm},\, \text{visible\_norm}]$.
- Non-stationary runs additionally include cyclic hour-of-day features: $\sin(2\pi h)$ and $\cos(2\pi h)$, where $h=(t\bmod 1440)/1440$.
- Nearest `K_NEAREST_PACK` packages: for each $(\Delta x/R,\,\Delta y/R,\, d_1/R)$.
- Up to `MAX_VISIBLE_RIDES` rides: for each $(\Delta x_{\text{pick}}/R,\,\Delta y_{\text{pick}}/R,\, d_{1,\text{pick}}/R,\,\Delta x_{\text{drop}}/R,\,\Delta y_{\text{drop}}/R,\, \text{trip\_len}/R,\, \text{remaining TTL steps}/\text{initial TTL steps})$. The TTL stays paired with the actual request after visibility sorting. Ride features remain encoded relative to the vehicle position, even though the visible set is screened from the delivery target.
- Accepted ride, appended after the visible list: `[pending_ride flag, pickup dx/R, pickup dy/R, pickup L1 distance/R, dropoff dx/R, dropoff dy/R, dropoff L1 distance/R]`. This includes a FOUR_ZONE commitment while deliveries continue. Pickup features clear after boarding; dropoff features clear after dropoff; reset clears the whole block. Existing pickup/passenger flags distinguish an active target at zero distance from no target.
- Zero-padded when fewer items exist.

With `K_NEAREST_PACK=10` and `MAX_VISIBLE_RIDES=5`, the input width is **79** for stationary and **81** for non-stationary runs (previously 67/69). Width is derived from these settings; network creation uses `env.obs_dim`. Models using the old input layout require retraining or an explicit weight-migration design. Only the nearest 10 packages and the capped visible list remain encoded, so this is still a **partial, compressed observation**, not the paper's complete dynamic-programming state.

### Action space (discrete)
- `0` = **deliver** (go toward nearest package).
- `1..MAX_VISIBLE_RIDES` = accept the $i$-th visible ride (closest pickups to the delivery-reference point first).  
  Invalid indices are **masked** (logits set to $-10^9$).


## Reward Function (Model)

Rewards are per step and **internally scaled** for stabilization:

- **Ride revenue** (only while carrying a passenger):
  $$r_{\text{ride}}=\texttt{rt}\times \text{distance\_traveled\_this\_step}$$
- **Package revenue** (upon arrival at package location):
  $$r_{\text{pkg}}=\texttt{rp}\times \{\text{number of packages delivered in this step}\}$$

**Scaling:** In `CoModalEnv.__init__`,  
$$\texttt{rt}=\texttt{RT}\times \texttt{REWARD\_SCALE},\qquad
\texttt{rp}=\texttt{RP}\times \texttt{REWARD\_SCALE}.$$
If `REPORT_UNSCALED=True`, all printed **reward** and rate metrics are **descaled** by `INV_REWARD_SCALE` to original units.

> Units: distance is L1; `V` is distance/min; `DT` is min/step; reward is in revenue units (ride: per distance, package: per item). Aggregated **Rate** is per minute of actual episode time.


## Policies

### DRL (PPO; greedy at eval)
- **Network**: shared MLP trunk (256 units × 2) with **LayerNorm → ReLU**; heads:
  - `pi`: logits over actions
  - `v`: scalar value
- **Action masking** applied to logits before sampling/argmax.
- **PPO training**:
  - Collect exactly `PPO_STEPS` **steps** per update (may span several episodes).
  - Compute **GAE(λ)** advantages & returns.
  - Optimize for `PPO_EPOCHS` passes with minibatches of size `PPO_MINI_BATCH`.
  - Hyperparams: `CLIP_EPS`, `VF_COEF`, `ENT_COEF`, `LR`, `DISCOUNT`, `GAE_LAMBDA`, total `UPDATES`.
- **Sampling**: stochastic during training; **greedy** (`argmax`) for evaluation.

The trainer owns a `RolloutState` across PPO updates. A full rollout batch keeps an unfinished episode's environment, observation/mask, accumulated reward and length; only a real episode ending resets it. Batch boundaries bootstrap from the current policy's next-state value. Completed-episode training statistics can span several policy updates; fixed-policy performance is measured by the separate evaluation environment. Calling `collect_rollout()` without a state still starts a standalone batch.

### Switching Heuristic (Zoning Policy when $N=n$)
- **Primarily deliver** the nearest package.
- **Switching window**: after a package delivery (and a one-step **pre-grace** if just about to deliver), open `SWITCH_GRACE_STEPS` decision steps:
  - If any **visible** ride exists (within $r_{\text{pick}}$ and ETA $\le$ TTL), **accept the nearest pickup**.
  - After completing that ride, immediately **return to deliver** the nearest package.
- No chaining: it does not keep accepting rides unless a new window opens around a delivery event.

### Pure Delivery
- Always chooses action `0`.

### OR-Tools Pure Delivery (PURE_OR)
- Plans an approximate open L1 route from the initial vehicle position through all packages, with no return trip.
- Uses integer costs at 1000 units per distance unit and a two-second local-search limit; this is not a guarantee of a globally optimal route.

### Four-Zone Policy (FOUR_ZONE)
- Clears one zone at a time along an approximate open OR-Tools route, using the same distance precision as PURE_OR.
- Screens rides during the final `screen_window_min` minutes (`t0`, default 10) before completing the **entire zone**. The pickup reference is its final delivery point; destinations must be other zones with undelivered packages.
- Accepts and stores at most one eligible request during that window, finishes the current zone, then picks up and serves the passenger. After dropoff it delivers in the passenger's destination zone before the next passenger trip. At most three rides connect four occupied zones.
- If unmatched when the zone is cleared, it immediately heads to the nearest unserved zone. It does not wait after completion or screen during a passenger trip.
- Completion prediction follows the existing discrete movement/projection rules without sampling future requests. Screening starts at the first decision time with remaining delivery time <= t0, includes the completion instant, and starts immediately for zones shorter than t0.
- Retains the display TTL, additional reference-point ETA filter, visibility cap, and project radius convention. `r_pick_alpha=None` uses `env.r_pick`; an explicit override only changes this baseline's screening radius. These retained simulation assumptions are extensions of the paper model.


## Key Parameters (defaults)

The core of this experiement is in parameter sweep part. These parameters here are just for default use.

**Geometry & time**
- `R = 5.5` (L1 radius), `V = 0.19` (distance/min), `DT = 0.5` (min/step), `HORIZON_MIN = 5760.0`.
- Feasible region: `|x| + |y| <= R`; area used for package sampling is `2 * R^2`.

**Demand & revenue**
- `LAMBDA = 0.60` (rides per minute), `GAMMA_PACK = 0.075` (packages per unit area),
- `R_PICK_ALPHA = 0.5` (pickup visibility $r_{\text{pick}} = \alpha R/\sqrt 2$),
- `RIDE_TTL_MINUTES = 5`,
- `RT = 5.5` (ride revenue / distance), `RP = 2.0` (package revenue / item),
- `REWARD_SCALE = 1/8`, `REPORT_UNSCALED = True`.

**DRL (PPO)**
- `MAX_VISIBLE_RIDES = 5`, `K_NEAREST_PACK = 10`,
- `DISCOUNT = 0.99`, `GAE_LAMBDA = 0.95`,
- `PPO_STEPS = 8192`, `PPO_MINI_BATCH = 256`, `PPO_EPOCHS = 4`,
- `CLIP_EPS = 0.2`, `VF_COEF = 0.5`, `ENT_COEF = 0.02`, `LR = 3e-4`,
- `UPDATES = 200`.

**Heuristic**
- `SWITCH_GRACE_STEPS = 6` (window length after delivery to consider rides),
- `HEUR_SOFT_PICK_CAP = None` (optional: accept if pickup extremely close).


## Logging & Metrics

At every `eval_every` updates (default **5**) and at the final update, we evaluate and print aggregated policy averages over `EVAL_EPISODES`. A final update that is also a periodic evaluation point is evaluated only once:
- **R** (reward), **T** (terminal time), **Rate** (reward/min), **Acc** (accepted rides; DRL only),
- PPO diagnostics: `pi` (policy loss), `v` (value loss), `ent` (entropy).

`train_policy_brief()` returns the final policy together with metrics from that same model state. With `eval_every=0`, it still evaluates the final model. With zero updates, it evaluates the initialized model at update 0. Every evaluation uses the same CSV logging path when `log_path` is provided. Non-stationary evaluation environments receive an independent copy of the training environment's actual 24-hour demand multiplier, including custom profiles.

Training logs include `run_id`, training seed, replicate, alpha-equivalence and environment metadata. One sweep has one automatically generated `run_id`; `combo_id` identifies a parameter/seed group within it. Logs, raw results, summaries, manifests, checkpoints and progress records carry the identity. Resume preserves it; a new sweep or standalone training call gets a new identity even with the same seed. Logs live in each run's group folder, and the obsolete common `TRAIN_LOG_PATH` is removed. Standalone training can still choose a log path; its rows are identified and schema mismatches are rejected.

The main aggregate rate metric is `rate` / `avg_rate`:
$$\text{avg\_rate}=\frac{\sum_i \text{reward}_i}{\sum_i \text{terminal\_time}_i}.$$
The diagnostic metric `ep_rate` / `avg_ep_rate` is the mean of per-episode rates:
$$\text{avg\_ep\_rate}=\frac{1}{N}\sum_i\frac{\text{reward}_i}{\text{terminal\_time}_i}.$$
Use `rate` for primary comparisons and `ep_rate` only to diagnose episode-level variability.

Per-episode breakdown (for each algorithm) includes:
- `reward`, `terminal_time`, per-episode `rate`, `accepted`, `ended_reason`.

Environment time counters classify each whole step by its executed activity: `time_rides_min` includes travel to pickup and passenger travel, including the final dropoff step. A FOUR_ZONE request accepted in advance while deliveries continue is still counted in `time_delivery_min`. The two counters sum to elapsed episode time.

Sweep CSVs store `rate` as the primary aggregate `avg_rate` and `ep_rate` as the diagnostic mean per-episode rate.

With `REPORT_UNSCALED=True`, printed **reward**/**rate** are in **original units** (descaled).


## Parameter Sweeps & Plots

Run grid experiments and save aggregated results with `run_param_sweep()`. The stationary notebook writes to `Results/`; the non-stationary notebook writes to `NonStationary/Results/` through its path helpers.

### Saving, resuming and progress

Saving is enabled by default in `run_param_sweep()`. Every completed parameter/seed group immediately commits all six policies' results and refreshes both the requested CSV and its alpha summary. Updates use temporary files plus atomic replacement, so a failed individual write preserves the previous file. Each group's `result.json` is the authoritative completion record; resuming rebuilds CSV exports from those records and skips completed groups.

Each new sweep creates `<csv_stem>_runs/<run_id>/` next to its requested CSV; the ID contains a UTC timestamp and unique suffix. The directory is printed before training starts and returned in `df.attrs['run_directory']`, with its identity in `df.attrs['run_id']`, after completion. It contains:

- `manifest.json`: original sweep plan and training settings.
- `progress.json`: completed/total groups, current group, current training stage and most recent status timestamp.
- `results.csv` and `alpha_summary.csv`: this run's saved results, retained even if another run reuses the convenience export path.
- `combo_0001/`, etc.: each group's `run.json`, `latest.pt`, `final.pt`, `training_log.csv`, `progress.json` and, after completion, `result.json`.

`checkpoint_every=5` saves after every five complete PPO updates; initialization, periodic evaluation completion and the final update are also saved. Only the latest recovery checkpoint and the final checkpoint are retained per group. A checkpoint includes model/optimizer state, Python/NumPy/PyTorch random states, the environment and its local generator, the unfinished episode, and completed evaluation metrics. Interruption inside an update rolls back to the last saved complete update. Logs beyond that saved point are removed only from the group's own log before replaying the work. Interrupted final evaluation retries evaluation without retraining the completed model.

To resume, keep the original grid, seeds, total updates and training settings, and set `resume_dir` to the printed run folder in the same `run_param_sweep()` call. The saved manifest and checkpoint configuration are checked; changing parameters or the total update target requires a new run. Use the same code and dependency versions when resuming. Do not run two writers against the same resume folder. Existing pre-checkpoint runs cannot be resumed from historical CSVs alone. Checkpoints use PyTorch serialization and should be loaded only from your own trusted runs.

`show_progress=True` displays group counts and a PPO update progress bar with elapsed time/estimated time remaining; the stage explicitly distinguishes sampling/optimization, evaluation and completion. If `tqdm` is unavailable, a text bar is printed instead. The two `progress.json` files can be inspected without waiting for the notebook to finish. Status is updated at phase/update boundaries, not a continuous activity heartbeat; training-update ETA does not predict the exact cost of all later evaluations. `show_progress=False` hides bars while preserving on-disk status.

Both convergence cells now resolve `log_path` automatically under their own scenario's `<csv_stem>_runs/`. By default they select the newest identified run containing log rows, then its highest numbered group with rows; the selected run, group, seed and absolute path are printed. Set `ANALYSIS_RUN_ID`, `ANALYSIS_COMBO_ID` or `ANALYSIS_RUN_DIR` in that cell to select a particular run/group. Explicit selections do not silently fall back to another run. A curve contains one run/group/seed, and duplicate update records are rejected instead of averaged. Units follow the recorded scaling convention. No historical common log is selected automatically; if no new training has run yet, the error explains what is missing.

The five pre-existing training logs are retained under `archive/training_logs/stationary/` and `archive/training_logs/nonstationary/`. Their missing identities are not fabricated. Checkpoint format 2 includes run identity; earlier unlabelled checkpoint folders are not automatically migrated. Historical result CSVs/figures remain available, separately from new training logs.

For standalone `train_policy_brief()`, pass `checkpoint_dir=...` to enable saving and `resume=True` to continue it. Its log must be the `training_log.csv` inside that same directory. Without a checkpoint directory, the existing standalone behavior is retained.

### Parameter grouping

The sweep automatically groups equivalent alpha values using the actual `R`, `V`, `DT` and discretized TTL: `ttl_steps=max(1, round(TTL/DT))`, `alpha_saturation=V*ttl_steps*DT*sqrt(2)/R`. Above this threshold the existing ETA filter dominates the radius. Each group uses its smallest requested alpha as the representative. Per-request screening still uses remaining TTL; this grouping does not change the visibility rules.

Set `TRAIN_SEEDS` in the sweep cell (default `[SEED]`) or pass `train_seeds=[42, 43, 44]` to train several replicates of every effective setting with matching seeds across alpha values. Explicit seeds are absolute; without them the API uses `[SEED + seed_offset]`. Do not combine explicit seeds with a nonzero offset. At the current defaults there are 240 stationary or 180 non-stationary effective settings per seed, compared with the previous 420/240 nominal settings.

Raw results keep `run_id`, representative `R_PICK_ALPHA`, requested aliases in `ALPHA_MEMBERS`, `ALPHA_EFFECTIVE`, `ALPHA_SATURATION`, `train_seed`, `replicate`, and the geometry/demand/training metadata. A companion `<csv_stem>_alpha_summary.csv` reports each scenario/group's mean rate, sample standard deviation, and `n_runs`, keeping different sweep `run_id` values separate while grouping paired training seeds within the same sweep. A single replicate has no sample SD; these are training-replicate summaries, not evaluation-episode sample sizes. Fixed-seed baselines may repeat across training seeds.

Raw results and training logs also record `OBS_DIM`, `R_PAPER`, `R_PICK_ALPHA_PAPER`, `PICKUP_RADIUS`, `ALPHA_EFFECTIVE_PAPER` and `ALPHA_SATURATION_PAPER`. `PICKUP_RADIUS` is the nominal physical radius, before the ETA filter. Scenario grouping keeps different recorded input widths separate. Group summaries include effective alpha in both conventions and the paper's region size; they do not invent one nominal paper alpha for a group containing several aliases. The existing training-log header check rejects appending the expanded schema to an old log; use a new log or archive the old one before a new run.

In the revenue-vs-lambda and revenue-vs-package-count analysis helpers, PURE/PURE_OR still average run rates across alpha settings. Ride-policy and HVOR/DRL ratio helpers first average equivalent-alpha runs, then select the best group within each fixed scenario. Overview plots containing several RT or other settings average their separately selected scenario results; they do not merge different scenarios into one equivalence group. Alpha trend plots use effective alpha in code units on the horizontal axis.

New CSVs carry their own geometry, so changing parameters requires no edits to the grouping logic. Historical CSVs without `R/V/DT` use the explicitly documented `LEGACY_GEOMETRY` in the notebook and emit a warning; it currently assumes `R=5.5`, `V=0.19`, `DT=0.5` and must match that historical run. New metadata takes precedence. This regrouping does not regenerate historical simulation results or make them results of the corrected trainer.

The terminal-time/revenue scatter panels select only `GAMMA_PACK=0.50/0.83` (absolute floating-point tolerance 1e-8). Their approximately 30/50 package labels refer to expected counts 30.25/50.215 at `R=5.5`, not realized episode counts; other densities are excluded. These panels retain the original PURE/HEUR_VOR/DRL selection and combine available RT settings.

```python
df = run_param_sweep(
    LAMBDA_list,
    R_PICK_ALPHA_list,
    GAMMA_PACK_list,
    RIDE_TTL_MINUTES_list,
    MAX_VISIBLE_RIDES_list,
    SWITCH_GRACE_STEPS_list,
    RT_list,
    train_updates_per_combo=50,
    csv_path="Results/param_sweep_results_2.csv",
    train_seeds=[42],  # use [42, 43, 44] for three training replicates per effective setting
    checkpoint_every=5,
    show_progress=True,
    resume_dir=None  # to resume, replace None with the previously printed run folder
)
```



## Notes & Assumptions

- **No zoning policy**: DRL selects actions directly on the original state.
- While **en-route to pickup** or **carrying a passenger**, the chosen action is ignored (auto-continue).
- **Action masking** prevents selecting non-existent ride indices.
- The heuristic implements **switching** $N=n$: only around package deliveries does it consider rides, then resumes package delivery.

## File Map (main entry points)

- **Environment**: `CoModalEnv` (state, dynamics, rewards, TTL, visibility)
- **DRL policy**: `ActorCritic`
- **PPO**: `collect_rollout` (GAE), `ppo_update`, `train_policy_brief`
- **Baselines**: `baseline_nearby_rule` (switching), `baseline_nearby_rule_voronoi`, `baseline_four_zone`, `run_episode` (pure delivery when `policy=None`)
- **Evaluation**: `evaluate_all`
- **Sweeps & plots**: `run_param_sweep`, `plot_from_csv`
- **Shared state/analysis helpers**: `experiment_support.py` (`RolloutState`, sweep planning, alpha equivalence and group summaries); imported from either notebook working directory.
- **Persistence/progress helpers**: `training_persistence.py` (atomic group results, full training checkpoints, recovery and progress); migrate it together with the notebooks and `experiment_support.py`.
- **Directory/log selection helpers**: `project_paths.py` (scenario paths, data directory discovery, selecting one identified training curve).

## Focused Regression Checks

The recent test suite and debugging material are archived under `archive/validation_2026-09-21/`. With NumPy, pandas, OR-Tools and PyTorch installed, run `python -m unittest discover -s archive/validation_2026-09-21/tests -v`. All 53 tests pass. Coverage includes prior environment/analysis fixes, actual CPU PPO and OR-Tools, exact checkpoint recovery, dynamic alpha grouping, observation features, four-notebook initialization from both working directories, demand-cache lookup, identified results/logs/checkpoints, and executing both convergence cells without mixing runs. Some focused tests substitute plotting or evaluation collaborators. The suite does not perform full sweeps, CUDA training, full Uber demand fitting or figure rendering. Historical notebook outputs/results need separate regeneration. Archived `fixes/` and `review_2026-09-05/` contain the prior debugging snapshots/scripts, not active project entry points.
