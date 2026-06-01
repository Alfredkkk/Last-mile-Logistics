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

CSV_PATH = "Results/param_sweep_results_2.csv"

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
  1) **Pickup within radius** $r_{\text{pick}}=\alpha \cdot \frac{R}{\sqrt 2}$ (`R_PICK_ALPHA = α`).
  2) **Feasible by TTL**: ETA to pickup $\le$ remaining TTL.  
     Code: `eta_min = manhattan(pos, pickup) / V` and require `eta_min <= ttl * DT`.
- At most `MAX_VISIBLE_RIDES` **closest** pickups are exposed each step.
- Unaccepted rides expire when their TTL reaches 0 (TTL is stored in **steps**; decremented by 1 per step).

### Termination
An episode ends when **either**:
- **All packages delivered** and the vehicle is neither with passenger nor en-route to pickup (`packages_done`), **or**
- **Horizon reached** (`horizon_reached`).

## Observation & Action

### Observation vector (normalized, flattened)
- Core: $[x/R,\, y/R,\, \text{time\_frac}=t/\text{HORIZON\_MIN},\, \mathbf{1}_{\text{to\_pickup}},\, \mathbf{1}_{\text{with\_pass}},\, \text{pkg\_count\_norm},\, \text{visible\_norm}]$.
- Non-stationary runs additionally include cyclic hour-of-day features: $\sin(2\pi h)$ and $\cos(2\pi h)$, where $h=(t\bmod 1440)/1440$.
- Nearest `K_NEAREST_PACK` packages: for each $(\Delta x/R,\,\Delta y/R,\, d_1/R)$.
- Up to `MAX_VISIBLE_RIDES` rides: for each $(\Delta x_{\text{pick}}/R,\,\Delta y_{\text{pick}}/R,\, d_{1,\text{pick}}/R,\,\Delta x_{\text{drop}}/R,\,\Delta y_{\text{drop}}/R,\, \text{trip\_len}/R)$.
- Zero-padded when fewer items exist.

### Action space (discrete)
- `0` = **deliver** (go toward nearest package).
- `1..MAX_VISIBLE_RIDES` = accept the $i$-th visible ride (closest pickups first).  
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

### Switching Heuristic (Zoning Policy when $N=n$)
- **Primarily deliver** the nearest package.
- **Switching window**: after a package delivery (and a one-step **pre-grace** if just about to deliver), open `SWITCH_GRACE_STEPS` decision steps:
  - If any **visible** ride exists (within $r_{\text{pick}}$ and ETA $\le$ TTL), **accept the nearest pickup**.
  - After completing that ride, immediately **return to deliver** the nearest package.
- No chaining: it does not keep accepting rides unless a new window opens around a delivery event.

### Pure Delivery
- Always chooses action `0`.


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

Every `eval_every` updates (default **5**), we print aggregated DRL/HEUR/PURE averages over `EVAL_EPISODES`:
- **R** (reward), **T** (terminal time), **Rate** (reward/min), **Acc** (accepted rides; DRL only),
- PPO diagnostics: `pi` (policy loss), `v` (value loss), `ent` (entropy).

The main aggregate rate metric is `rate` / `avg_rate`:
$$\text{avg\_rate}=\frac{\sum_i \text{reward}_i}{\sum_i \text{terminal\_time}_i}.$$
The diagnostic metric `ep_rate` / `avg_ep_rate` is the mean of per-episode rates:
$$\text{avg\_ep\_rate}=\frac{1}{N}\sum_i\frac{\text{reward}_i}{\text{terminal\_time}_i}.$$
Use `rate` for primary comparisons and `ep_rate` only to diagnose episode-level variability.

Per-episode breakdown (for each algorithm) includes:
- `reward`, `terminal_time`, per-episode `rate`, `accepted`, `ended_reason`.

Sweep CSVs store `rate` as the primary aggregate `avg_rate` and `ep_rate` as the diagnostic mean per-episode rate.

With `REPORT_UNSCALED=True`, printed **reward**/**rate** are in **original units** (descaled).


## Parameter Sweeps & Plots

Run grid experiments and save aggregated results with `run_param_sweep()`. The stationary notebook writes to `Results/`; the non-stationary notebook writes to `NonStationary/Results/` through its path helpers.

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
    seed_offset=0
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
