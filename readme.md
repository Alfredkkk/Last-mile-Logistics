# DRL for Joint Ride-Hailing & Package Delivery

This project simulates a **single vehicle** operating in a diamond-shaped city (L1 metric) that must decide, at each step, whether to **deliver packages** or **accept a passenger ride**. Passenger requests arrive as a **spatially homogeneous Poisson process**; packages are drawn from a **spatial Poisson field**. We compare a **Deep RL (PPO)** policy against a **switching heuristic** (paper’s $N=n$ case), a **pure delivery** and a **heuristic zoning policy** baseline.

The project and part of the analysis is in the `experiment.ipynb` file.

Formal analysis in `analysis.ipynb` file.

This project is based on paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4565248.

For non-stationary scenario, the data we prepare to use is https://www.kaggle.com/datasets/shuhengmo/uber-nyc-forhire-vehicles-trip-data-2021

The readme file may not be up-to-date. If you observe some rendering issues of the Readme file on the Github page, cloning this repo to your local device should fix them.

## Quick Start

Train & evaluate a single setting (inside a notebook/script):

```python
# Un-comment in code if running as a script:
# if __name__ == "__main__":
#     main()

# Or call main() directly:
main()
```
Sweep parameters and write results to CSV with plots:
```python
df = run_param_sweep(
    LAMBDA_list, R_PICK_ALPHA_list, GAMMA_PACK_list,
    RIDE_TTL_MINUTES_list, MAX_VISIBLE_RIDES_list, SWITCH_GRACE_STEPS_list,
    train_updates_per_combo=50,                 # PPO updates per combo (not episodes)
    csv_path="param_sweep_results.csv",
    seed_offset=0
)
plot_from_csv(
    "param_sweep_results.csv",
    vary_keys=["LAMBDA","R_PICK_ALPHA","GAMMA_PACK","RIDE_TTL_MIN","MAX_VISIBLE_RIDES","SWITCH_GRACE_STEPS"],
    metrics=("rate","reward","accepted")
)

```
Which is the last cell of this Jupyter Notebook.

## Environment & Dynamics

### City geometry
- **Diamond region** (L1 ball): feasible $(x,y)$ satisfy $|x|+|y|\le R$.
- **Distance**: Manhattan $d_1(a,b)=|a_x-b_x|+|a_y-b_y|$.

### Time & motion
- **Step size**: `DT` minutes per step.
- **Vehicle speed**: `V` distance units per minute.
- **Horizon**: `HORIZON_MIN` minutes max per episode; steps per episode `STEPS_PER_EP = HORIZON_MIN / DT`.

### Stochastic primitives
- **Packages**: on reset, sample $N\sim \text{Poisson}(\gamma R^2)$; locations IID uniform in the diamond  
  (`GAMMA_PACK = γ` is **intensity per unit area**; diamond area under L1 is $R^2$.
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
- Core: $$[x/R,\, y/R,\, \text{time\_frac}=t/\text{HORIZON\_MIN},\, \mathbf{1}_{\text{to\_pickup}},\, \mathbf{1}_{\text{with\_pass}},\, \text{pkg\_count\_norm},\, \text{visible\_norm}]$$.
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
If `REPORT_UNSCALED=True`, all printed **reward** and **rate = reward / terminal_time** are **descaled** by `INV_REWARD_SCALE` to original units.

> Units: distance is L1; `V` is distance/min; `DT` is min/step; reward is in revenue units (ride: per distance, package: per item). Printed **Rate** is per minute of actual episode time.


## Policies

### DRL (PPO; greedy at eval)
- **Network**: shared MLP trunk (256 units × 2) with **LayerNorm → ReLU → Dropout(0.1)**; heads:
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
- `R = 5.5` (L1 radius), `V = 1.0` (distance/min), `DT = 0.5` (min/step), `HORIZON_MIN = 1440.0`.

**Demand & revenue**
- `LAMBDA = 0.60` (rides per minute), `GAMMA_PACK = 0.8` (packages per area),
- `R_PICK_ALPHA = 0.5` (pickup visibility $r_{\text{pick}} = \alpha R/\sqrt 2$),
- `RIDE_TTL_MINUTES = 5`,
- `RT = 8.0` (ride revenue / distance), `RP = 1.0` (package revenue / item),
- `REWARD_SCALE = 1/8`, `REPORT_UNSCALED = True`.

**DRL (PPO)**
- `MAX_VISIBLE_RIDES = 5`, `K_NEAREST_PACK = 10`,
- `DISCOUNT = 0.99`, `GAE_LAMBDA = 0.95`,
- `PPO_STEPS = 4096`, `PPO_MINI_BATCH = 256`, `PPO_EPOCHS = 4`,
- `CLIP_EPS = 0.2`, `VF_COEF = 0.5`, `ENT_COEF = 0.02`, `LR = 3e-4`,
- `UPDATES = 200`.

**Heuristic**
- `SWITCH_GRACE_STEPS = 6` (window length after delivery to consider rides),
- `HEUR_SOFT_PICK_CAP = None` (optional: accept if pickup extremely close).


## Logging & Metrics

Every `eval_every` updates (default **5**), we print aggregated DRL/HEUR/PURE averages over `EVAL_EPISODES`:
- **R** (reward), **T** (terminal time), **Rate** (reward/min), **Acc** (accepted rides; DRL only),
- PPO diagnostics: `pi` (policy loss), `v` (value loss), `ent` (entropy).

Per-episode breakdown (for each algorithm) includes:
- `reward`, `terminal_time`, `rate`, `accepted`, `ended_reason`.

With `REPORT_UNSCALED=True`, printed **reward**/**rate** are in **original units** (descaled).


## Parameter Sweeps & Plots

Run grid experiments and save aggregated results:

```python
df = run_param_sweep(
    LAMBDA_list, R_PICK_ALPHA_list, GAMMA_PACK_list,
    RIDE_TTL_MINUTES_list, MAX_VISIBLE_RIDES_list, SWITCH_GRACE_STEPS_list,
    train_updates_per_combo=50,                 # PPO updates per combo
    csv_path="param_sweep_results.csv",
    seed_offset=0                               # change to 1,2,... for independent repeats
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
- **PPO**: `collect_rollout` (GAE), `ppo_update`, `main`
- **Baselines**: `baseline_nearby_rule` (switching), `run_episode` (pure delivery when `policy=None`)
- **Evaluation**: `evaluate_all`
- **Sweeps & plots**: `run_param_sweep`, `plot_from_csv`
