"""One-time source-only implementation of the approved B1/B5 changes."""
import ast
import json
from pathlib import Path
from edit_notebooks import ROOT, edit_sources, replace_function

IMPORTS = '''
# Shared training state and alpha-equivalence accounting (root or NonStationary cwd).
import sys
import json
from pathlib import Path
_support_root = next(p for p in (Path.cwd(), *Path.cwd().parents)
                     if (p / "experiment_support.py").is_file())
if str(_support_root) not in sys.path:
    sys.path.insert(0, str(_support_root))
from experiment_support import (RolloutState, SWEEP_METADATA_COLUMNS, build_sweep_plan,
                                summarize_alpha_groups, with_effective_alpha,
                                best_equivalent_alpha_rate)
# Only for historical CSVs without geometry metadata; use the parameters of that run.
# New result rows carry R/V/DT and do not use this fallback.
LEGACY_GEOMETRY = {"R": 5.5, "V": 0.19, "DT": 0.5}
'''


def once(code, old, new):
    assert code.count(old) == 1, old
    return code.replace(old, new, 1)


def fix_rollout(code):
    tree = ast.parse(code)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "collect_rollout")
    source = "".join(code.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    source = once(source, "def collect_rollout(env: CoModalEnv, policy: ActorCritic, steps: int):",
                  "def collect_rollout(env: CoModalEnv, policy: ActorCritic, steps: int, state: Optional[RolloutState] = None):\n"
                  '    """Continue a training-owned state; without one, collect a fresh standalone batch."""\n'
                  "    if steps <= 0:\n        raise ValueError('steps must be positive')\n"
                  "    if state is None:\n        state = RolloutState()\n"
                  "    if state.env is not None and state.env is not env:\n"
                  "        raise ValueError('RolloutState belongs to another environment')\n"
                  "    if state.obs is None:\n"
                  "        state.env = env\n        state.obs, state.mask = env.reset()\n")
    source = once(source, "    cur_ret = 0.0; cur_len = 0\n\n    obs, mask = env.reset()\n",
                  "    cur_ret, cur_len = state.episode_return, state.episode_steps\n"
                  "    obs, mask = state.obs, state.mask\n")
    source = once(source, "    # ---------- GAE(λ) advantages & returns ----------",
                  "    state.obs, state.mask = obs, mask\n"
                  "    state.episode_return, state.episode_steps = cur_ret, cur_len\n\n"
                  "    # A batch boundary is not an episode ending. Bootstrap using this batch's\n"
                  "    # policy values; only real episode endings cut the GAE recursion.\n"
                  "    # ---------- GAE(λ) advantages & returns ----------")
    source = once(source, "    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)",
                  "    adv_std = adv_t.std() if adv_t.numel() > 1 else 0.0\n"
                  "    adv_t = (adv_t - adv_t.mean()) / (adv_std + 1e-8)")
    source = once(source, "step_avg_reward / DT", "step_avg_reward / env.dt")
    source = once(source, "t_min = ln * DT", "t_min = ln * env.dt")
    return replace_function(code, "collect_rollout", source)


SWEEP = '''def run_param_sweep(
    LAMBDA_list, R_PICK_ALPHA_list, GAMMA_PACK_list, RIDE_TTL_MINUTES_list,
    MAX_VISIBLE_RIDES_list, SWITCH_GRACE_STEPS_list, RT_list, *,
    train_updates_per_combo: int = 0, csv_path=None, seed_offset: int = 0,
    train_seeds=None,
):
    """Train once per effective setting/seed and retain all requested alpha aliases.

    train_seeds contains absolute seeds shared across alpha settings. If omitted,
    use [SEED + seed_offset]; explicit seeds must not be combined with seed_offset.
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV
    if train_seeds is not None and seed_offset != 0:
        raise ValueError("Use explicit train_seeds or seed_offset, not both")
    seeds = [SEED + seed_offset] if train_seeds is None else list(train_seeds)
    plan = build_sweep_plan(
        LAMBDA_list, R_PICK_ALPHA_list, GAMMA_PACK_list, RIDE_TTL_MINUTES_list,
        MAX_VISIBLE_RIDES_list, SWITCH_GRACE_STEPS_list, RT_list,
        R=R, v=V, dt=DT, train_seeds=seeds)
    if not plan:
        raise ValueError("The sweep grid is empty")
    print(f"[GRID] {len(plan)} training runs after alpha-equivalence grouping; seeds={seeds}")
    rows = []
    for setting in plan:
        seed = setting['train_seed']
        set_global_seeds(seed)
        env = CoModalEnv(R=R, v=V, dt=DT, rp=RP,
                         lam=setting['LAMBDA'], gamma_pack=setting['GAMMA_PACK'],
                         r_pick_alpha=setting['R_PICK_ALPHA'],
                         ride_ttl_minutes=setting['RIDE_TTL_MIN'],
                         max_visible=setting['MAX_VISIBLE_RIDES'], rt=setting['RT'], seed=seed)
        meta = dict(setting, RP=RP, HORIZON_MIN=HORIZON_MIN,
                    TRAIN_UPDATES=train_updates_per_combo, REPORT_UNSCALED=REPORT_UNSCALED,
                    DEMAND_PROFILE=(json.dumps(env.hourly_multiplier.tolist())
                                    if hasattr(env, 'hourly_multiplier') else 'stationary'))
        policy, metrics = train_policy_brief(
            env, updates=train_updates_per_combo, eval_every=5,
            heur_grace_steps=setting['SWITCH_GRACE_STEPS'],
            log_path=TRAIN_LOG_PATH, combo_meta=meta)
        for algo_key in ("drl", "heur", "heur_vor", "four_zone", "pure", "pure_or"):
            m = metrics[algo_key]
            rows.append(dict(meta, algo=algo_key.upper(),
                             reward=float(m['avg_reward']), rate=float(m['avg_rate']),
                             ep_rate=float(m['avg_ep_rate']), terminal_time=float(m['avg_t']),
                             avg_finish_time=float(m['avg_finish_time']), finish_rate=float(m['finish_rate']),
                             accepted=float(m.get('avg_acc', float('nan'))),
                             time_rides=float(m.get('avg_time_rides', float('nan'))),
                             time_delivery=float(m.get('avg_time_delivery', float('nan')))))
    df = pd.DataFrame(rows)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_path = csv_path.with_name(csv_path.stem + "_alpha_summary.csv")
    summarize_alpha_groups(df).to_csv(summary_path, index=False)
    print(f"[GRID] Wrote {len(df)} run rows to {csv_path}; group means/SD to {summary_path}")
    return df
'''


def fix_experiment(code, nonstationary=False):
    if "import torch.optim as optim" in code:
        code += IMPORTS
    if "def collect_rollout(" in code:
        code = fix_rollout(code)
    if "def train_policy_brief(" in code:
        code = once(code, "    for upd in range(1, updates + 1):\n",
                    "    rollout_state = RolloutState()\n    for upd in range(1, updates + 1):\n")
        code = once(code, "collect_rollout(env, policy, PPO_STEPS)",
                    "collect_rollout(env, policy, PPO_STEPS, rollout_state)")
        code = once(code, "    'GAMMA_PACK', 'RIDE_TTL_MIN', 'MAX_VISIBLE_RIDES', 'SWITCH_GRACE_STEPS', 'RT'\n]",
                    "    'GAMMA_PACK', 'RIDE_TTL_MIN', 'MAX_VISIBLE_RIDES', 'SWITCH_GRACE_STEPS', 'RT'\n] + SWEEP_METADATA_COLUMNS")
        code = once(code, "    with open(log_path, 'a', newline='') as f:\n",
                    "    if not write_header:\n"
                    "        with open(log_path, newline='') as existing:\n"
                    "            if next(csv.reader(existing), []) != TRAIN_LOG_COLUMNS:\n"
                    "                raise ValueError('Training log schema changed; choose a fresh log path before this run.')\n"
                    "    with open(log_path, 'a', newline='') as f:\n")
        code = replace_function(code, "run_param_sweep", SWEEP.replace(
            "DEFAULT_CSV", 'ns_results_path("param_sweep_results.csv")' if nonstationary else '"param_sweep_results.csv"'))
    if "def plot_rate_vs_lambda_optimal_alpha_by_gamma(" in code:
        code = once(code, "                    alpha_rates = filtered.groupby('R_PICK_ALPHA')['rate'].mean()\n", "")
        code = once(code, "                        rate_value = alpha_rates.max()",
                    "                        rate_value = best_equivalent_alpha_rate(filtered, legacy_geometry=LEGACY_GEOMETRY)")
        code = code.replace("Alpha-sensitive algorithms use their best average R_PICK_ALPHA at each lambda/gamma.",
                            "Ride policies optimize equivalent-alpha means within fixed scenarios, then average scenarios.")
    if "def plot_single_param_trend(" in code:
        code = once(code, "    # Keep only necessary columns\n",
                    "    if vary_key == 'R_PICK_ALPHA':\n"
                    "        df = with_effective_alpha(df, legacy_geometry=LEGACY_GEOMETRY)\n"
                    "        vary_key = 'ALPHA_EFFECTIVE'\n"
                    "    # Keep only necessary columns\n")
    if "R_PICK_ALPHA_list      =" in code:
        old_rt = next(line for line in code.splitlines(keepends=True) if line.startswith('RT_list '))
        code = once(code, old_rt, old_rt +
                    'TRAIN_SEEDS = [SEED]  # e.g. [42, 43, 44] for paired multi-seed training across effective alphas\n')
        code = once(code, '                     seed_offset=0)',
                    '                     train_seeds=TRAIN_SEEDS)')
    return code


def fix_analysis(code):
    if "# Common Initialization:" in code:
        # Shared imports must run before the other analysis cells.
        code = once(code, "import matplotlib.pyplot as plt\n", "import matplotlib.pyplot as plt\n" + IMPORTS)
    code = replace_function(code, "best_alpha_series", '''def best_alpha_series(df_in: pd.DataFrame, gamma_val, algo_name):
    """Average pure-delivery runs; optimize equivalent-alpha means for ride policies."""
    sub = df_in[(df_in["GAMMA_PACK"] == gamma_val) & (df_in["algo"] == algo_name)]
    if sub.empty:
        return np.array([]), np.array([])
    if algo_name in {"PURE", "PURE_OR"}:
        rates = sub.groupby("LAMBDA")["rate"].mean().sort_index()
        return rates.index.to_numpy(), rates.to_numpy()
    xs, ys = [], []
    for lam, group in sub.groupby("LAMBDA", sort=True):
        xs.append(lam)
        ys.append(best_equivalent_alpha_rate(group, legacy_geometry=LEGACY_GEOMETRY))
    return np.asarray(xs), np.asarray(ys)
''')
    code = replace_function(code, "best_alpha_by_lambda_gamma", '''def best_alpha_by_lambda_gamma(df_in: pd.DataFrame, lam, gamma, algo):
    """Select group means within fixed scenarios; pure delivery retains run averaging."""
    sub = df_in[(df_in["LAMBDA"] == lam) &
                (df_in["GAMMA_PACK"] == gamma) & (df_in["algo"] == algo)]
    if sub.empty:
        return np.nan
    if algo in {"PURE", "PURE_OR"}:
        return float(sub[RATE_COL].mean())
    return best_equivalent_alpha_rate(sub, rate_col=RATE_COL, legacy_geometry=LEGACY_GEOMETRY)
''')
    return replace_function(code, "best_alpha_rate", '''def best_alpha_rate(df_in, lam, gamma, algo):
    """Use the same equivalent-alpha selection for both sides of the policy ratio."""
    sub = df_in[(df_in["LAMBDA"] == lam) &
                (df_in["GAMMA_PACK"] == gamma) & (df_in["algo"] == algo)]
    if sub.empty:
        return np.nan
    if algo in {"PURE", "PURE_OR"}:
        return float(sub[RATE_COL].mean())
    return best_equivalent_alpha_rate(sub, rate_col=RATE_COL, legacy_geometry=LEGACY_GEOMETRY)
''')


if __name__ == "__main__":
    files = ("experiment.ipynb", "NonStationary/experiment2.ipynb", "analysis.ipynb", "NonStationary/analysis2.ipynb")
    snapshot = Path(__file__).with_name("before_b1_b5.json")
    assert not snapshot.exists(), "One-time migration; do not overwrite the pre-edit snapshot."
    snapshot.write_text(json.dumps({f: json.loads((ROOT / f).read_text(encoding="utf-8")) for f in files}, ensure_ascii=False), encoding="utf-8")
    for f in files[:2]:
        edit_sources(f, lambda source: fix_experiment(source, f.startswith("NonStationary/")))
    for f in files[2:]:
        edit_sources(f, fix_analysis)
