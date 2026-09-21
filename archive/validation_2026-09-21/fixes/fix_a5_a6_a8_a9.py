"""Source-only fixes approved after the remaining-issues discussion."""
import json
from pathlib import Path

from edit_notebooks import ROOT, edit_sources


def replace_once(code, old, new):
    assert code.count(old) == 1, old
    return code.replace(old, new, 1)


def fix_experiment(code, nonstationary=False):
    if "class CoModalEnv:" in code:
        code = replace_once(code, "        delivered_step = 0 # track how many packages delivered in this step\n",
                            "        delivered_step = 0 # track how many packages delivered in this step\n"
                            "        is_ride_phase = False  # Record this step's executed activity, not its ending state.\n")
        code = code.replace("        # --- At the beginning of CoModalEnv.step(self, action), determine the attribution of this step ---\n\n\n        \n", "")
        code = replace_once(code, "        if self.to_pickup is not None:\n            # move towards pickup\n",
                            "        if self.to_pickup is not None:\n            is_ride_phase = True\n            # move towards pickup\n")
        code = replace_once(code, "        elif self.with_passenger:\n            # move towards dropoff; accrue ride revenue per distance traveled\n",
                            "        elif self.with_passenger:\n            is_ride_phase = True\n            # move towards dropoff; accrue ride revenue per distance traveled\n")
        code = replace_once(code, "                # move towards pickup immediately this step\n",
                            "                is_ride_phase = True\n                # move towards pickup immediately this step\n")
        code = replace_once(code,
                            "        #make sure the time is recorded correctly, aka is_ride_phase is computed after the action is taken\n"
                            "        is_ride_phase = (self.to_pickup is not None) or self.with_passenger\n",
                            "        # Pickup and passenger travel, including the final dropoff step, are ride time.\n"
                            "        # An accepted request waiting for zone delivery to finish remains delivery time.\n")

    if "def train_policy_brief(" in code:
        code = replace_once(code,
                            "    Trainer: every `eval_every` updates call evaluate_all (same style as main()) and print logs.\n"
                            "    Returns (policy, last_metrics).\n",
                            "    Evaluate periodically and after the final update, logging each evaluation once.\n"
                            "    With no updates, evaluate/log the initialized model at update 0.\n"
                            "    Returns (policy, last_metrics) for the same model state.\n")
        old_log = """            if log_path:
                eval_meta = (combo_meta or {}).copy()
                eval_ep_stats = [{'reward': drl['avg_reward'], 'rate': drl['avg_rate'], 'ep_rate': drl['avg_ep_rate'], 'terminal_time_min': drl['avg_t'], 'steps': float('nan')}]
                append_training_log(log_path, upd, 'eval', eval_ep_stats, None, None, eval_meta,
                                    values_are_unscaled=REPORT_UNSCALED)

"""
        code = replace_once(code, old_log, "")
        code = replace_once(code, """    if updates <= 0:
        last_metrics = evaluate_all(env, policy, heur_grace_steps=heur_grace_steps)
        return policy, last_metrics
""", """    def evaluate_and_log(update):
        metrics = evaluate_all(env, policy, heur_grace_steps=heur_grace_steps)
        if log_path:
            drl = metrics["drl"]
            eval_meta = (combo_meta or {}).copy()
            eval_ep_stats = [{'reward': drl['avg_reward'], 'rate': drl['avg_rate'], 'ep_rate': drl['avg_ep_rate'], 'terminal_time_min': drl['avg_t'], 'steps': float('nan')}]
            append_training_log(log_path, update, 'eval', eval_ep_stats, None, None, eval_meta,
                                values_are_unscaled=REPORT_UNSCALED)
        return metrics

    if updates <= 0:
        return policy, evaluate_and_log(0)
""")
        code = replace_once(code, """        if eval_every and (upd % eval_every == 0):
            metrics = evaluate_all(env, policy, heur_grace_steps=heur_grace_steps)
""", """        if (eval_every and upd % eval_every == 0) or upd == updates:
            metrics = evaluate_and_log(upd)
""")
        code = replace_once(code, """    if last_metrics is None:
        last_metrics = evaluate_all(env, policy, heur_grace_steps=heur_grace_steps)
    return policy, last_metrics
""", "    return policy, last_metrics\n")

    if nonstationary and "def _make_eval_env_from(" in code:
        code = replace_once(code, "        max_visible=env.max_visible,\n        seed=0\n",
                            "        max_visible=env.max_visible,\n"
                            "        hourly_multiplier=env.hourly_multiplier.copy(),\n"
                            "        seed=0\n")
    return code


SCATTER_HELPER = '''def select_scatter_package_sizes(df_in: pd.DataFrame) -> pd.DataFrame:
    """Select the gamma=.50/.83 settings (E[N]=30.25/50.215 for R=5.5).

    Panel labels 30 and 50 are approximate expected counts, not realized counts.
    Other package densities must not be rounded into these two groups.
    """
    gamma = df_in["GAMMA_PACK"].to_numpy(dtype=float)
    labels = np.select([np.isclose(gamma, .50, rtol=0, atol=1e-8),
                        np.isclose(gamma, .83, rtol=0, atol=1e-8)],
                       [30, 50], default=-1)
    keep = labels != -1
    selected = df_in.loc[keep].copy()
    selected["n_label"] = labels[keep]
    return selected


'''


def fix_analysis(code):
    if "# Scatter: termination time vs revenue rate (n=30/50, lambda=10/40)" not in code:
        return code
    code = replace_once(code, "import numpy as np\n\n", "import numpy as np\n\n" + SCATTER_HELPER)
    code = replace_once(code,
                        "targets = np.array([30, 50], dtype=float)\n"
                        "df['n_round'] = df['n_est'].apply(lambda x: targets[np.argmin(np.abs(targets - x))])\n",
                        "df = select_scatter_package_sizes(df)\n")
    code = replace_once(code, "sub = df[(df['n_round'] == n)", "sub = df[(df['n_label'] == n)")
    code = replace_once(code, "        ax.set_title(f'n={n}, lambda={lam}')\n",
                        "        ax.set_title(f'Expected packages ≈ {n}, lambda={lam}')\n")
    return code


if __name__ == "__main__":
    files = ("experiment.ipynb", "NonStationary/experiment2.ipynb", "analysis.ipynb", "NonStationary/analysis2.ipynb")
    snapshot = Path(__file__).with_name("before_a_class.json")
    assert not snapshot.exists(), "Keep the original snapshot; this edit script is a one-time migration."
    snapshot.write_text(json.dumps({name: json.loads((ROOT / name).read_text(encoding="utf-8"))
                                    for name in files}, ensure_ascii=False), encoding="utf-8")
    for filename in files[:2]:
        edit_sources(filename, lambda code: fix_experiment(code, filename.startswith("NonStationary/")))
    for filename in files[2:]:
        edit_sources(filename, fix_analysis)
