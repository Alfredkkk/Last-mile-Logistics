"""Recheck open findings against current notebooks; do not alter project code/results."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_notebook_regressions import (EXPERIMENTS, ANALYSES, experiment_definitions,
                                       definitions, notebook_tree, set_packages, np, pd)

results = []


def record(issue, filename, **evidence):
    results.append(dict(issue=issue, file=filename, **evidence))


for filename in EXPERIMENTS:
    ns = experiment_definitions(filename)
    Env = ns["CoModalEnv"]
    a, b = [Env(seed=123, lam=0) for _ in range(2)]
    for env in (a, b):
        set_packages(env, [(4, 0)])
        env.with_passenger = True
    a.drop_target = np.array([.05, 0], np.float32)
    b.drop_target = np.array([1, 0], np.float32)
    same_observation = np.array_equal(a._get_obs()[0], b._get_obs()[0])
    rewards = [a.step(0)[1], b.step(0)[1]]
    record("A5", filename, step_min=a.dt, counted_ride=a.time_rides_min,
           counted_delivery=a.time_delivery_min, ride_reward=float(rewards[0]))
    record("B2", filename, identical_observations=bool(same_observation),
           different_next_rewards=[float(r) for r in rewards])

    _, tree = notebook_tree(filename)
    rollout = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "collect_rollout")
    top_level_reset = any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                          and ast.unparse(n.value.func) == "env.reset" for n in rollout.body)
    record("B1", filename, unconditional_reset=top_level_reset,
           maximum_sampled_episode_min=ns["PPO_STEPS"] * ns["DT"],
           evaluation_horizon_min=ns["HORIZON_MIN"])

    a, b = [Env(seed=71, lam=40, r_pick_alpha=alpha) for alpha in (.25, .4)]
    equal = True
    for _ in range(200):
        action = 1 if a._visible_rides() else 0
        oa, ob = a.step(action), b.step(action)
        equal &= np.array_equal(oa[0], ob[0]) and np.array_equal(oa[4], ob[4]) and oa[1:4] == ob[1:4]
    record("B5", filename, alpha_values=[.25, .4], identical_200_steps=bool(equal),
           maximum_visible_distance=ns["V"] * ns["RIDE_TTL_MINUTES"],
           saturation_alpha=ns["V"] * ns["RIDE_TTL_MINUTES"] * np.sqrt(2) / ns["R"])

    if filename.startswith("NonStationary/"):
        definitions(filename, {"_make_eval_env_from"}, ns)
        train_env = Env(seed=71, hourly_multiplier=np.linspace(.5, 1.5, 24))
        eval_env = ns["_make_eval_env_from"](train_env)
        record("A9", filename, custom_profile_preserved=bool(np.array_equal(
            train_env.hourly_multiplier, eval_env.hourly_multiplier)))

    class DummyPolicy:
        def __init__(self, *args): self.updates = 0
        def to(self, *args): return self
        def train(self): pass
        def parameters(self): return []

    def update_stub(policy, *args):
        policy.updates += 1
        return (0., 0., 0., 0.)

    def eval_stub(env, policy, **kwargs):
        metric = dict(avg_reward=policy.updates, avg_t=1., avg_rate=policy.updates,
                      avg_ep_rate=policy.updates, finish_rate=1., avg_finish_time=1.)
        return {name: metric.copy() for name in ("drl", "heur", "heur_vor", "four_zone", "pure", "pure_or")}

    ns.update(ActorCritic=DummyPolicy, DEVICE="cpu",
              optim=types.SimpleNamespace(Adam=lambda *args, **kwargs: object()),
              collect_rollout=lambda *args: (None, [], 0., 0.), make_minibatches=lambda *args: [None],
              ppo_update=update_stub, evaluate_all=eval_stub, PPO_EPOCHS=1)
    definitions(filename, {"train_policy_brief"}, ns)
    with contextlib.redirect_stdout(io.StringIO()):
        policy, metrics = ns["train_policy_brief"](Env(seed=1, lam=0), updates=7, eval_every=5)
    record("A8", filename, model_updates=policy.updates, metrics_update=metrics["drl"]["avg_reward"],
           validation="Actual trainer control flow; training/evaluation collaborators stubbed")

for filename in ANALYSES:
    nb, _ = notebook_tree(filename)
    source = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "targets = np.array([30, 50]" in "".join(c["source"]))
    ns = dict(np=np, df=pd.DataFrame({"GAMMA_PACK": [.33, .5, .67, .83, 1.]}))
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in (
                "R_val", "df['n_est']", "targets", "df['n_round']"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
    record("A6", filename, actual_mapping=ns["df"].to_dict(orient="records"))

output = Path(__file__).with_name("remaining_results.json")
output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
