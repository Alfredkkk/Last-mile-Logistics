"""Check B2/B4 against the immediately preceding notebook/support snapshot."""
import ast
import copy
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_notebook_regressions import EXPERIMENTS, experiment_definitions, np, pd
from experiment_support import build_sweep_plan, summarize_alpha_groups

snapshot = json.loads(Path(__file__).with_name("before_b2_b4.json").read_text(encoding="utf-8"))


def tree(nb):
    return ast.parse("\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"))


def definition(parsed, name):
    return next(n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


trees = {}
for filename, original in snapshot["notebooks"].items():
    current = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    old = copy.deepcopy(original)
    assert len(old["cells"]) == len(current["cells"])
    changed = []
    for i, (before, after) in enumerate(zip(old["cells"], current["cells"]), 1):
        if after["cell_type"] == "code":
            ast.parse("".join(after["source"]))
            if before["source"] != after["source"]:
                changed.append(i)
            before["source"] = after["source"]
    assert old == current, "Non-source notebook content changed: " + filename
    trees[filename] = tree(current)
    if filename not in EXPERIMENTS:
        assert ast.dump(tree(original)) == ast.dump(trees[filename])
    print(f"{filename}: syntax OK; outputs/metadata preserved; edited cells {changed}")

for filename in EXPERIMENTS:
    ns = experiment_definitions(filename)
    before_tree = tree(snapshot["notebooks"][filename])
    before_cls = definition(before_tree, "CoModalEnv")
    after_cls = definition(trees[filename], "CoModalEnv")
    for method in before_cls.body:
        if isinstance(method, ast.FunctionDef) and method.name not in ("_get_obs", "obs_dim"):
            assert ast.dump(method) == ast.dump(definition(after_cls, method.name)), method.name
    for name in ("FixedPackageEnv", "baseline_four_zone", "_make_eval_env_from", "evaluate_all",
                 "collect_rollout", "train_policy_brief", "append_training_log"):
        assert ast.dump(definition(before_tree, name)) == ast.dump(definition(trees[filename], name)), name
    old_ns = dict(ns)
    exec(compile(ast.Module(body=[before_cls], type_ignores=[]), filename, "exec"), old_ns)
    options = dict(seed=917, lam=8., gamma_pack=1.)
    if filename.startswith("NonStationary/"):
        options["hourly_multiplier"] = np.linspace(.5, 1.5, 24)
    old_env, new_env = old_ns["CoModalEnv"](**options), ns["CoModalEnv"](**options)
    core = 9 if filename.startswith("NonStationary/") else 7
    prefix = core + 3 * new_env.k_pack

    def original_features(obs):
        return np.concatenate([obs[:prefix], obs[prefix:-7].reshape(new_env.max_visible, 7)[:, :6].ravel()])

    dropoffs = 0
    for step in range(1000):
        old_obs, old_mask = old_env._get_obs()
        new_obs, new_mask = new_env._get_obs()
        np.testing.assert_array_equal(old_obs, original_features(new_obs))
        np.testing.assert_array_equal(old_mask, new_mask)
        action = 1 if step % 4 < 2 and new_mask[1] else 0
        carrying = new_env.with_passenger
        before, after = old_env.step(action), new_env.step(action)
        dropoffs += int(carrying and not new_env.with_passenger)
        np.testing.assert_array_equal(before[0], original_features(after[0]))
        assert before[1:4] == after[1:4]
        np.testing.assert_array_equal(before[4], after[4])
        np.testing.assert_array_equal(old_env.pos, new_env.pos)
        np.testing.assert_array_equal(old_env.pkg_delivered, new_env.pkg_delivered)
        assert old_env.accepted_rides == new_env.accepted_rides
        assert old_env.revenue_cum == new_env.revenue_cum
        assert old_env.t == new_env.t
        if before[2]:
            break
    assert dropoffs > 0
    print(f"{filename}: {step + 1} fixed-action steps unchanged, including {dropoffs} dropoffs; only observation features added")

for name in ("_accepted_ride_features",):
    methods = [definition(definition(trees[f], "CoModalEnv"), name) for f in EXPERIMENTS]
    assert ast.dump(methods[0]) == ast.dump(methods[1]), name

original_support = types.ModuleType("original_b2_b4_support")
sys.modules[original_support.__name__] = original_support
exec(snapshot["support_source"], original_support.__dict__)
for R, v, dt, ttl in ((5.5, .19, .5, 5.), (3., .4, .3, 1.4), (7., .1, 1., 0.)):
    args = ([1., 5.], [.1, .2, .25, .3, .4, 1.], [.075, .5], [ttl], [3, 5], [0, 2], [5.5])
    kwargs = dict(R=R, v=v, dt=dt, train_seeds=[42, 53])
    old_plan = original_support.build_sweep_plan(*args, **kwargs)
    new_plan = build_sweep_plan(*args, **kwargs)
    assert old_plan == [{k: row[k] for k in old_plan[0]} for row in new_plan]
    df = pd.DataFrame([dict(row, algo="DRL", rate=i / 10.) for i, row in enumerate(new_plan)])
    old_summary = original_support.summarize_alpha_groups(df)
    new_summary = summarize_alpha_groups(df)
    pd.testing.assert_frame_equal(old_summary, new_summary[old_summary.columns])
print("B5 sweep groups, seeds, thresholds and numeric summaries unchanged by paper-unit metadata")
