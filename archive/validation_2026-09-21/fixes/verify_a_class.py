"""Compare this fix with its saved pre-edit notebooks, not an older Git HEAD."""
import ast
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_notebook_regressions import EXPERIMENTS, experiment_definitions, np

snapshot = json.loads(Path(__file__).with_name("before_a_class.json").read_text(encoding="utf-8"))
trees = {}
for filename, original in snapshot.items():
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
    trees[filename] = ast.parse("\n".join("".join(c["source"]) for c in current["cells"] if c["cell_type"] == "code"))
    print(f"{filename}: syntax OK; outputs/metadata preserved; edited cells {changed}")

for filename in EXPERIMENTS:
    ns = experiment_definitions(filename)
    source = "\n".join("".join(c["source"]) for c in snapshot[filename]["cells"] if c["cell_type"] == "code")
    original_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CoModalEnv")
    old_ns = dict(ns)
    exec(compile(ast.Module(body=[original_class], type_ignores=[]), filename, "exec"), old_ns)
    options = dict(seed=917, lam=8., gamma_pack=1.)
    if filename.startswith("NonStationary/"):
        options["hourly_multiplier"] = np.linspace(.5, 1.5, 24)
    old_env, new_env = old_ns["CoModalEnv"](**options), ns["CoModalEnv"](**options)
    dropoffs = 0
    for step in range(1000):
        old_obs, old_mask = old_env._get_obs()
        new_obs, new_mask = new_env._get_obs()
        np.testing.assert_array_equal(old_obs, new_obs)
        np.testing.assert_array_equal(old_mask, new_mask)
        action = 1 if step % 4 < 2 and new_mask[1] else 0
        carrying = new_env.with_passenger
        before, after = old_env.step(action), new_env.step(action)
        dropoffs += int(carrying and not new_env.with_passenger)
        np.testing.assert_array_equal(before[0], after[0])
        assert before[1:3] == after[1:3]
        clean_info = lambda info: {k: v for k, v in info.items() if k not in ("time_rides_min", "time_delivery_min")}
        assert clean_info(before[3]) == clean_info(after[3])
        np.testing.assert_array_equal(before[4], after[4])
        np.testing.assert_array_equal(old_env.pos, new_env.pos)
        np.testing.assert_array_equal(old_env.pkg_delivered, new_env.pkg_delivered)
        assert old_env.accepted_rides == new_env.accepted_rides
        assert old_env.revenue_cum == new_env.revenue_cum
        assert old_env.t == new_env.t
        assert new_env.time_rides_min - old_env.time_rides_min == dropoffs * new_env.dt
        assert old_env.time_delivery_min - new_env.time_delivery_min == dropoffs * new_env.dt
        assert new_env.time_rides_min + new_env.time_delivery_min == new_env.t
        if before[2]:
            break
    assert dropoffs > 0
    print(f"{filename}: {step + 1} steps match movement/observations/rewards; {dropoffs} dropoff steps correctly reclassified")

for name in ("baseline_four_zone", "solve_zone_tsp_L1", "baseline_pure_ortools", "train_policy_brief"):
    nodes = [next(n for n in trees[f].body if isinstance(n, ast.FunctionDef) and n.name == name) for f in EXPERIMENTS]
    assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
steps = []
for filename in EXPERIMENTS:
    cls = next(n for n in trees[filename].body if isinstance(n, ast.ClassDef) and n.name == "CoModalEnv")
    steps.append(ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")))
assert steps[0] == steps[1]
print("Shared environment step, trainer and routing definitions match across scenarios")
