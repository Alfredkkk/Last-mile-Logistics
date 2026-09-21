"""Final source/metadata and ordinary-environment compatibility checks."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_notebook_regressions import ANALYSES, EXPERIMENTS, experiment_definitions, np

for filename in (*ANALYSES, *EXPERIMENTS):
    old = json.loads(subprocess.check_output(["git", "show", "HEAD:" + filename], cwd=ROOT))
    new = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    assert len(old["cells"]) == len(new["cells"])
    for before, after in zip(old["cells"], new["cells"]):
        if after["cell_type"] == "code":
            ast.parse("".join(after["source"]))
            before["source"] = after["source"]
    assert old == new, "Non-source notebook content changed: " + filename
    print(filename + ": all cells parse; stored outputs and metadata preserved")

for filename in EXPERIMENTS:
    ns = experiment_definitions(filename)
    original = json.loads(subprocess.check_output(["git", "show", "HEAD:" + filename], cwd=ROOT))
    source = "\n".join("".join(c["source"]) for c in original["cells"] if c["cell_type"] == "code")
    original_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "CoModalEnv")
    old_ns = dict(ns)
    exec(compile(ast.Module(body=[original_class], type_ignores=[]), filename, "exec"), old_ns)
    options = dict(seed=917, lam=8., gamma_pack=1.)
    if filename.startswith("NonStationary/"):
        options["hourly_multiplier"] = np.linspace(.5, 1.5, 24)
    old_env, new_env = old_ns["CoModalEnv"](**options), ns["CoModalEnv"](**options)
    for step in range(200):
        old_obs, old_mask = old_env._get_obs()
        new_obs, new_mask = new_env._get_obs()
        np.testing.assert_array_equal(old_obs, new_obs)
        np.testing.assert_array_equal(old_mask, new_mask)
        action = 1 if step % 4 < 2 and new_mask[1] else 0
        before, after = old_env.step(action), new_env.step(action)
        np.testing.assert_array_equal(before[0], after[0])
        assert before[1:4] == after[1:4]
        np.testing.assert_array_equal(before[4], after[4])
        if before[2]:
            break
    print(filename + f": {step + 1} ordinary-action steps match pre-change observations, masks, rewards and info")

stationary, nonstationary = [experiment_definitions(f) for f in EXPERIMENTS]
for name in ("baseline_four_zone", "solve_zone_tsp_L1", "_l1_distance", "baseline_pure_ortools"):
    sources = []
    for filename in EXPERIMENTS:
        nb = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        tree = ast.parse("\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"))
        sources.append(ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)))
    assert sources[0] == sources[1], name
print("Shared routing and FOUR_ZONE definitions match in both notebooks")
