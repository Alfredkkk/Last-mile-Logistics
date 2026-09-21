"""Static and notebook-integration checks against the immediate pre-B1/B5 snapshot."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment_support import build_sweep_plan

snapshot = json.loads(Path(__file__).with_name("before_b1_b5.json").read_text(encoding="utf-8"))
trees = {}
for filename, original in snapshot.items():
    current = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    old = copy.deepcopy(original)
    changed = []
    for i, (before, after) in enumerate(zip(old["cells"], current["cells"]), 1):
        if after["cell_type"] == "code":
            ast.parse("".join(after["source"]))
            if before["source"] != after["source"]:
                changed.append(i)
            before["source"] = after["source"]
    assert old == current, filename
    text = "\n".join("".join(c["source"]) for c in current["cells"] if c["cell_type"] == "code")
    tree = trees[filename] = ast.parse(text)
    initial = "".join(current["cells"][0]["source"])
    start = initial.index("# Shared training state")
    end = initial.index('\n', initial.index('LEGACY_GEOMETRY =', start))
    imports = initial[start:end] + '\nprint("shared imports OK")'
    # Execute the actual bootstrap from both notebook working directories.
    for cwd in (ROOT, ROOT / "NonStationary"):
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", imports], cwd=cwd,
                                check=True, capture_output=True, text=True)
        assert "shared imports OK" in result.stdout
    print(f"{filename}: source parses; outputs/metadata unchanged; imports work; edited cells {changed}")
    if "experiment" in filename:
        original_tree = ast.parse("\n".join("".join(c["source"]) for c in original["cells"] if c["cell_type"] == "code"))
        for name in ("CoModalEnv", "FixedPackageEnv", "baseline_four_zone", "_make_eval_env_from", "evaluate_all"):
            nodes = [next(n for n in t.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
                     for t in (original_tree, tree)]
            assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
        print("  Environment, visibility, evaluation and FOUR_ZONE behavior definitions unchanged")
        ns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    (t.id.endswith('_list') or t.id in {'R', 'V', 'DT'}) for t in node.targets):
                try:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), ns)
                except NameError:
                    pass
        plan = build_sweep_plan(ns['LAMBDA_list'], ns['R_PICK_ALPHA_list'], ns['GAMMA_PACK_list'],
                               ns['RIDE_TTL_MINUTES_list'], ns['MAX_VISIBLE_RIDES_list'],
                               ns['SWITCH_GRACE_STEPS_list'], ns['RT_list'],
                               R=ns['R'], v=ns['V'], dt=ns['DT'], train_seeds=[42])
        print(f"  Current grid: {len(plan)} effective parameter settings per training seed")

for name in ("collect_rollout", "train_policy_brief", "append_training_log"):
    nodes = [next(n for n in trees[f].body if isinstance(n, ast.FunctionDef) and n.name == name)
             for f in ("experiment.ipynb", "NonStationary/experiment2.ipynb")]
    assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
print("Shared rollout/trainer/logger implementations agree across scenarios")
