"""Compare persistence changes with the immediately preceding notebook snapshot."""
import ast
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
snapshot = json.loads(Path(__file__).with_name("before_c1.json").read_text(encoding="utf-8"))
trees = []
for filename, original in snapshot["notebooks"].items():
    current = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    restored = copy.deepcopy(original)
    assert len(restored["cells"]) == len(current["cells"])
    for before, after in zip(restored["cells"], current["cells"]):
        if after["cell_type"] == "code":
            ast.parse("".join(after["source"]))
            before["source"] = after["source"]
    assert restored == current, filename
    def definitions(nb):
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        return {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    before, after = definitions(original), definitions(current)
    assert before.keys() == after.keys()
    for name in before.keys() - {"run_param_sweep", "train_policy_brief"}:
        assert ast.dump(before[name]) == ast.dump(after[name]), (filename, name)
    trees.append(after)
    print(filename + ": outputs/metadata preserved; all non-trainer/sweep definitions unchanged")
for name in ("train_policy_brief", "run_param_sweep"):
    assert ast.dump(trees[0][name]) == ast.dump(trees[1][name]), name
assert (ROOT / "experiment_support.py").read_text(encoding="utf-8") == snapshot["support_source"]
print("Shared trainer/sweep implementations match; B1/B5/B4 support logic unchanged")

if "--training" in sys.argv:
    import contextlib
    import io
    import tempfile
    import torch
    sys.path.insert(0, str(ROOT / "tests"))
    from test_training_persistence import small_training, seed_all, environment, assert_nested
    import unittest
    for filename, original in snapshot["notebooks"].items():
        old_ns, _ = small_training(filename)
        old_function = definitions(original)["train_policy_brief"]
        exec(compile(ast.Module(body=[old_function], type_ignores=[]), filename, "exec"), old_ns)
        with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as directory:
            seed_all(71, old_ns)
            old_policy, old_metrics = old_ns["train_policy_brief"](environment(old_ns), 3, eval_every=2)
            new_ns, _ = small_training(filename)
            seed_all(71, new_ns)
            new_policy, new_metrics = new_ns["train_policy_brief"](environment(new_ns), 3, eval_every=2,
                checkpoint_dir=directory, checkpoint_every=1, show_progress=False)
        case = unittest.TestCase()
        assert_nested(case, old_policy.state_dict(), new_policy.state_dict())
        assert_nested(case, old_metrics, new_metrics)
        print(filename + ": before/after real CPU PPO weights and metrics exactly match")
