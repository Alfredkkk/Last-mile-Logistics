"""Preserve notebook outputs/model behavior while auditing all current paths."""
import ast
import copy
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_paths.py").is_file())
snapshot = json.loads(Path(__file__).with_name("before_paths_runid.json").read_text(encoding="utf-8"))
audit = {}
for filename, original in snapshot["notebooks"].items():
    current = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    baseline = copy.deepcopy(original)
    assert len(baseline["cells"]) == len(current["cells"])
    for before, after in zip(baseline["cells"], current["cells"]):
        if after["cell_type"] == "code":
            ast.parse("".join(after["source"]))
            before["source"] = after["source"]
    assert baseline == current, filename
    def definitions(nb):
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        return code, {n.name: n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    old_code, before = definitions(original)
    code, after = definitions(current)
    if "experiment" in filename:
        for name in ("CoModalEnv", "FixedPackageEnv", "ActorCritic", "collect_rollout", "ppo_update",
                     "evaluate_all", "baseline_four_zone", "baseline_pure_ortools"):
            assert ast.dump(before[name]) == ast.dump(after[name]), (filename, name)
    assert "TRAIN_LOG_PATH =" not in code
    assert "ns_results_path(" not in code
    assert "ns_path(" not in code
    assert "Results/training_log.csv" not in code
    assert "Results/param_sweep_results" not in code
    audit[filename] = {"code_cells_parse": True, "outputs_metadata_preserved": True,
                       "scenario_paths": "NonStationary/Results" if filename.startswith("NonStationary/") else "Results",
                       "model_logic_unchanged": "experiment" in filename}
    print(filename + ": source parses; outputs/metadata preserved; scenario paths checked")
report = Path(__file__).with_name("paths_runid_audit.json")
report.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("Environment, policies, evaluation and PPO sampling/update definitions unchanged")
