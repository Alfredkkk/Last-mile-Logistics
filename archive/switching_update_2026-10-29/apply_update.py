"""Apply the reviewed switching source without replacing saved notebook outputs."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "switching_source.py").read_text(encoding="utf-8").rstrip() + "\n\n\n"
DATE_HEADING = "## 2026-10-29 Switching Windows and Package-First Reservations"


def edit_notebook(root, relative):
    path = root / relative
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original = copy.deepcopy(notebook)
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code"
                and "def baseline_nearby_rule(" in "".join(c["source"]))
    source = "".join(cell["source"])
    start_name = "_baseline_switching" if "def _baseline_switching(" in source else "baseline_nearby_rule"
    before_name = "nonstationary_baselines_before.py" if relative.startswith("NonStationary") else "stationary_baselines_before.py"
    expected = HERE / before_name
    if start_name == "baseline_nearby_rule" and expected.exists():
        if source != expected.read_text(encoding="utf-8"):
            raise ValueError("Baseline code has changed since review: " + str(path))
    tree = ast.parse(source)
    start = next(n.lineno - 1 for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == start_name)
    stop = next(n.lineno - 1 for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "baseline_pure_ortools")
    lines = source.splitlines(keepends=True)
    updated = "".join(lines[:start]) + SOURCE + "".join(lines[stop:])
    ast.parse(updated)
    cell["source"] = updated.splitlines(keepends=True)
    # Only this cell's source may differ; outputs, execution counts, other cells
    # and notebook metadata must survive even when applying to another copy.
    check = copy.deepcopy(notebook)
    index = notebook["cells"].index(cell)
    check["cells"][index]["source"] = original["cells"][index]["source"]
    assert check == original
    evidence = root / "archive" / "switching_update_2026-10-29"
    evidence.mkdir(parents=True, exist_ok=True)
    before_path = evidence / ("nonstationary_baselines_before.py" if relative.startswith("NonStationary") else "stationary_baselines_before.py")
    if source != updated and not before_path.exists():
        before_path.write_text(source, encoding="utf-8")
    if source != updated:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"file": relative, "changed": source != updated,
            "source_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "saved_outputs_and_other_cells_preserved": True}


def replace_once(text, old, new):
    if new in text:
        return text
    if text.count(old) != 1:
        raise ValueError("Expected exactly one documentation fragment: " + old[:80])
    return text.replace(old, new, 1)


def edit_docs(root):
    path = root / "readme.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text,
        "of the current delivery target, not the vehicle's current position.",
        "of the policy's delivery reference (defined below), not the vehicle's current position.")
    text = replace_once(text,
        "ETA from that delivery target to pickup",
        "ETA from that delivery reference to pickup")
    text = replace_once(text,
        "- The default visibility reference is the package delivery target currently used by `_nearest_package()`; if no package target exists, the environment falls back to the vehicle position. FOUR_ZONE explicitly screens from the final delivery point of its current zone route.",
        "- The default DRL visibility reference is the package delivery target currently used by `_nearest_package()`; if no package target exists, the environment falls back to the vehicle position. HEUR/HEUR_VOR explicitly keep each screening window anchored to its package address, including the retained post-delivery window. FOUR_ZONE explicitly screens from the final delivery point of its current zone route.")
    text = replace_once(text,
        "- TTL is the lifetime of an **unaccepted** request. An accepted FOUR_ZONE request is removed from the buffer and remains committed while the current zone's deliveries finish.",
        "- TTL is the lifetime of an **unaccepted** request. Advance-accepted HEUR/HEUR_VOR and FOUR_ZONE requests are removed from the buffer and remain committed while the associated package or zone finishes; pickup starts afterward.")
    start = text.index("### Switching Heuristic")
    stop = text.index("\n### ", start + 4)
    text = text[:start] + """### Switching Heuristics (modified paper N=n policies)
- HEUR and HEUR_VOR share `_baseline_switching`. Both retain the one-step pre-grace trigger (`distance <= V*DT`, currently 0.095) and the intentional post-delivery window (`SWITCH_GRACE_STEPS`, currently 6 decisions / 3 minutes). The extra post-window remains a research extension, not a strict paper reproduction.
- A window stays anchored to its own package A. Pre-grace screening uses A; after A is delivered, the retained window still uses A, even as an unmatched vehicle moves toward B. Delivering B starts a fresh B window. Merely selecting B as the next delivery target never changes an active A window.
- In pre-grace, accept a visible request by identity and reserve it immediately, finish the locked package A, then start pickup. TTL no longer applies after acceptance. An accepted final-package reservation must still be served unless the simulation reaches its horizon.
- In the post-A window, A is already delivered, so an accepted request starts pickup immediately. Without a match, continue toward the nearest remaining package while the window counts down; do not wait in place.
- Pickup and passenger travel do not open, decrement or refresh grace windows. After dropoff, resume package service; the ordinary pre-grace rule can open a new window near its next delivery. An actual package delivery at a coincident dropoff is still a delivery event.
- HEUR accepts the pickup closest to the fixed reference, with no destination restriction. Its optional `HEUR_SOFT_PICK_CAP` is retained (currently disabled).
- HEUR_VOR additionally rejects dropoffs owned by already-delivered L1 Voronoi cells and the current pre-grace package's cell (including coincident packages), since those will be complete before passenger service. The existing visibility/ETA/top-K filters still run first.
- Both wrappers retain their public signatures and metric tuple. DRL, FOUR_ZONE, rewards, horizon and observation/action dimensions are unchanged. Saved notebook outputs and historical CSVs still describe earlier runs until reevaluation.
""" + text[stop:]
    path.write_text(text, encoding="utf-8")

    path = root / "debug_log.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text,
        "10. `baseline_nearby_rule()` uses a pre-grace window that may accept a ride before actually completing the package delivery, which is more aggressive than the paper's switching description. Status: accepted intentional modeling change; keep current behavior and document as a modified/pre-grace switching heuristic.",
        "10. Switching previously diverted immediately when accepting in pre-grace. Early acceptance itself is permitted by the paper; leaving before completing the current package was the service-order deviation. Status: revised by user decision on 2026-10-29. HEUR/HEUR_VOR now reserve the request, finish the current package, then pick up. Retain the one-step pre-grace trigger and the extra post-delivery window as intentional extensions; see the October 29 entry below.")
    text = replace_once(text,
        "11. Ride visibility is computed from current vehicle position, not the final delivery location used in the paper's screening logic. Status: fixed on 2026-06-09 after advisor confirmation by adding a delivery-target visibility reference in both experiment notebooks and aligning heuristic ride selection with that reference.",
        "11. Ride visibility originally used vehicle position rather than a delivery reference; the June 9 fix introduced a delivery-target reference. Follow-up fixed on 2026-10-29: HEUR/HEUR_VOR windows now retain their own package reference after delivery instead of immediately switching to the next remaining package. DRL keeps its existing nearest-undelivered-package reference; FOUR_ZONE keeps its final zone-delivery reference.")
    if DATE_HEADING not in text:
        text += "\n\n" + DATE_HEADING + "\n\n" + """- Supersedes the historical keep-as-is wording for issues 10/11: retain extra post-delivery screening, but fix its package anchor and require package completion before pickup of pre-grace reservations.
- Both heuristic wrappers share one runner. Their current default difference is the Voronoi destination filter; the legacy optional HEUR-only soft pickup cap remains disabled.
- HEUR_VOR prospectively excludes the current package's cell, including coincident deliveries. Busy pickup/passenger states do not update grace. Direct request-identity acceptance avoids index mismatch between the fixed window reference and the environment's default reference.
- No full FOUR_ZONE rewrite: its no-post-window and destination-zone routing would change the retained heuristic. DRL mechanics and deferred reward/objective/horizon decisions remain unchanged.
- Regression tests and validation evidence: archive/switching_update_2026-10-29. Stored figures/results are historical; this change requires new baseline evaluation before comparing corrected ratios.
"""
    path.write_text(text, encoding="utf-8")

    path = root / "progress_log.md"
    text = path.read_text(encoding="utf-8")
    if DATE_HEADING not in text:
        text += "\n\n" + DATE_HEADING + "\n\n" + """- Implemented the user's approved HEUR/HEUR_VOR changes in both stationary and non-stationary notebooks. Retained the existing one-step pre-grace trigger, full post-delivery grace reset, and movement toward the next package when unmatched.
- Fixed each screening window to its associated package address; completing a subsequent package starts that package's window. Pre-grace requests are accepted once, held beyond their display TTL, and served only after the locked package finishes, using the existing reservation APIs.
- Shared one switching runner across both wrappers. HEUR_VOR also excludes the soon-to-be-completed origin cell; pickup/passenger travel cannot open or consume windows. Post-window requests are committed by identity to avoid resolving an index against the next package's visible list.
- Preserved all other notebook cells, stored outputs and metadata, including current results. DRL/environment dynamics, PPO, FOUR_ZONE, reward, horizon, analysis aggregation and historical logs/checkpoints were not changed. Evaluation results for the corrected heuristics must be regenerated in a new run; do not resume old committed baseline results as though they used this version.
- Added deterministic behavior regressions and smoke checks under archive/switching_update_2026-10-29; see validation.txt for executed checks and outcomes. No full parameter sweep or PPO retraining was started.
"""
    path.write_text(text, encoding="utf-8")

    path = root / "PROJECT_HANDOFF.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text,
        "2. 保留有意设计的 pre-grace switching heuristic。",
        "2. 2026-10-29 更新：保留 one-step pre-grace 和配送后额外窗口；HEUR/HEUR_VOR 固定每轮窗口对应的包裹位置，预接单后先完成该包裹再接客，忙碌时不更新窗口。详见 progress_log/debug_log 当日记录。")
    text = replace_once(text,
        "3. 可见乘客请求从当前配送目标位置筛选距离和 TTL，而不是从车辆位置筛选；观测中相对坐标仍相对车辆。",
        "3. 可见乘客请求从策略指定的配送参考点筛选距离和 TTL，而不是从车辆位置筛选：DRL 仍用最近未完成包裹，HEUR/HEUR_VOR 固定为当前窗口对应包裹，FOUR_ZONE 用当前区域最后配送点；观测中相对坐标仍相对车辆。")
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if not (root / "experiment_support.py").is_file():
        raise ValueError("Expected the research project's root")
    results = [edit_notebook(root, relative)
               for relative in ("experiment.ipynb", "NonStationary/experiment2.ipynb")]
    edit_docs(root)
    # The archived regression loader names notebook dependencies explicitly.
    loader = root / "archive" / "validation_2026-09-21" / "tests" / "test_b1_b5.py"
    if loader.exists():
        text = loader.read_text(encoding="utf-8")
        text = replace_once(text,
            '"baseline_nearby_rule", "baseline_nearby_rule_voronoi",',
            '"_baseline_switching", "baseline_nearby_rule", "baseline_nearby_rule_voronoi",')
        loader.write_text(text, encoding="utf-8")
    print(json.dumps({"root": str(root), "notebooks": results}, indent=2))


if __name__ == "__main__":
    main()
