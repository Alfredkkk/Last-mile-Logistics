"""Project-rooted paths and explicit single-run convergence-log selection."""
import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


class ProjectPaths:
    def __init__(self, scenario, root=None):
        if scenario not in ("stationary", "nonstationary"):
            raise ValueError("scenario must be stationary or nonstationary")
        self.scenario = scenario
        self.root = Path(root).resolve() if root is not None else PROJECT_ROOT
        self.scenario_dir = self.root / "NonStationary" if scenario == "nonstationary" else self.root
        self.results_dir = self.scenario_dir / "Results"

    def results(self, *parts):
        return self.results_dir.joinpath(*parts)

    def result_file(self, path):
        """Relative result paths are anchored to this scenario, independent of cwd."""
        path = Path(path).expanduser()
        if path.is_absolute():
            return path.resolve()
        parts = path.parts
        if parts[:2] == ("NonStationary", "Results"):
            if self.scenario != "nonstationary":
                raise ValueError("A stationary result cannot use a NonStationary/Results relative path")
            parts = parts[2:]
        elif parts[:1] == ("Results",):
            parts = parts[1:]
        return self.results(*parts).resolve()

    def output_file(self, path):
        path = self.result_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_uber_parquet_dir(self, preferred=None):
        explicit = preferred if preferred is not None else os.environ.get("UBER_PARQUET_DIR")
        if explicit is not None:
            path = Path(explicit).expanduser()
            return (path if path.is_absolute() else self.root / path).resolve()
        candidates = (self.root.parent / "Uber_NYC", self.root / "Uber_NYC",
                      self.root / "NonStationary" / "Uber_NYC")
        for path in candidates:
            if path.is_dir() and any(path.glob("fhvhv_tripdata_2021-*.parquet")):
                return path.resolve()
        # The caller can still load its cache without having raw parquet files.
        return candidates[0].resolve()


def select_training_log(csv_path, *, run_id=None, combo_id=None, run_dir=None):
    """Pick one identified run/group; never fall back to a legacy common log.

    The default is the newest run (creation timestamp in its generated name) that
    has logged data, then its highest group number with data. Explicit selectors
    never fall back to another run/group. Returns the selected path and identity.
    """
    csv_path = Path(csv_path).resolve()
    if run_dir is not None:
        chosen = Path(run_dir).expanduser()
        candidates = [(chosen if chosen.is_absolute() else csv_path.parent / chosen).resolve()]
    else:
        runs = csv_path.parent / (csv_path.stem + "_runs")
        candidates = sorted((p for p in runs.glob("*") if p.is_dir()), reverse=True)
    for directory in candidates:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest.get("run_id")
        if not identity:
            if run_dir is not None:
                raise ValueError("This historical run has no run_id; use a new identified run")
            continue
        if run_id is not None and identity != run_id:
            if run_dir is not None:
                raise ValueError("run_id does not match the selected directory")
            continue
        logs = sorted(directory.glob("combo_*/training_log.csv"),
                      key=lambda p: int(p.parent.name.removeprefix("combo_")), reverse=True)
        for path in logs:
            group = int(path.parent.name.removeprefix("combo_"))
            if combo_id is not None and group != int(combo_id):
                continue
            try:
                first = pd.read_csv(path, nrows=1)
            except pd.errors.EmptyDataError:
                continue
            if first.empty:
                continue
            if "run_id" not in first or str(first.iloc[0]["run_id"]) != identity:
                raise ValueError(f"Log run_id does not match its manifest: {path}")
            return path, identity, group
        if run_id is not None or run_dir is not None:
            break
        # Once a run with data is found, an absent explicit group must not select
        # a different experiment by accident.
        if combo_id is not None and any(p.stat().st_size for p in logs):
            break
    raise FileNotFoundError(
        f"No matching identified training log under {csv_path.parent / (csv_path.stem + '_runs')}. "
        "Run the experiment first, or set ANALYSIS_RUN_ID / ANALYSIS_COMBO_ID / ANALYSIS_RUN_DIR. "
        "Legacy common logs are kept in archive/training_logs and are not selected automatically.")


def read_run_training_log(path, *, run_id=None, combo_id=None):
    """Validate identity before plotting; a run/group/seed is one learning curve."""
    frame = pd.read_csv(path)
    required = {"run_id", "combo_id", "train_seed", "update", "kind"}
    if not required.issubset(frame):
        raise ValueError(f"Training log lacks identity columns: {sorted(required - set(frame))}")
    if frame.empty:
        raise ValueError("Training log contains no recorded updates yet")
    if frame["run_id"].isna().any():
        raise ValueError("Training log contains unidentified rows")
    if run_id is not None:
        frame = frame[frame.run_id == run_id]
    if combo_id is not None:
        frame = frame[pd.to_numeric(frame.combo_id, errors="raise") == int(combo_id)]
    if frame.empty:
        raise ValueError("No rows match the requested run/group")
    if len(frame[["run_id", "combo_id", "train_seed"]].drop_duplicates()) != 1:
        raise ValueError("Select one run_id/combo_id/train_seed before plotting a convergence curve")
    frame = frame.copy()
    frame["update"] = pd.to_numeric(frame["update"], errors="raise").astype(int)
    if frame.duplicated(["run_id", "combo_id", "train_seed", "kind", "update"]).any():
        raise ValueError("Duplicate update records found; refusing to average repeated training rows")
    return frame.sort_values(["update", "kind"]).reset_index(drop=True)
