"""Project-rooted paths and identified training-curve selection and aggregation."""
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


def read_grid_training_curve(run_dir):
    """Mean rollout rate over a run's full effective grid at common updates.

    Average seeds within each effective setting, then settings equally. Alpha
    aliases are not additional settings. Missing groups/updates are never filled
    or silently dropped from the denominator. Returns (curve, coverage).
    """
    import numpy as np

    directory = Path(run_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("run_id")
    plan = pd.DataFrame(manifest.get("plan", []))
    setting_columns = ["LAMBDA", "GAMMA_PACK", "ALPHA_EFFECTIVE", "RIDE_TTL_MIN",
                       "MAX_VISIBLE_RIDES", "SWITCH_GRACE_STEPS", "RT", "R", "V", "DT"]
    required = ["combo_id", "train_seed", *setting_columns]
    if not identity or plan.empty or not set(required).issubset(plan):
        raise ValueError("A full-grid curve requires an identified run and its complete sweep plan")
    if plan[required].isna().any().any() or plan.combo_id.duplicated().any():
        raise ValueError("Sweep plan contains missing values or duplicate group IDs")
    if plan.duplicated(setting_columns + ["train_seed"]).any():
        raise ValueError("Sweep plan repeats the same effective setting and training seed")
    plan = plan.copy()
    plan["setting_id"] = plan.groupby(setting_columns, sort=False).ngroup()
    expected = set(plan.combo_id)
    unexpected = [p.parent.name for p in directory.glob("combo_*/training_log.csv")
                  if int(p.parent.name.removeprefix("combo_")) not in expected]
    if unexpected:
        raise ValueError(f"Training logs outside the selected sweep plan: {unexpected}")

    frames, missing = [], []
    for row in plan.to_dict("records"):
        group = int(row["combo_id"])
        path = directory / f"combo_{group:04d}" / "training_log.csv"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(group)
            continue
        # Header-only logs are normal for a newly started group.
        if pd.read_csv(path, nrows=1).empty:
            missing.append(group)
            continue
        frame = read_run_training_log(path)  # validate all rows before selection
        if (frame.run_id.iloc[0] != identity or frame.combo_id.iloc[0] != group
                or frame.train_seed.iloc[0] != row["train_seed"]):
            raise ValueError(f"Training log identity does not match its sweep plan: {path}")
        for column in setting_columns:
            if column not in frame or not np.isclose(
                    pd.to_numeric(frame[column], errors="raise"), row[column],
                    rtol=1e-10, atol=1e-12).all():
                raise ValueError(f"Training log {column} does not match its sweep plan: {path}")
        expected_units = str(manifest["settings"]["REPORT_UNSCALED"]).lower()
        if ("REPORT_UNSCALED" not in frame or
                not frame.REPORT_UNSCALED.astype(str).str.lower().eq(expected_units).all()):
            raise ValueError(f"Training log reporting units do not match its run: {path}")
        train = frame.loc[frame.kind == "train", ["update", "step_avg_rate"]].copy()
        if train.empty:
            missing.append(group)
            continue
        train["step_avg_rate"] = pd.to_numeric(train.step_avg_rate, errors="raise")
        if not np.isfinite(train.step_avg_rate).all():
            raise ValueError(f"Training log contains non-finite rollout rates: {path}")
        train["combo_id"], train["setting_id"] = group, row["setting_id"]
        frames.append(train)

    coverage = dict(run_id=identity, planned_groups=len(plan), logged_groups=len(frames),
                    effective_settings=plan.setting_id.nunique(),
                    train_seeds=sorted(plan.train_seed.unique().tolist()),
                    missing_groups=missing, common_updates=0, observed_updates=0,
                    report_unscaled=manifest["settings"]["REPORT_UNSCALED"])
    curve = pd.DataFrame(columns=["update", "step_avg_rate"])
    if frames:
        training = pd.concat(frames, ignore_index=True)
        counts = training.groupby("update").combo_id.nunique()
        common = counts.index[counts == len(plan)]
        coverage.update(common_updates=len(common), observed_updates=len(counts))
        if len(common):
            per_setting = (training[training["update"].isin(common)]
                           .groupby(["update", "setting_id"]).step_avg_rate.mean())
            curve = per_setting.groupby("update").mean().reset_index()
    return curve, coverage
