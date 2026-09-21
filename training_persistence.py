"""Atomic sweep results, resumable PPO checkpoints and optional progress displays.

Checkpoints contain data, never notebook-defined classes/functions. Load only this
project's own checkpoints: torch.load uses pickle for NumPy and RNG state.
"""
import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import uuid

import numpy as np
import pandas as pd

from experiment_support import RolloutState, summarize_alpha_groups

FORMAT_VERSION = 2
SETTING_NAMES = (
    "HORIZON_MIN", "REWARD_SCALE", "INV_REWARD_SCALE", "REPORT_UNSCALED",
    "K_NEAREST_PACK", "DISCOUNT", "PPO_STEPS", "PPO_MINI_BATCH", "PPO_EPOCHS",
    "CLIP_EPS", "VF_COEF", "ENT_COEF", "LR", "GAE_LAMBDA", "EVAL_EPISODES",
    "HEUR_SOFT_PICK_CAP",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def training_settings(namespace):
    """Settings which must agree when continuing an existing training run."""
    settings = {k: namespace[k] for k in SETTING_NAMES if k in namespace}
    if "HOURLY_MULTIPLIER" in namespace:
        settings["HOURLY_MULTIPLIER"] = namespace["HOURLY_MULTIPLIER"]
    return _jsonable(settings)


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]


def log_run_id(path):
    """Low-level logger fallback; normal trainers always supply their run identity."""
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        import csv
        with path.open(newline="", encoding="utf-8") as stream:
            identities = {r.get("run_id") for r in csv.DictReader(stream)} - {None, ""}
        if len(identities) > 1:
            raise ValueError("Multiple runs in this log; supply run_id explicitly")
        if identities:
            return identities.pop()
    return new_run_id()


def atomic_write(path, writer):
    """Replace one completed file; a failed write leaves its previous version intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, data):
    atomic_write(path, lambda stream: stream.write(
        json.dumps(_jsonable(data), ensure_ascii=False, indent=2).encode("utf-8")))


def atomic_csv(path, frame):
    atomic_write(path, lambda stream: stream.write(frame.to_csv(index=False).encode("utf-8")))


def _pack(value):
    if isinstance(value, np.random.Generator):
        return {"__checkpoint_type__": "rng", "state": copy.deepcopy(value.bit_generator.state)}
    if is_dataclass(value) and type(value).__name__ == "RideReq":
        return {"__checkpoint_type__": "ride", "fields": {f.name: _pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_pack(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_pack(v) for v in value)
    if callable(value):
        raise TypeError("Checkpointing an environment with instance-level function overrides is unsupported")
    return copy.deepcopy(value)


def _unpack(value, ride_type):
    if isinstance(value, dict):
        tag = value.get("__checkpoint_type__")
        if tag == "rng":
            state = value["state"]
            bitgen = getattr(np.random, state["bit_generator"])()
            bitgen.state = state
            return np.random.Generator(bitgen)
        if tag == "ride":
            return ride_type(**{k: _unpack(v, ride_type) for k, v in value["fields"].items()})
        return {k: _unpack(v, ride_type) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v, ride_type) for v in value]
    if isinstance(value, tuple):
        return tuple(_unpack(v, ride_type) for v in value)
    return copy.deepcopy(value)


class TrainingCheckpoint:
    def __init__(self, directory, env, settings, meta, *, resume=False):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latest.pt"
        self.log_path = self.directory / "training_log.csv"
        identity_path = self.directory / "run.json"
        if not resume and (self.path.exists() or self.log_path.exists()):
            raise FileExistsError(f"Existing training artifacts: {self.directory}; use resume=True or a new directory")
        if resume and not self.path.is_file():
            raise FileNotFoundError(f"No checkpoint at {self.path}")
        self.meta = dict(meta or {})
        if resume:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))["run_id"]
            if self.meta.get("run_id", identity) != identity:
                raise ValueError("Checkpoint run_id differs from the requested run")
        else:
            identity = self.meta.get("run_id") or new_run_id()
            atomic_json(identity_path, {"run_id": identity})
        self.run_id = self.meta["run_id"] = identity
        keys = ("R", "v", "dt", "lam", "gamma", "rp", "rt", "r_pick", "max_visible",
                "k_pack", "ride_ttl_steps", "hourly_multiplier", "n_packages_fixed")
        self.signature = _jsonable(dict(
            environment=type(env).__name__, obs_dim=env.obs_dim, act_dim=env.act_dim,
            parameters={k: getattr(env, k) for k in keys if hasattr(env, k)},
            settings=settings, meta=self.meta))

    def save(self, policy, optimizer, env, rollout, namespace, *, update, last_eval,
             metrics, final=False):
        import torch
        log = self.log_path.read_bytes() if self.log_path.exists() else b""
        payload = dict(
            format_version=FORMAT_VERSION, run_id=self.run_id, signature=self.signature, update=update,
            last_eval=last_eval, metrics=metrics, policy=policy.state_dict(),
            optimizer=optimizer.state_dict(), environment=_pack(env.__dict__),
            rollout=_pack({k: v for k, v in vars(rollout).items() if k != "env"}),
            rollout_active=rollout.env is not None,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            notebook_rng=_pack(namespace.get("rng")), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            log_size=len(log), log_sha256=hashlib.sha256(log).hexdigest(),
        )
        atomic_write(self.path, lambda stream: torch.save(payload, stream))
        if final:
            atomic_write(self.directory / "final.pt", lambda stream: torch.save(payload, stream))

    def load(self, policy, optimizer, env, namespace, ride_type):
        import torch
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != FORMAT_VERSION or payload.get("signature") != self.signature:
            raise ValueError("Checkpoint configuration differs; restore the original settings or start a new run")
        log = self.log_path.read_bytes() if self.log_path.exists() else b""
        prefix = log[:payload["log_size"]]
        if len(prefix) != payload["log_size"] or hashlib.sha256(prefix).hexdigest() != payload["log_sha256"]:
            raise ValueError("Checkpoint training log is missing or modified; preserve the checkpoint and its log together")
        # Discard only uncheckpointed records from this checkpoint's own log.
        if len(log) != len(prefix):
            atomic_write(self.log_path, lambda stream: stream.write(prefix))
        policy.load_state_dict(payload["policy"])
        optimizer.load_state_dict(payload["optimizer"])
        restored = _unpack(payload["environment"], ride_type)
        env.__dict__.clear()
        env.__dict__.update(restored)
        rollout = RolloutState(env=env if payload["rollout_active"] else None,
                               **_unpack(payload["rollout"], ride_type))
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        if payload["notebook_rng"] is not None:
            namespace["rng"] = _unpack(payload["notebook_rng"], ride_type)
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"] is not None and torch.cuda.is_available():
            if len(payload["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("Checkpoint CUDA device count differs from this machine")
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return payload["update"], payload["last_eval"], payload["metrics"], rollout


class TrainingProgress:
    """One updating bar plus an on-disk status; no widget dependency required."""
    def __init__(self, total, *, initial=0, enabled=True, path=None, callback=None, label="PPO", run_id=None):
        self.total, self.initial = total, initial
        self.started = time.monotonic()
        self.path, self.callback = path, callback
        self.run_id = run_id
        self.enabled, self.bar = enabled, None
        if enabled:
            try:
                from tqdm.auto import tqdm
                self.bar = tqdm(total=total, initial=initial, desc=label, unit="update", leave=True, file=sys.stdout)
            except ImportError:
                pass

    def report(self, update, phase, *, saved_update=None):
        elapsed = time.monotonic() - self.started
        completed = max(0, update - self.initial)
        status = dict(update=update, total_updates=self.total, phase=phase,
                      percent=100 * update / self.total if self.total else 100.,
                      elapsed_seconds=elapsed,
                      eta_seconds=elapsed / completed * (self.total - update) if completed else None,
                      updated_at=datetime.now(timezone.utc).isoformat())
        if self.run_id is not None:
            status["run_id"] = self.run_id
        if saved_update is not None:
            status["saved_update"] = saved_update
        if self.path:
            atomic_json(self.path, status)
        if self.callback:
            self.callback(status)
        if self.bar is not None:
            self.bar.update(update - self.bar.n)
            self.bar.set_postfix_str(phase, refresh=True)
        elif self.enabled:
            filled = int(status["percent"] / 5)
            eta = "?" if status["eta_seconds"] is None else f"{status['eta_seconds']:.0f}s"
            print(f"[PPO {'#' * filled}{'-' * (20-filled)}] {update}/{self.total} "
                  f"{status['percent']:.0f}% | {phase} | elapsed {elapsed:.0f}s | ETA {eta}")

    def close(self):
        if self.bar is not None:
            self.bar.close()


class SweepStore:
    """Per-group result.json is the commit record; CSVs can always be rebuilt."""
    def __init__(self, csv_path, plan, settings, *, resume_dir=None):
        self.csv_path = Path(csv_path).resolve()
        specification = _jsonable(dict(format_version=FORMAT_VERSION, plan=plan, settings=settings))
        if resume_dir is None:
            ident = new_run_id()
            self.directory = self.csv_path.parent / (self.csv_path.stem + "_runs") / ident
            self.directory.mkdir(parents=True, exist_ok=False)
            self.run_id = ident
            atomic_json(self.directory / "manifest.json", dict(specification, run_id=self.run_id))
        else:
            self.directory = Path(resume_dir).resolve()
            actual = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
            self.run_id = actual.pop("run_id", None)
            if not self.run_id:
                raise ValueError("Historical sweep has no run_id; start a new identified run")
            if actual != specification:
                raise ValueError("Sweep settings differ from the saved manifest; use the original configuration to resume")
        self.plan, self.rows, self.completed = plan, [], set()
        self.progress = {}
        for setting in plan:
            path = self.combo_dir(setting["combo_id"]) / "result.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if len(result) != 6 or any(row["combo_id"] != setting["combo_id"] or
                                            row.get("run_id") != self.run_id for row in result):
                    raise ValueError(f"Invalid group result: {path}")
                self.rows.extend(result)
                self.completed.add(setting["combo_id"])
        if self.rows:
            self.export()
        self.status("ready")

    def combo_dir(self, combo_id):
        return self.directory / f"combo_{combo_id:04d}"

    def status(self, phase, **extra):
        self.progress.update(dict(
            phase=phase, run_id=self.run_id, completed_groups=len(self.completed), total_groups=len(self.plan),
            run_directory=str(self.directory), updated_at=datetime.now(timezone.utc).isoformat(), **extra))
        atomic_json(self.directory / "progress.json", self.progress)

    def commit(self, combo_id, rows):
        if combo_id in self.completed:
            raise ValueError("This group is already committed")
        if any(row.get("run_id", self.run_id) != self.run_id for row in rows):
            raise ValueError("Result rows belong to a different run")
        rows = [dict(row, run_id=self.run_id) for row in rows]
        atomic_json(self.combo_dir(combo_id) / "result.json", rows)
        self.rows.extend(rows)
        self.completed.add(combo_id)
        self.export()
        self.status("group_saved", combo_id=combo_id)

    def export(self):
        frame = pd.DataFrame(self.rows)
        summary = summarize_alpha_groups(frame)
        # The run folder is authoritative; the requested CSV remains a convenience
        # export for existing analysis notebooks. Each file is individually atomic.
        atomic_csv(self.directory / "results.csv", frame)
        atomic_csv(self.directory / "alpha_summary.csv", summary)
        atomic_csv(self.csv_path, frame)
        atomic_csv(self.csv_path.with_name(self.csv_path.stem + "_alpha_summary.csv"), summary)
        return frame
