"""Shared rollout state and parameter-equivalence accounting for the notebooks."""
from dataclasses import dataclass
import itertools
import json
import math
import warnings

import pandas as pd


@dataclass
class RolloutState:
    """Owned by one training run; never shared with evaluation or another environment."""
    env: object = None
    obs: object = None
    mask: object = None
    episode_return: float = 0.0
    episode_steps: int = 0


SWEEP_METADATA_COLUMNS = [
    "run_id",
    "train_seed", "replicate", "ALPHA_EFFECTIVE", "ALPHA_SATURATION", "ALPHA_MEMBERS",
    "R", "V", "DT", "RIDE_TTL_STEPS", "RP", "HORIZON_MIN", "DEMAND_PROFILE",
    "TRAIN_UPDATES", "REPORT_UNSCALED", "OBS_DIM",
    "R_PAPER", "R_PICK_ALPHA_PAPER", "PICKUP_RADIUS",
    "ALPHA_EFFECTIVE_PAPER", "ALPHA_SATURATION_PAPER",
]

# Keep independent sweep runs separate; paired seeds within one run remain replicates.
SCENARIO_COLUMNS = [
    "run_id",
    "algo", "LAMBDA", "GAMMA_PACK", "RIDE_TTL_MIN", "MAX_VISIBLE_RIDES",
    "SWITCH_GRACE_STEPS", "RT", "R", "V", "DT", "RIDE_TTL_STEPS", "RP",
    "HORIZON_MIN", "DEMAND_PROFILE", "TRAIN_UPDATES", "REPORT_UNSCALED", "OBS_DIM",
]


def alpha_metadata(alpha, *, R, v, dt, ttl_minutes):
    """Record code/paper conventions without changing the physical pickup radius.

    R is the code's L1 radius; R_PAPER is the edge length of the same diamond.
    Paper alpha = code alpha / sqrt(2). Effective/saturation values are radius
    upper bounds from the retained ETA rule, not the paper's arrival-rate model.
    Each request still uses its own remaining TTL and the existing boundary rules.
    """
    alpha, R, v, dt, ttl_minutes = map(float, (alpha, R, v, dt, ttl_minutes))
    if not all(math.isfinite(x) for x in (alpha, R, v, dt, ttl_minutes)):
        raise ValueError("Alpha, geometry and TTL must be finite.")
    if alpha < 0 or R <= 0 or v <= 0 or dt <= 0 or ttl_minutes < 0:
        raise ValueError("Require alpha/TTL >= 0 and R/V/DT > 0.")
    ttl_steps = max(1, int(round(ttl_minutes / dt)))  # Same discretization as CoModalEnv.
    saturation = v * (ttl_steps * dt) * math.sqrt(2.0) / R
    # Compare physical radii, matching _visible_rides. Do not merge nearby unsaturated values.
    effective = saturation if alpha * R / math.sqrt(2.0) >= v * ttl_steps * dt else alpha
    return {"ALPHA_EFFECTIVE": effective, "ALPHA_SATURATION": saturation,
            "RIDE_TTL_STEPS": ttl_steps,
            "R_PAPER": R * math.sqrt(2.0), "R_PICK_ALPHA_PAPER": alpha / math.sqrt(2.0),
            "PICKUP_RADIUS": alpha * R / math.sqrt(2.0),
            "ALPHA_EFFECTIVE_PAPER": effective / math.sqrt(2.0),
            "ALPHA_SATURATION_PAPER": saturation / math.sqrt(2.0)}


def build_sweep_plan(lambda_values, alpha_values, gamma_values, ttl_values,
                     max_visible_values, grace_values, rt_values, *, R, v, dt, train_seeds):
    """One representative per equivalent alpha, with the same seed list for every setting."""
    seeds = list(train_seeds)
    if not seeds or any(isinstance(s, bool) or int(s) != s or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("train_seeds must contain distinct integers in [0, 2**32).")
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate train_seeds would count the same run twice.")
    alphas = sorted(set(float(a) for a in alpha_values))
    rows = []
    for lam, gamma, ttl, visible, grace, rt in itertools.product(
            lambda_values, gamma_values, ttl_values, max_visible_values, grace_values, rt_values):
        groups = {}
        for alpha in alphas:
            meta = alpha_metadata(alpha, R=R, v=v, dt=dt, ttl_minutes=ttl)
            groups.setdefault(meta["ALPHA_EFFECTIVE"], []).append(alpha)
        for members in groups.values():
            representative = members[0]
            meta = alpha_metadata(representative, R=R, v=v, dt=dt, ttl_minutes=ttl)
            for replicate, seed in enumerate(seeds, 1):
                rows.append(dict(combo_id=len(rows) + 1, LAMBDA=lam, GAMMA_PACK=gamma,
                                 R_PICK_ALPHA=representative, RIDE_TTL_MIN=ttl,
                                 MAX_VISIBLE_RIDES=visible, SWITCH_GRACE_STEPS=grace, RT=rt,
                                 R=float(R), V=float(v), DT=float(dt), train_seed=seed,
                                 replicate=replicate, ALPHA_MEMBERS=json.dumps(members), **meta))
    return rows


def with_effective_alpha(frame, *, legacy_geometry=None):
    """Keep nominal alpha; derive equivalence from each row's recorded physical parameters.

    Old CSVs lack R/V/DT. Callers must explicitly provide their historical geometry.
    New CSV metadata always takes precedence, including in a mixed old/new table.
    """
    result = frame.copy()
    if result.empty:
        for col in ("ALPHA_EFFECTIVE", "ALPHA_SATURATION", "RIDE_TTL_STEPS", "R_PAPER",
                    "R_PICK_ALPHA_PAPER", "PICKUP_RADIUS", "ALPHA_EFFECTIVE_PAPER",
                    "ALPHA_SATURATION_PAPER"):
            result[col] = pd.Series(dtype=float)
        return result
    for col in ("R", "V", "DT"):
        missing = result[col].isna() if col in result else pd.Series(True, index=result.index)
        if missing.any():
            if legacy_geometry is None or col not in legacy_geometry:
                raise ValueError(f"Missing {col}: provide legacy_geometry for this historical CSV.")
            warnings.warn(f"Historical rows lack {col}; using explicit legacy_geometry[{col}]="
                          f"{legacy_geometry[col]}. New results use their recorded parameters.",
                          UserWarning, stacklevel=2)
            if col not in result:
                result[col] = float(legacy_geometry[col])
            else:
                result[col] = result[col].fillna(float(legacy_geometry[col]))
    required = ["R_PICK_ALPHA", "RIDE_TTL_MIN", "R", "V", "DT"]
    missing = [c for c in required if c not in result]
    if missing:
        raise ValueError(f"Missing parameter columns for alpha equivalence: {missing}")
    metadata = [alpha_metadata(a, ttl_minutes=t, R=r, v=v, dt=dt)
                for a, t, r, v, dt in result[required].itertuples(index=False, name=None)]
    for col in metadata[0]:
        result[col] = [m[col] for m in metadata]
    return result


def summarize_alpha_groups(frame, *, rate_col="rate", legacy_geometry=None):
    """Mean and sample SD across run rows, within each otherwise identical scenario.

    n_runs is not the number of evaluation episodes; fixed-seed baselines may repeat.
    """
    data = with_effective_alpha(frame, legacy_geometry=legacy_geometry)
    context = [c for c in SCENARIO_COLUMNS if c in data]
    keys = context + ["ALPHA_EFFECTIVE"]
    summary = (data.groupby(keys, dropna=False)[rate_col]
               .agg(rate_mean="mean", rate_std="std", n_runs="count").reset_index())
    # A saturated group can contain several nominal radii; report its common
    # effective alpha, not an invented single nominal paper alpha for the group.
    summary["ALPHA_EFFECTIVE_PAPER"] = summary["ALPHA_EFFECTIVE"] / math.sqrt(2.0)
    if "R" in summary:
        summary["R_PAPER"] = summary["R"] * math.sqrt(2.0)
    return summary


def best_equivalent_alpha_rate(frame, *, rate_col="rate", legacy_geometry=None):
    """Optimize group means separately for each fixed scenario, then average scenarios.

    Broad overview plots may intentionally mix RT or other settings. These are kept
    separate during alpha selection and receive equal weight in the final overview.
    Pure-delivery averaging is handled by the caller, preserving issue 17's rule.
    """
    if frame.empty:
        return float("nan")
    summary = summarize_alpha_groups(frame, rate_col=rate_col, legacy_geometry=legacy_geometry)
    context = [c for c in SCENARIO_COLUMNS if c in summary]
    if context:
        best = summary.groupby(context, dropna=False)["rate_mean"].max()
        return float(best.mean())
    return float(summary["rate_mean"].max())
